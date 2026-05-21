import numpy as np
from app.pose_detector import KEYPOINT_NAMES


# 유사도 임계값: 이 값 이상이면 "올바른 자세"로 판정
GOOD_POSE_THRESHOLD = 0.80

# 키포인트별 개별 유사도 임계값 (각도 오차 허용 범위: 도 단위)
KEYPOINT_ANGLE_TOLERANCE = 25.0  # 25도 이내면 해당 관절 OK
KEYPOINT_DISTANCE_TOLERANCE = 0.75
MIN_PROCRUSTES_KEYPOINTS = 5

MIRROR_KEYPOINT_NAMES = {
    "nose": "nose",
    "left_eye": "right_eye",
    "right_eye": "left_eye",
    "left_ear": "right_ear",
    "right_ear": "left_ear",
    "left_shoulder": "right_shoulder",
    "right_shoulder": "left_shoulder",
    "left_elbow": "right_elbow",
    "right_elbow": "left_elbow",
    "left_wrist": "right_wrist",
    "right_wrist": "left_wrist",
    "left_hip": "right_hip",
    "right_hip": "left_hip",
    "left_knee": "right_knee",
    "right_knee": "left_knee",
    "left_ankle": "right_ankle",
    "right_ankle": "left_ankle",
}
MIRROR_KEYPOINT_INDICES = np.array(
    [KEYPOINT_NAMES.index(MIRROR_KEYPOINT_NAMES[name]) for name in KEYPOINT_NAMES]
)


class SimilarityCalculator:
    """
    레퍼런스 포즈(영상)와 사용자 포즈(카메라) 간의 유사도를 계산합니다.
    """

    def compute(
        self,
        ref_pose: dict,
        user_pose: dict,
    ) -> dict:
        """
        반환:
          'overall': float (0~1) 전체 유사도
          'angle_similarity': float (0~1) 각도 기반
          'coord_similarity': float (0~1) 좌표 기반
          'keypoint_status': dict[str, bool]  True=OK, False=수정 필요
          'keypoint_errors': dict[str, float] 프로크루스테스 정렬 후 좌표 거리
          'angle_errors': dict[str, float]    관절별 각도 오차(도)
          'mirror_used': bool                  좌우 반전 후보가 선택되었는지
        """
        normal_angle_sim, normal_angle_errors = self._angle_similarity(
            ref_pose["angles"], user_pose["angles"], mirror=False
        )
        normal_coord_sim, normal_keypoint_errors = self._coord_similarity(
            ref_pose.get("normalized_keypoints", ref_pose["keypoints"]),
            user_pose.get("normalized_keypoints", user_pose["keypoints"]),
            ref_pose.get("valid"),
            user_pose.get("valid"),
            mirror=False,
        )

        mirror_angle_sim, mirror_angle_errors = self._angle_similarity(
            ref_pose["angles"], user_pose["angles"], mirror=True
        )
        mirror_coord_sim, mirror_keypoint_errors = self._coord_similarity(
            ref_pose.get("normalized_keypoints", ref_pose["keypoints"]),
            user_pose.get("normalized_keypoints", user_pose["keypoints"]),
            ref_pose.get("valid"),
            user_pose.get("valid"),
            mirror=True,
        )

        normal_overall = float(np.clip(0.7 * normal_angle_sim + 0.3 * normal_coord_sim, 0.0, 1.0))
        mirror_overall = float(np.clip(0.7 * mirror_angle_sim + 0.3 * mirror_coord_sim, 0.0, 1.0))

        if mirror_overall > normal_overall:
            overall = mirror_overall
            angle_sim = mirror_angle_sim
            coord_sim = mirror_coord_sim
            angle_errors = mirror_angle_errors
            keypoint_errors = mirror_keypoint_errors
            mirror_used = True
        else:
            overall = normal_overall
            angle_sim = normal_angle_sim
            coord_sim = normal_coord_sim
            angle_errors = normal_angle_errors
            keypoint_errors = normal_keypoint_errors
            mirror_used = False

        keypoint_status = self._keypoint_status(angle_errors, keypoint_errors)

        return {
            "overall": overall,
            "angle_similarity": angle_sim,
            "coord_similarity": coord_sim,
            "keypoint_status": keypoint_status,
            "keypoint_errors": keypoint_errors,
            "angle_errors": angle_errors,
            "mirror_used": mirror_used,
        }

    def _angle_similarity(self, ref_angles: dict, user_angles: dict, mirror: bool):
        """공통 관절의 각도 오차를 계산해 0~1 유사도로 변환."""
        pairs = []
        for ref_joint in ref_angles:
            user_joint = MIRROR_KEYPOINT_NAMES[ref_joint] if mirror else ref_joint
            if user_joint in user_angles:
                pairs.append((ref_joint, user_joint))
        if not pairs:
            return 0.0, {}

        errors = {}
        sims = []
        for ref_joint, user_joint in pairs:
            err = abs(ref_angles[ref_joint] - user_angles[user_joint])
            errors[user_joint] = err
            # 각도 오차 → 유사도: 0도=1.0, 90도=0.0 (선형 감쇠)
            sim = max(0.0, 1.0 - err / 90.0)
            sims.append(sim)

        return float(np.mean(sims)), errors

    def _coord_similarity(
        self,
        ref_kp: np.ndarray,
        user_kp: np.ndarray,
        ref_valid: np.ndarray | None,
        user_valid: np.ndarray | None,
        mirror: bool,
    ) -> tuple[float, dict[str, float]]:
        """프로크루스테스 정렬 후 관절별 거리를 0~1 유사도로 변환."""
        if ref_valid is None:
            ref_valid = np.ones(len(ref_kp), dtype=bool)
        if user_valid is None:
            user_valid = np.ones(len(user_kp), dtype=bool)

        if mirror:
            user_kp = self._mirror_keypoints(user_kp)
            user_valid = user_valid[MIRROR_KEYPOINT_INDICES]

        valid = ref_valid & user_valid
        if not np.any(valid):
            return 0.0, {}

        ref_points = ref_kp[valid]
        user_points = user_kp[valid]
        if len(ref_points) >= MIN_PROCRUSTES_KEYPOINTS:
            user_points = self._procrustes_align(ref_points, user_points)

        distances = np.linalg.norm(ref_points - user_points, axis=1)
        sims = np.clip(1.0 - distances / KEYPOINT_DISTANCE_TOLERANCE, 0.0, 1.0)

        keypoint_errors = {}
        valid_indices = np.where(valid)[0]
        for idx, dist in zip(valid_indices, distances):
            display_idx = MIRROR_KEYPOINT_INDICES[idx] if mirror else idx
            keypoint_errors[KEYPOINT_NAMES[display_idx]] = float(dist)

        return float(np.mean(sims)), keypoint_errors

    @staticmethod
    def _mirror_keypoints(keypoints: np.ndarray) -> np.ndarray:
        mirrored = keypoints[MIRROR_KEYPOINT_INDICES].copy()
        mirrored[:, 0] *= -1
        return mirrored

    @staticmethod
    def _procrustes_align(ref_points: np.ndarray, user_points: np.ndarray) -> np.ndarray:
        """
        user_points를 ref_points에 가장 잘 맞도록 이동/스케일/회전 정렬합니다.
        reflection은 허용하지 않아 좌우 반전 동작을 같은 자세로 오판하지 않게 합니다.
        """
        ref_center = np.mean(ref_points, axis=0)
        user_center = np.mean(user_points, axis=0)
        ref_centered = ref_points - ref_center
        user_centered = user_points - user_center

        ref_norm = np.linalg.norm(ref_centered)
        user_norm = np.linalg.norm(user_centered)
        if ref_norm < 1e-6 or user_norm < 1e-6:
            return user_points

        ref_scaled = ref_centered / ref_norm
        user_scaled = user_centered / user_norm

        covariance = user_scaled.T @ ref_scaled
        try:
            u, _, vt = np.linalg.svd(covariance)
        except np.linalg.LinAlgError:
            return user_points

        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt

        aligned = user_scaled @ rotation
        return aligned * ref_norm + ref_center

    def _keypoint_status(self, angle_errors: dict, keypoint_errors: dict) -> dict:
        """좌표 오차와 관절 각도 오차를 함께 반영해 키포인트별 색상 상태를 만듭니다."""
        status = {
            name: (err <= KEYPOINT_DISTANCE_TOLERANCE)
            for name, err in keypoint_errors.items()
        }
        for joint, err in angle_errors.items():
            angle_ok = err <= KEYPOINT_ANGLE_TOLERANCE
            status[joint] = status.get(joint, True) and angle_ok
        return status
