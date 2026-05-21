import numpy as np
from ultralytics import YOLO


MIN_KEYPOINT_CONFIDENCE = 0.30

# YOLO-Pose 17개 키포인트 이름 (COCO 순서)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 유사도 비교에 사용할 주요 관절 쌍 (각도 계산용)
JOINT_TRIPLETS = [
    # (관절A, 꼭짓점, 관절B) → 꼭짓점의 각도 계산
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_elbow", "left_shoulder", "right_shoulder"),
    ("right_elbow", "right_shoulder", "left_shoulder"),
]


class PoseDetector:
    def __init__(self, model_path: str = "yolo26x-pose.pt"):
        """YOLO Pose 모델 로드 (없으면 자동 다운로드)"""
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        프레임에서 포즈를 감지합니다.
        반환: {
          'keypoints': np.ndarray shape (17, 2),  # 화면 기준 (x, y) 정규화 좌표
          'normalized_keypoints': np.ndarray shape (17, 2),  # 몸 기준 정규화 좌표
          'confidence': np.ndarray shape (17,),
          'valid': np.ndarray shape (17,),
          'angles': dict[str, float]              # 관절명 → 각도(도)
        }
        """
        results = self.model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            return None

        kp_data = results[0].keypoints
        if kp_data.xy is None or len(kp_data.xy) == 0:
            return None

        # 첫 번째 사람만 사용
        xy = kp_data.xy[0].cpu().numpy()        # (17, 2) 픽셀 좌표
        conf = kp_data.conf[0].cpu().numpy()    # (17,)

        h, w = frame.shape[:2]
        xy_norm = xy / np.array([w, h])         # 정규화 (0~1)
        valid = conf >= MIN_KEYPOINT_CONFIDENCE

        body_norm = self._normalize_body_keypoints(xy_norm, valid)
        angles = self._compute_angles(xy_norm, valid)

        return {
            "keypoints": xy_norm,
            "raw_keypoints": xy,
            "normalized_keypoints": body_norm,
            "confidence": conf,
            "valid": valid,
            "angles": angles,
        }

    def _compute_angles(self, kp: np.ndarray, valid: np.ndarray) -> dict:
        """관절 삼중쌍을 이용해 각도(0~180도)를 계산합니다."""
        name_to_idx = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
        angles = {}
        for a_name, vertex_name, b_name in JOINT_TRIPLETS:
            ia, iv, ib = name_to_idx[a_name], name_to_idx[vertex_name], name_to_idx[b_name]
            if not (valid[ia] and valid[iv] and valid[ib]):
                continue
            pa, pv, pb = kp[ia], kp[iv], kp[ib]
            angle = self._angle_between(pa, pv, pb)
            if angle is not None:
                angles[vertex_name] = angle
        return angles

    def _normalize_body_keypoints(self, kp: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        카메라 구도와 체격 차이를 줄이기 위해 골반 중심을 원점으로 옮기고,
        어깨/골반 폭과 몸통 길이를 섞은 값으로 스케일을 맞춥니다.
        """
        name_to_idx = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

        left_hip = name_to_idx["left_hip"]
        right_hip = name_to_idx["right_hip"]
        left_shoulder = name_to_idx["left_shoulder"]
        right_shoulder = name_to_idx["right_shoulder"]

        center_indices = [idx for idx in (left_hip, right_hip) if valid[idx]]
        if len(center_indices) < 2:
            center_indices = [idx for idx in (left_shoulder, right_shoulder) if valid[idx]]
        if center_indices:
            center = np.mean(kp[center_indices], axis=0)
        else:
            center = np.mean(kp[valid], axis=0) if np.any(valid) else np.array([0.5, 0.5])

        scales = []
        if valid[left_shoulder] and valid[right_shoulder]:
            scales.append(np.linalg.norm(kp[left_shoulder] - kp[right_shoulder]))
        if valid[left_hip] and valid[right_hip]:
            scales.append(np.linalg.norm(kp[left_hip] - kp[right_hip]))
        if valid[left_shoulder] and valid[left_hip]:
            scales.append(np.linalg.norm(kp[left_shoulder] - kp[left_hip]))
        if valid[right_shoulder] and valid[right_hip]:
            scales.append(np.linalg.norm(kp[right_shoulder] - kp[right_hip]))

        scale = float(np.mean(scales)) if scales else 1.0
        if scale < 1e-6:
            scale = 1.0

        normalized = (kp - center) / scale
        normalized[~valid] = 0.0
        return normalized

    @staticmethod
    def _angle_between(pa, pv, pb) -> float | None:
        """pv를 꼭짓점으로 pa-pv-pb 각도를 반환 (도 단위)."""
        va = pa - pv
        vb = pb - pv
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return None
        cos_theta = np.dot(va, vb) / (norm_a * norm_b)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))
