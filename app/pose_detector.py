import numpy as np
from ultralytics import YOLO


MIN_KEYPOINT_CONFIDENCE = 0.30
EMA_ALPHA = 0.55
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.35
ONE_EURO_D_CUTOFF = 1.0

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
        self._filter_states: dict[str, PoseSmoothingFilter] = {}

    def reset_filter(self, stream_id: str | None = None):
        """시퀀스가 바뀔 때 temporal smoothing 상태를 초기화합니다."""
        if stream_id is None:
            self._filter_states.clear()
        else:
            self._filter_states.pop(stream_id, None)

    def count_people(self, frame: np.ndarray) -> int:
        """프레임에서 감지된 사람 수를 반환합니다."""
        results = self.model(frame, verbose=False)
        if not results or results[0].keypoints is None:
            return 0

        kp_data = results[0].keypoints
        if kp_data.xy is None:
            return 0
        return len(kp_data.xy)

    def detect(
        self,
        frame: np.ndarray,
        filter_stream: str | None = None,
        timestamp: float | None = None,
    ) -> dict | None:
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
        if filter_stream is not None:
            xy_norm = self._smooth_keypoints(filter_stream, xy_norm, valid, timestamp)

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

    def _smooth_keypoints(
        self,
        stream_id: str,
        keypoints: np.ndarray,
        valid: np.ndarray,
        timestamp: float | None,
    ) -> np.ndarray:
        state = self._filter_states.get(stream_id)
        if state is None:
            state = PoseSmoothingFilter()
            self._filter_states[stream_id] = state
        return state.apply(keypoints, valid, timestamp)

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


class PoseSmoothingFilter:
    """EMA 후 One Euro Filter를 이어 적용해 프레임 간 keypoint 떨림을 줄입니다."""

    def __init__(
        self,
        ema_alpha: float = EMA_ALPHA,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ):
        self.ema_alpha = ema_alpha
        self.ema_values: np.ndarray | None = None
        self.ema_valid: np.ndarray | None = None
        self.filters = [
            [OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(2)]
            for _ in range(len(KEYPOINT_NAMES))
        ]

    def apply(self, keypoints: np.ndarray, valid: np.ndarray, timestamp: float | None) -> np.ndarray:
        smoothed = keypoints.copy()
        if self.ema_values is None:
            self.ema_values = keypoints.copy()
            self.ema_valid = valid.copy()
        elif self.ema_valid is not None:
            for idx, is_valid in enumerate(valid):
                if not is_valid:
                    continue
                if not self.ema_valid[idx]:
                    self.ema_values[idx] = keypoints[idx]
                else:
                    self.ema_values[idx] = (
                        self.ema_alpha * keypoints[idx]
                        + (1.0 - self.ema_alpha) * self.ema_values[idx]
                    )
                self.ema_valid[idx] = True

        for kp_idx, is_valid in enumerate(valid):
            if not is_valid:
                continue
            for axis in range(2):
                smoothed[kp_idx, axis] = self.filters[kp_idx][axis].apply(
                    float(self.ema_values[kp_idx, axis]),
                    timestamp,
                )
        return smoothed


class OneEuroFilter:
    """속도가 빠를수록 덜 부드럽게 처리해 지연과 떨림을 함께 줄이는 low-pass filter."""

    def __init__(
        self,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.prev_value: float | None = None
        self.prev_derivative = 0.0
        self.prev_timestamp: float | None = None

    def apply(self, value: float, timestamp: float | None) -> float:
        if self.prev_value is None:
            self.prev_value = value
            self.prev_timestamp = timestamp
            return value

        dt = self._delta_time(timestamp)
        derivative = (value - self.prev_value) / dt
        derivative_alpha = self._alpha(dt, self.d_cutoff)
        filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self.prev_derivative
        )
        cutoff = self.min_cutoff + self.beta * abs(filtered_derivative)
        value_alpha = self._alpha(dt, cutoff)
        filtered_value = value_alpha * value + (1.0 - value_alpha) * self.prev_value

        self.prev_value = filtered_value
        self.prev_derivative = filtered_derivative
        self.prev_timestamp = timestamp
        return filtered_value

    def _delta_time(self, timestamp: float | None) -> float:
        if timestamp is None or self.prev_timestamp is None:
            return 1.0 / 30.0
        dt = timestamp - self.prev_timestamp
        if dt <= 1e-6:
            return 1.0 / 30.0
        return dt

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
