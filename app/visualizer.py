import cv2
import numpy as np
from app.pose_detector import KEYPOINT_NAMES

# COCO 스켈레톤 연결 쌍 (인덱스 기준)
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # 머리
    (5, 6),                                  # 어깨
    (5, 7), (7, 9),                          # 왼쪽 팔
    (6, 8), (8, 10),                         # 오른쪽 팔
    (5, 11), (6, 12),                        # 몸통
    (11, 12),                                # 골반
    (11, 13), (13, 15),                      # 왼쪽 다리
    (12, 14), (14, 16),                      # 오른쪽 다리
]

COLOR_GOOD = (0, 220, 80)      # 초록: 올바른 자세 (BGR)
COLOR_BAD = (30, 30, 220)       # 빨강: 수정 필요 (BGR)
COLOR_NEUTRAL = (180, 180, 180) # 회색: 상태 미결정


class Visualizer:
    def draw_pose(
        self,
        frame: np.ndarray,
        pose: dict,
        keypoint_status: dict | None = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        frame에 키포인트와 스켈레톤을 그립니다.
        keypoint_status: {관절명: True/False} 없으면 회색으로 표시
        """
        kp = pose["keypoints"]          # (17, 2) 정규화 좌표
        conf = pose["confidence"]       # (17,)
        h, w = frame.shape[:2]

        # 픽셀 좌표로 변환
        pts = (kp * np.array([w, h])).astype(int)

        name_to_idx = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

        # 관절별 색상 결정
        def get_color(joint_name: str) -> tuple:
            if keypoint_status is None:
                return COLOR_NEUTRAL
            status = keypoint_status.get(joint_name)
            if status is None:
                return COLOR_NEUTRAL
            return COLOR_GOOD if status else COLOR_BAD

        # 스켈레톤 선 그리기
        for i, j in SKELETON_PAIRS:
            if conf[i] < 0.3 or conf[j] < 0.3:
                continue
            name_i = KEYPOINT_NAMES[i]
            name_j = KEYPOINT_NAMES[j]
            c_i = get_color(name_i)
            c_j = get_color(name_j)
            # 두 관절 모두 OK일 때만 초록선, 하나라도 NG면 빨강
            if c_i == COLOR_GOOD and c_j == COLOR_GOOD:
                line_color = COLOR_GOOD
            elif c_i == COLOR_NEUTRAL or c_j == COLOR_NEUTRAL:
                line_color = COLOR_NEUTRAL
            else:
                line_color = COLOR_BAD
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), line_color, 2)

        # 키포인트 원 그리기
        for idx, name in enumerate(KEYPOINT_NAMES):
            if conf[idx] < 0.3:
                continue
            color = get_color(name)
            cv2.circle(frame, tuple(pts[idx]), 6, color, -1)
            cv2.circle(frame, tuple(pts[idx]), 6, (255, 255, 255), 1)

        return frame

    def draw_similarity_hud(
        self,
        frame: np.ndarray,
        similarity: dict,
    ) -> np.ndarray:
        """화면 상단에 유사도 HUD를 오버레이합니다."""
        overall = similarity["overall"]
        angle_sim = similarity["angle_similarity"]
        coord_sim = similarity["coord_similarity"]
        mirror_mode = "MIRRORED" if similarity.get("mirror_used") else "NORMAL"

        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 112), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        bar_color = COLOR_GOOD if overall >= 0.80 else COLOR_BAD

        # 전체 유사도 바
        bar_w = int(overall * 300)
        cv2.rectangle(frame, (10, 10), (310, 28), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 10), (10 + bar_w, 28), bar_color, -1)
        cv2.putText(frame, f"Overall: {overall*100:.1f}%",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(frame, f"Angle: {angle_sim*100:.1f}%  Coord: {coord_sim*100:.1f}%",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame, f"Compare: {mirror_mode}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        status_text = "GOOD POSE!" if overall >= 0.80 else "Adjust your pose"
        status_color = COLOR_GOOD if overall >= 0.80 else COLOR_BAD
        cv2.putText(frame, status_text,
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return frame

    def draw_reference_thumbnail(
        self,
        canvas: np.ndarray,
        ref_frame: np.ndarray,
        x: int, y: int,
        thumb_w: int = 200,
    ) -> np.ndarray:
        """캔버스 우측에 레퍼런스 영상 썸네일을 그립니다."""
        h, w = ref_frame.shape[:2]
        thumb_h = int(thumb_w * h / w)
        thumb = cv2.resize(ref_frame, (thumb_w, thumb_h))
        canvas_h, canvas_w = canvas.shape[:2]
        x = max(0, min(x, canvas_w - thumb_w))
        y = max(0, min(y, canvas_h - thumb_h))
        canvas[y:y+thumb_h, x:x+thumb_w] = thumb
        cv2.rectangle(canvas, (x, y), (x+thumb_w, y+thumb_h), (255, 255, 255), 1)
        cv2.putText(canvas, "Reference", (x+4, y+thumb_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        return canvas