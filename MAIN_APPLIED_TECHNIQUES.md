# main.py 기준 적용 기법 정리

이 문서는 `main.py`를 실행했을 때 실제로 연결되는 `PoseCoachApp` 기준으로, 현재 프로젝트에 구현된 주요 기법과 처리 흐름을 정리한 문서입니다.

## 1. 전체 실행 구조

`main.py`는 앱 실행 진입점 역할만 수행합니다.

```python
from app.ui import PoseCoachApp

if __name__ == "__main__":
    app = PoseCoachApp()
    app.run()
```

실제 기능은 `app` 패키지 내부 모듈로 분리되어 있습니다.

| 모듈 | 역할 |
| --- | --- |
| `app/ui.py` | Tkinter 기반 데스크톱 UI, 전체 워크플로우 제어 |
| `app/video_loader.py` | 로컬 영상 및 YouTube 영상 로딩 |
| `app/pose_detector.py` | YOLO-Pose 기반 keypoint 검출, 각도 계산, 정규화 |
| `app/similarity.py` | 레퍼런스 포즈와 사용자 포즈 간 유사도 계산 |
| `app/visualizer.py` | skeleton, keypoint, 유사도 HUD 시각화 |

## 2. 입력 영상 처리

### 2.1 로컬 영상 입력

`VideoLoader.load_local()`을 통해 로컬 동영상 파일을 PyAV로 엽니다.

적용 기법:

- PyAV 기반 프레임 디코딩
- 영상 메타데이터 추출
  - FPS
  - 전체 프레임 수
  - 영상 길이
  - 영상 해상도
- 특정 프레임 seek
- 시작 프레임부터 끝 프레임까지 구간 프레임 순회

### 2.2 YouTube URL 입력

`VideoLoader.load_youtube()`는 `yt-dlp`를 사용해 YouTube URL에서 실제 스트리밍 URL을 추출합니다.

적용 기법:

- YouTube URL 패턴 판별
- `yt-dlp`를 통한 스트리밍 URL 추출
- 720p 이하 우선 포맷 선택
- playlist 제외 처리
- Node.js 또는 Deno 기반 JavaScript runtime 설정
- PyAV로 추출된 스트리밍 URL 열기
- 스트리밍 소스 재연결 옵션 적용

## 3. 구간 선택 및 ROI crop

### 3.1 시간 구간 선택

`RangeSlider`를 통해 레퍼런스 영상의 시작/끝 프레임을 선택합니다.

적용 기법:

- 시작 thumb / 끝 thumb을 가진 custom Tkinter Canvas slider
- 마우스 드래그로 프레임 범위 조정
- 마우스 휠로 가까운 thumb 미세 조정
- 선택된 프레임 범위만 포즈 추출 대상으로 사용
- 영상 내부 재생 컨트롤은 마우스 이동 시 표시되고, 일정 시간 움직임이 없으면 하단으로 숨김
- Tkinter 기본 위젯 제약상 실제 alpha 투명도 대신 영상 배경과 가까운 어두운 컨트롤 배경색으로 반투명에 가까운 시각 효과 적용

### 3.2 화면 영역 crop

사용자는 미리보기 화면 위에서 마우스 드래그로 ROI를 지정할 수 있습니다.

적용 기법:

- Tkinter canvas 좌표를 원본 영상 프레임 좌표로 역변환
- ROI를 `(x1, y1, x2, y2)` 정규화 비율로 저장
- 마우스 드래그 중 임시 ROI 사각형을 반투명 오버레이로 표시
- 드래그 종료 지점이 영상 표시 영역 밖으로 나가면 영상 경계로 보정해 ROI 저장
- 프레임마다 ROI 비율을 실제 픽셀 좌표로 변환
- 레퍼런스 포즈 추출 시 crop된 프레임만 YOLO-Pose에 입력
- 레퍼런스 포즈 추출 직전 현재 미리보기 프레임에서 감지 인원 수 확인
- ROI 미설정 상태에서 여러 사람이 감지되면 포즈 추출을 중단하고 ROI 선택 안내
- ROI 내부에 여러 사람이 남아 있으면 더 좁은 영역 재선택 안내

이 방식은 사용자가 영상에서 원하는 운동 동작 영역만 선택해 배경이나 불필요한 사람을 줄이는 목적입니다.
여러 사람이 등장하는 레퍼런스 영상은 분석할 사람 한 명만 들어오도록 미리보기 화면에서 드래그한 뒤 `포즈 추출`을 눌러야 합니다.

## 4. Pose Detection

### 4.1 YOLO-Pose 모델

`PoseDetector`는 Ultralytics YOLO 모델을 사용합니다.

기본 모델 경로:

```text
yolo26x-pose.pt
```

적용 기법:

- Ultralytics `YOLO` API 사용
- COCO 17개 keypoint 검출
- 한 프레임에서 첫 번째로 감지된 사람만 사용
- keypoint confidence 기반 유효 관절 필터링

confidence 기준:

```text
MIN_KEYPOINT_CONFIDENCE = 0.30
```

### 4.2 COCO 17개 Keypoint

현재 사용하는 keypoint 순서는 다음과 같습니다.

```text
nose
left_eye, right_eye
left_ear, right_ear
left_shoulder, right_shoulder
left_elbow, right_elbow
left_wrist, right_wrist
left_hip, right_hip
left_knee, right_knee
left_ankle, right_ankle
```

### 4.3 좌표 정규화

YOLO가 반환한 픽셀 좌표는 프레임 크기로 나누어 0~1 범위의 화면 기준 좌표로 변환됩니다.

```text
xy_norm = xy / [frame_width, frame_height]
```

반환 데이터에는 다음 좌표가 함께 포함됩니다.

- `raw_keypoints`: 원본 픽셀 좌표
- `keypoints`: 프레임 크기 기준 0~1 정규화 좌표
- `normalized_keypoints`: 신체 기준 정규화 좌표

## 5. 신체 기준 Keypoint 정규화

사용자와 레퍼런스 영상은 카메라 구도, 화면 내 위치, 체격, 키가 다를 수 있으므로 신체 기준 정규화를 수행합니다.

적용 기법:

- 골반 중심을 우선 기준점으로 사용
  - `left_hip`, `right_hip`
- 골반이 감지되지 않으면 어깨 중심 사용
  - `left_shoulder`, `right_shoulder`
- 둘 다 부족하면 유효 keypoint들의 평균 좌표 사용
- 기준점을 원점처럼 보고 모든 keypoint를 이동
- 신체 scale을 계산해 좌표 크기 보정

scale 계산에 사용하는 값:

- 어깨 폭
- 골반 폭
- 왼쪽 몸통 길이
- 오른쪽 몸통 길이

최종 정규화:

```text
normalized = (keypoint - body_center) / body_scale
```

유효하지 않은 keypoint는 0으로 처리합니다.

## 6. 관절 각도 계산

`PoseDetector`는 주요 관절 삼중쌍을 이용해 각도를 계산합니다.

사용되는 삼중쌍:

| 관절 A | 꼭짓점 | 관절 B | 의미 |
| --- | --- | --- | --- |
| left_shoulder | left_elbow | left_wrist | 왼쪽 팔꿈치 각도 |
| right_shoulder | right_elbow | right_wrist | 오른쪽 팔꿈치 각도 |
| left_hip | left_knee | left_ankle | 왼쪽 무릎 각도 |
| right_hip | right_knee | right_ankle | 오른쪽 무릎 각도 |
| left_shoulder | left_hip | left_knee | 왼쪽 골반/고관절 각도 |
| right_shoulder | right_hip | right_knee | 오른쪽 골반/고관절 각도 |
| left_elbow | left_shoulder | right_shoulder | 왼쪽 어깨 각도 |
| right_elbow | right_shoulder | left_shoulder | 오른쪽 어깨 각도 |

각도 계산 방식:

- 꼭짓점 기준 두 벡터 생성
- 벡터 내적 기반 cosine 계산
- `arccos`로 라디안 각도 계산
- degree 단위로 변환
- 결과 범위는 0~180도

## 7. 레퍼런스 포즈 추출

사용자가 선택한 영상 구간과 ROI를 기준으로 레퍼런스 포즈를 미리 추출합니다.

적용 기법:

- 선택 프레임 범위 순회
- `ref_sample_step = 5` 기준으로 일부 프레임만 샘플링
- ROI가 지정된 경우 crop 후 포즈 검출
- 포즈가 감지된 프레임만 저장
- 레퍼런스 재생 FPS는 원본 FPS와 sampling step을 기준으로 계산

```text
ref_playback_fps = loader.fps / ref_sample_step
```

레퍼런스 포즈는 실시간 분석과 녹화본 분석에서 비교 기준으로 사용됩니다.

## 8. 사용자 카메라 처리

### 8.1 실시간 분석

`_live_loop()`는 OpenCV `VideoCapture(0)`로 웹캠을 열고 프레임을 계속 읽습니다.

적용 기법:

- OpenCV 웹캠 입력
- 사용자 화면 좌우 반전
  - 거울 모드처럼 보이도록 `cv2.flip(frame, 1)` 적용
- 실시간 YOLO-Pose keypoint 추출
- 현재 레퍼런스 포즈와 사용자 포즈 비교
- 결과를 skeleton 색상과 HUD로 표시
- 사용자 화면과 레퍼런스 화면을 좌우 split screen으로 합성

### 8.2 녹화 후 분석

실시간 분석 외에 사용자의 카메라 영상을 mp4로 녹화한 뒤 분석할 수 있습니다.

적용 기법:

- OpenCV `VideoWriter` 기반 mp4 저장
- 임시 폴더에 녹화 파일 생성
- 녹화본을 다시 `VideoLoader`로 로드
- 녹화본에서도 시작/끝 구간 선택 가능
- 녹화 구간에서 사용자 포즈 시퀀스 추출
- 레퍼런스 포즈 시퀀스와 DTW 기반 정렬 후 분석

## 9. 유사도 계산

유사도 계산은 `SimilarityCalculator`에서 수행합니다.

최종 유사도는 각도 기반 유사도와 좌표 기반 유사도를 가중합합니다.

```text
overall = 0.7 * angle_similarity + 0.3 * coord_similarity
```

현재 기준:

```text
GOOD_POSE_THRESHOLD = 0.80
```

전체 유사도가 0.80 이상이면 올바른 자세로 판단합니다.

## 10. 각도 기반 유사도

레퍼런스 포즈와 사용자 포즈에서 공통으로 존재하는 관절 각도를 비교합니다.

적용 기법:

- 관절별 각도 오차 계산
- 0도 오차는 유사도 1.0
- 90도 이상 오차는 유사도 0.0
- 선형 감쇠 방식 적용

```text
angle_similarity = max(0, 1 - angle_error / 90)
```

관절별 색상 판정에는 다음 기준이 사용됩니다.

```text
KEYPOINT_ANGLE_TOLERANCE = 25.0
```

즉, 해당 관절의 각도 오차가 25도 이내이면 각도 측면에서 정상으로 봅니다.

## 11. 좌표 기반 유사도

좌표 기반 유사도는 신체 기준 정규화 keypoint를 비교합니다.

적용 기법:

- 유효 keypoint만 비교
- 좌우 반전 후보도 별도로 비교
- 공통 유효 keypoint가 충분할 경우 Procrustes alignment 적용
- 정렬 후 keypoint별 Euclidean distance 계산
- 거리 기반 선형 유사도 변환

거리 허용 기준:

```text
KEYPOINT_DISTANCE_TOLERANCE = 0.75
```

좌표 유사도 계산:

```text
coord_similarity = mean(clip(1 - distance / 0.75, 0, 1))
```

## 12. Procrustes Alignment

사용자와 레퍼런스의 카메라 구도, 위치, 크기 차이를 줄이기 위해 좌표 비교 전에 Procrustes alignment를 적용합니다.

적용 조건:

```text
MIN_PROCRUSTES_KEYPOINTS = 5
```

즉, 공통 유효 keypoint가 5개 이상일 때 적용됩니다.

보정하는 차이:

- 위치 차이
- scale 차이
- 회전 차이

보정하지 않는 차이:

- reflection

reflection은 허용하지 않습니다. 좌우 반전 자세를 같은 자세로 잘못 판단하지 않기 위해서입니다. 좌우 반전 여부는 별도 mirror 후보 비교에서 처리합니다.

## 13. 좌우 반전 Mirror 보정

레퍼런스 영상과 사용자 카메라는 좌우 방향이 다르게 보일 수 있습니다.

이를 보정하기 위해 유사도 계산 시 두 후보를 모두 계산합니다.

- normal 비교
- mirror 비교

mirror 비교에서는 왼쪽/오른쪽 keypoint를 교환합니다.

예:

| 원래 keypoint | mirror keypoint |
| --- | --- |
| left_shoulder | right_shoulder |
| left_elbow | right_elbow |
| left_wrist | right_wrist |
| left_hip | right_hip |
| left_knee | right_knee |
| left_ankle | right_ankle |

두 후보 중 최종 유사도가 더 높은 쪽을 선택하고, 결과에 `mirror_used` 값을 포함합니다.

## 14. DTW 기반 시퀀스 정렬

녹화본 분석에서는 레퍼런스 포즈 시퀀스와 사용자 포즈 시퀀스 길이 및 속도가 다를 수 있습니다.

이를 보정하기 위해 Dynamic Time Warping을 사용합니다.

적용 기법:

- 레퍼런스 pose feature sequence 생성
- 사용자 pose feature sequence 생성
- 사용자 mirror pose feature sequence도 생성
- normal/mirror 중 더 작은 feature distance를 사용
- window 제한을 둔 DTW 계산
- 최적 경로 `[(ref_idx, user_idx), ...]` 반환
- 경로 평균 비용을 `dtw_cost`로 표시

DTW window 기준:

```text
DTW_WINDOW_RATIO = 0.35
```

pose feature 구성:

- 정규화 keypoint 좌표
- 관절 각도 값
- keypoint valid mask

이 방식은 사용자가 레퍼런스보다 조금 빠르거나 느리게 동작해도 프레임 단위로 더 적절히 매칭하기 위한 목적입니다.

## 15. 관절별 상태 판정

관절별 상태는 좌표 오차와 각도 오차를 함께 반영합니다.

판정 기준:

- keypoint 좌표 거리 <= `KEYPOINT_DISTANCE_TOLERANCE`
- 관절 각도 오차 <= `KEYPOINT_ANGLE_TOLERANCE`

둘 중 하나라도 기준을 넘으면 해당 관절은 수정 필요로 표시됩니다.

결과 형태:

```text
keypoint_status = {
    "left_elbow": True,
    "right_knee": False,
    ...
}
```

## 16. 시각화

`Visualizer`는 OpenCV로 사용자 프레임 위에 자세 분석 결과를 그립니다.

### 16.1 Skeleton 시각화

COCO skeleton 연결 쌍을 사용해 관절을 선으로 연결합니다.

적용 기법:

- confidence 0.3 미만 keypoint는 표시하지 않음
- keypoint별 상태에 따라 색상 지정
- 연결된 두 관절이 모두 정상일 때만 초록선 표시
- 하나라도 수정 필요이면 빨간선 표시
- 상태 미결정이면 회색 표시

색상 기준:

| 상태 | 색상 | 의미 |
| --- | --- | --- |
| 정상 | 초록 | 올바른 자세 |
| 수정 필요 | 빨강 | 자세 수정 필요 |
| 미결정 | 회색 | 비교 상태 없음 |

### 16.2 유사도 HUD

화면 상단에 다음 정보를 표시합니다.

- 전체 유사도
- 각도 유사도
- 좌표 유사도
- normal/mirror 비교 모드
- GOOD POSE 또는 Adjust your pose 상태 문구
- 녹화본 분석 시 DTW cost

## 17. 화면 합성 및 UI 표시

사용자 화면과 레퍼런스 화면은 좌우 split screen으로 합성됩니다.

적용 기법:

- 두 프레임을 같은 panel 크기에 맞춤
- 비율 유지 resize
- 남는 영역은 검은 배경으로 padding
- 왼쪽 `USER`, 오른쪽 `REFERENCE` label 표시
- OpenCV BGR 이미지를 RGB로 변환
- Pillow `ImageTk.PhotoImage`로 Tkinter canvas에 표시

## 18. 비동기 처리

UI 멈춤을 줄이기 위해 오래 걸리는 작업은 별도 thread에서 실행합니다.

thread로 처리되는 작업:

- 레퍼런스 포즈 추출
- 레퍼런스 구간 재생
- 실시간 카메라 분석
- 카메라 녹화
- 녹화본 분석
- 분석 결과 재생

Tkinter UI 업데이트는 `root.after()`를 통해 메인 스레드로 위임합니다.

프레임 전달에는 `queue.Queue`를 사용합니다.

## 19. 현재 구현된 전체 워크플로우

1. 사용자가 로컬 영상 또는 YouTube URL을 입력합니다.
2. `VideoLoader`가 영상을 로드하고 FPS, 프레임 수, 길이를 계산합니다.
3. 사용자가 분석할 시간 구간을 선택합니다.
4. 필요하면 영상 미리보기에서 ROI crop 영역을 지정합니다.
5. 선택 구간의 프레임을 sampling하면서 YOLO-Pose로 레퍼런스 포즈를 추출합니다.
6. 레퍼런스 포즈마다 keypoint, confidence, 정규화 좌표, 관절 각도를 저장합니다.
7. 사용자가 실시간 분석을 시작하면 웹캠 프레임에서 사용자 포즈를 검출합니다.
8. 현재 레퍼런스 포즈와 사용자 포즈의 normal/mirror 유사도를 계산합니다.
9. 각도 유사도 70%, 좌표 유사도 30%로 최종 점수를 계산합니다.
10. 관절별 상태를 초록/빨강 skeleton으로 표시합니다.
11. 녹화본 분석에서는 사용자 포즈 시퀀스를 추출하고 DTW로 레퍼런스 시퀀스와 정렬합니다.
12. DTW 경로에 따라 프레임별 유사도를 계산하고 평균 유사도를 표시합니다.

## 20. 핵심 알고리즘 요약

| 분류 | 적용 기법 |
| --- | --- |
| 포즈 검출 | Ultralytics YOLO-Pose |
| keypoint 체계 | COCO 17 keypoints |
| 좌표 정규화 | 프레임 크기 기준 0~1 정규화 |
| 신체 정규화 | 골반/어깨 중심 이동 + 어깨/골반/몸통 scale 보정 |
| 각도 특징 | 팔꿈치, 무릎, 고관절, 어깨 중심 관절 각도 |
| 좌표 정렬 | Procrustes alignment |
| 좌우 보정 | left/right keypoint mirror 후보 비교 |
| 시퀀스 정렬 | Dynamic Time Warping |
| 최종 유사도 | 각도 70% + 좌표 30% |
| 자세 판정 | 전체 유사도 0.80 이상 GOOD POSE |
| 관절 색상 | 초록 정상, 빨강 수정 필요, 회색 미결정 |
| 영상 처리 | PyAV, OpenCV |
| YouTube 처리 | yt-dlp |
| UI | Tkinter, Pillow ImageTk |

## 21. 주요 임계값

| 상수 | 값 | 의미 |
| --- | --- | --- |
| `MIN_KEYPOINT_CONFIDENCE` | `0.30` | keypoint 유효 confidence 기준 |
| `GOOD_POSE_THRESHOLD` | `0.80` | 전체 자세 정상 판정 기준 |
| `KEYPOINT_ANGLE_TOLERANCE` | `25.0` | 관절 각도 오차 허용 범위 |
| `KEYPOINT_DISTANCE_TOLERANCE` | `0.75` | 정렬 후 좌표 거리 허용 범위 |
| `MIN_PROCRUSTES_KEYPOINTS` | `5` | Procrustes alignment 최소 keypoint 수 |
| `DTW_WINDOW_RATIO` | `0.35` | DTW 탐색 window 비율 |
| `ref_sample_step` | `5` | 레퍼런스 포즈 추출 sampling 간격 |

## 22. 현재 구현 관점의 특징

- 레퍼런스 영상과 사용자 카메라의 체격 차이를 줄이기 위한 정규화가 구현되어 있습니다.
- 카메라 구도와 위치 차이를 줄이기 위한 Procrustes alignment가 구현되어 있습니다.
- 좌우 반전 문제를 normal/mirror 후보 비교로 보정합니다.
- 실시간 비교와 녹화본 사후 분석을 모두 지원합니다.
- 녹화본 분석에서는 동작 속도 차이를 DTW로 보정합니다.
- 사용자에게는 숫자 점수뿐 아니라 관절별 색상 피드백을 제공합니다.
