# Pose Coach

Keypoint 기반 실시간 홈트레이닝 자세 교정 솔루션입니다. 사용자가 선택한 레퍼런스 영상 또는 YouTube 링크에서 원하는 구간과 영역을 crop한 뒤, YOLO-Pose로 관절 keypoint를 추출합니다. 이후 사용자 카메라 또는 녹화본에서 추출한 keypoint와 비교해 자세 유사도를 정량적으로 계산하고, 관절별 색상 피드백으로 교정이 필요한 부위를 보여줍니다.

## 주요 기능

- 로컬 동영상 파일 또는 YouTube URL 입력
- 레퍼런스 영상 구간 선택 및 ROI crop
- 여러 사람이 감지된 경우 ROI crop으로 분석 대상 1명 선택
- ROI 드래그 중 선택 영역 실시간 표시
- 영상 내부 재생 버튼과 progress bar 제공
- 마우스 움직임 기반 영상 컨트롤 자동 표시/숨김
- 재생 중 구간 시작점을 조정하면 변경된 시작점부터 즉시 재생
- YOLO-Pose 기반 17개 COCO keypoint 검출
- 사용자 카메라 실시간 자세 분석
- 사용자 카메라 녹화 후 특정 구간 분석
- 각도 기반 유사도와 좌표 기반 유사도 계산
- 사용자 체격과 카메라 구도 차이를 줄이기 위한 keypoint 정규화
- 좌우 반전 후보 비교를 통한 mirror pose 보정
- 관절별 색상 피드백
  - 초록: 올바른 자세
  - 빨강: 자세 수정 필요

## 프로젝트 구조

```text
.
├── app/
│   ├── pose_detector.py   # YOLO-Pose keypoint 검출 및 body normalization
│   ├── similarity.py      # 각도/좌표 기반 유사도 계산
│   ├── ui.py              # Tkinter 기반 데스크톱 UI
│   ├── video_loader.py    # 로컬/YouTube 영상 로딩
│   └── visualizer.py      # skeleton, keypoint, HUD 시각화
├── main.py                # 앱 실행 진입점
├── pyproject.toml         # 프로젝트 의존성 정의
├── requirements.txt       # 현재 가상환경 기준 패키지 목록
└── README.md
```

## 실행 환경

- Python 3.12 이상
- Windows 권장
- 웹캠
- YOLO pose 모델 가중치 파일

이 프로젝트는 기본적으로 `yolo26x-pose.pt` 모델 파일을 사용합니다. 로컬에 모델 파일이 없으면 Ultralytics가 다운로드를 시도할 수 있으나, 안정적인 실행을 위해 프로젝트 루트에 모델 파일을 미리 준비하는 것을 권장합니다. 모델 가중치(`*.pt`)는 용량이 커서 Git에 포함하지 않습니다.

```text
D:\project\dlproj\yolo26x-pose.pt
```

## 설치

이미 `.venv`를 만들어둔 경우 가상환경을 활성화한 뒤 의존성을 설치합니다.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pyproject.toml` 기반으로 설치하려면 다음 명령을 사용할 수 있습니다.

```powershell
pip install -e .
```

CUDA 버전의 PyTorch를 사용하는 경우 환경에 맞는 PyTorch 설치 명령이 필요할 수 있습니다. `requirements.txt`에는 현재 개발 환경 기준 패키지 버전이 고정되어 있습니다.

## 실행

```powershell
python main.py
```

실행 후 앱에서 다음 순서로 사용할 수 있습니다.

1. 레퍼런스 영상 파일을 선택하거나 YouTube 링크를 입력합니다.
2. 분석할 구간을 slider로 선택합니다.
3. 레퍼런스 영상에 여러 사람이 있으면 영상 위에서 분석할 사람만 포함되도록 ROI 영역을 드래그합니다.
4. ROI 안에 여러 사람이 남아 있으면 더 좁게 다시 선택합니다.
5. `포즈 추출`을 눌러 레퍼런스 keypoint를 추출합니다.
6. `분석 시작`으로 실시간 카메라 자세를 비교하거나, `카메라 녹화 시작` 후 녹화 구간을 분석합니다.

영상 내부 재생 컨트롤은 마우스를 영상 영역 위에서 움직이면 나타나고, 일정 시간 움직임이 없으면 하단으로 숨겨집니다. 레퍼런스 구간 재생 중 slider 시작점을 조정하면 변경된 시작점부터 다시 재생됩니다.

## 유사도 계산 방식

자세 유사도는 크게 두 가지 값을 조합합니다.

- 각도 유사도: 팔꿈치, 무릎, 어깨, 골반 등 주요 관절 각도 차이를 비교합니다.
- 좌표 유사도: keypoint를 골반 또는 어깨 중심 기준으로 정규화한 뒤, Procrustes alignment로 위치/스케일/회전 차이를 보정하여 비교합니다.

최종 점수는 각도 유사도 70%, 좌표 유사도 30% 비율로 계산합니다. 좌우 반전된 자세도 별도로 비교하여 더 높은 유사도 후보를 선택합니다.

## 기술 스택

- Python
- Tkinter
- OpenCV
- NumPy
- PyAV
- Pillow
- Ultralytics YOLO-Pose
- yt-dlp

## 현재 상태

이 프로젝트는 데스크톱에서 레퍼런스 영상과 사용자 카메라 영상을 비교하는 프로토타입입니다. 자세 검출 품질은 카메라 각도, 조명, 인체가 화면에 보이는 정도, YOLO 모델 성능에 영향을 받습니다.
