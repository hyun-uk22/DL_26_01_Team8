# Push-up Benchmark

`pushup_videos` 폴더에 넣은 mp4 팔굽혀펴기 영상들을 이용해 Procrustes, DTW, EMA + One Euro Filter 적용 전후를 pairwise로 비교합니다.

## 입력 데이터

기본 입력 폴더:

```text
pushup_videos/
├── pushup_01.mp4
├── pushup_02.mp4
└── ...
```

영상 전체가 아니라 팔굽혀펴기 구간만 평가하고 싶으면 `pushup_videos/manifest.csv`를 만듭니다.

```csv
filename,start_sec,end_sec,label
pushup_01.mp4,3.0,10.5,good
pushup_02.mp4,1.2,8.0,good
pushup_03.mp4,2.0,9.5,bad
```

`label`은 선택 항목입니다. 정답 정확도 계산에는 쓰지 않고, 결과를 good-good, good-bad처럼 나눠 해석할 때 사용합니다.

## 실행

```powershell
.\.venv\Scripts\python.exe benchmarks\pushup_benchmark.py
```

주요 옵션:

```powershell
.\.venv\Scripts\python.exe benchmarks\pushup_benchmark.py --sample-fps 6 --force
```

- `--sample-fps`: 초당 몇 개 pose를 추출할지 지정합니다. 기본값은 6입니다.
- `--force`: 기존 pose cache를 무시하고 다시 추출합니다.
- `--video-dir`: mp4 폴더를 바꿉니다.
- `--output-dir`: 결과 저장 폴더를 바꿉니다.

## 평가 조건

각 영상 pair에 대해 아래 8가지 조건을 모두 평가합니다.

| config | Procrustes | DTW | EMA + One Euro |
| --- | --- | --- | --- |
| `baseline` | OFF | OFF | OFF |
| `procrustes_only` | ON | OFF | OFF |
| `dtw_only` | OFF | ON | OFF |
| `filter_only` | OFF | OFF | ON |
| `procrustes_dtw` | ON | ON | OFF |
| `procrustes_filter` | ON | OFF | ON |
| `dtw_filter` | OFF | ON | ON |
| `all_enabled` | ON | ON | ON |

10개 영상이면 `10C2 = 45`개 조합이고, 방향을 바꾼 비교까지 수행해 config당 90개 pair row가 생성됩니다.

## 결과 파일

결과는 기본적으로 `benchmark_outputs/pushup`에 저장됩니다.

| 파일 | 의미 |
| --- | --- |
| `pose_cache/*.pkl` | 영상별 추출 pose cache |
| `video_pose_quality.csv` | 영상별 pose 추출 개수와 filter 전후 jitter 지표 |
| `pair_results.csv` | 모든 영상 pair와 config별 상세 결과 |
| `summary_by_config.csv` | config별 평균 성능 요약 |
| `technique_effects.csv` | baseline 대비 단독/전체 기법 개선량 |
| `procrustes_based_effects.csv` | 카메라 각도 차이가 큰 데이터에 맞춘 Procrustes 기준 개선량 |

## 주요 지표 의미

| 지표 | 의미 | 해석 |
| --- | --- | --- |
| `avg_overall` | 각도 70% + 좌표 30% 최종 유사도 평균 | 높을수록 두 영상 동작이 유사 |
| `avg_coord_similarity` | keypoint 좌표 기반 유사도 평균 | Procrustes 효과 확인에 적합 |
| `avg_angle_similarity` | 관절 각도 기반 유사도 평균 | 자세 형태 유사성 확인 |
| `avg_coord_error` | 비교된 keypoint 간 평균 거리 | 낮을수록 좌표 차이가 작음 |
| `avg_angle_error_deg` | 관절 각도 평균 오차 | 낮을수록 관절 각도가 유사 |
| `std_overall` | pair 내부 frame별 overall score 표준편차 | 낮을수록 점수가 안정적 |
| `dtw_cost` | DTW 정렬 경로의 평균 feature 거리 | 낮을수록 시퀀스 정렬이 잘 맞음 |
| `status_switches_per_frame` | 초록/빨강 관절 상태 변화 빈도 | 낮을수록 피드백 깜빡임이 적음 |
| `raw_keypoint_jitter` | filter 전 연속 프레임 keypoint 이동량 | 낮을수록 검출 흔들림이 적음 |
| `filtered_keypoint_jitter` | filter 후 연속 프레임 keypoint 이동량 | filter 안정화 효과 확인 |
| `jitter_reduction_ratio` | keypoint jitter 감소율 | 양수면 filter가 흔들림을 줄임 |

## 해석 시 주의점

YouTube 또는 공개 mp4는 정답 자세 라벨이 없는 경우가 많습니다. 따라서 이 벤치마크는 모델 정확도라기보다 알고리즘 적용 전후의 안정성, 정렬 성능, 유사도 변화를 보는 실험입니다.

- Procrustes는 `avg_coord_similarity` 증가와 `avg_coord_error` 감소를 중심으로 봅니다.
- DTW는 `avg_overall` 증가와 `dtw_cost`를 중심으로 봅니다.
- EMA + One Euro Filter는 `jitter_reduction_ratio`, `std_overall`, `status_switches_per_frame` 감소를 중심으로 봅니다.

## 카메라 각도가 다른 영상에서의 권장 해석

수집한 푸쉬업 영상들의 카메라 각도가 서로 다르면, DTW와 Filter를 baseline에 바로 비교하기보다 Procrustes를 먼저 기준 보정으로 적용한 뒤 추가 효과를 보는 것이 더 적절합니다.

이 해석을 위해 `procrustes_based_effects.csv`를 별도로 생성합니다.

| comparison | 비교 | 의미 |
| --- | --- | --- |
| `camera_normalization_effect` | `baseline` -> `procrustes_only` | 카메라 각도/위치/크기 차이 보정 효과 |
| `dtw_after_procrustes` | `procrustes_only` -> `procrustes_dtw` | 카메라 보정 이후 속도 차이 보정 효과 |
| `filter_after_procrustes` | `procrustes_only` -> `procrustes_filter` | 카메라 보정 이후 keypoint 안정화 효과 |
| `dtw_filter_after_procrustes` | `procrustes_only` -> `all_enabled` | 카메라 보정 이후 DTW + Filter 종합 효과 |
| `filter_added_after_procrustes_dtw` | `procrustes_dtw` -> `all_enabled` | DTW까지 적용된 상태에서 Filter 추가 효과 |
| `dtw_added_after_procrustes_filter` | `procrustes_filter` -> `all_enabled` | Filter까지 적용된 상태에서 DTW 추가 효과 |

발표에서는 이 파일을 중심으로 다음 순서로 설명하는 것을 권장합니다.

1. 먼저 `baseline` 대비 `procrustes_only`로 카메라 구도 차이 보정 효과를 확인합니다.
2. 이후 `procrustes_only`를 기준선으로 두고 DTW와 Filter의 추가 개선량을 확인합니다.
