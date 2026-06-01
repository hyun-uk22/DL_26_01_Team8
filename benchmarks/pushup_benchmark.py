import argparse
import csv
import itertools
import pickle
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pose_detector import KEYPOINT_NAMES, PoseDetector
from app.similarity import (
    DTW_WINDOW_RATIO,
    KEYPOINT_DISTANCE_TOLERANCE,
    MIRROR_KEYPOINT_INDICES,
    MIRROR_KEYPOINT_NAMES,
    MIN_PROCRUSTES_KEYPOINTS,
    SimilarityCalculator,
)
from app.video_loader import VideoLoader


DEFAULT_VIDEO_DIR = Path("pushup_videos")
DEFAULT_OUTPUT_DIR = Path("benchmark_outputs") / "pushup"
DEFAULT_SAMPLE_FPS = 6.0


@dataclass(frozen=True)
class VideoSpec:
    video_id: str
    path: Path
    start_sec: float = 0.0
    end_sec: float | None = None
    label: str = "unknown"


@dataclass(frozen=True)
class EvalConfig:
    name: str
    use_filter: bool
    use_procrustes: bool
    use_dtw: bool


EVAL_CONFIGS = [
    EvalConfig("baseline", use_filter=False, use_procrustes=False, use_dtw=False),
    EvalConfig("procrustes_only", use_filter=False, use_procrustes=True, use_dtw=False),
    EvalConfig("dtw_only", use_filter=False, use_procrustes=False, use_dtw=True),
    EvalConfig("filter_only", use_filter=True, use_procrustes=False, use_dtw=False),
    EvalConfig("procrustes_dtw", use_filter=False, use_procrustes=True, use_dtw=True),
    EvalConfig("procrustes_filter", use_filter=True, use_procrustes=True, use_dtw=False),
    EvalConfig("dtw_filter", use_filter=True, use_procrustes=False, use_dtw=True),
    EvalConfig("all_enabled", use_filter=True, use_procrustes=True, use_dtw=True),
]


class BenchmarkSimilarity(SimilarityCalculator):
    def __init__(self, use_procrustes: bool):
        self.use_procrustes = use_procrustes

    def _coord_similarity(
        self,
        ref_kp: np.ndarray,
        user_kp: np.ndarray,
        ref_valid: np.ndarray | None,
        user_valid: np.ndarray | None,
        mirror: bool,
    ) -> tuple[float, dict[str, float]]:
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
        if self.use_procrustes and len(ref_points) >= MIN_PROCRUSTES_KEYPOINTS:
            user_points = self._procrustes_align(ref_points, user_points)

        distances = np.linalg.norm(ref_points - user_points, axis=1)
        sims = np.clip(1.0 - distances / KEYPOINT_DISTANCE_TOLERANCE, 0.0, 1.0)

        keypoint_errors = {}
        valid_indices = np.where(valid)[0]
        for idx, dist in zip(valid_indices, distances):
            display_idx = MIRROR_KEYPOINT_INDICES[idx] if mirror else idx
            keypoint_errors[KEYPOINT_NAMES[display_idx]] = float(dist)

        return float(np.mean(sims)), keypoint_errors


def main():
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "pose_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs = load_video_specs(video_dir)
    if len(specs) < 2:
        raise SystemExit(f"{video_dir} 안에 mp4 영상이 최소 2개 필요합니다.")

    detector = PoseDetector(args.model)
    pose_bank = {}
    quality_rows = []
    for spec in specs:
        print(f"[extract] {spec.video_id}")
        raw_poses = load_or_extract_poses(
            detector=detector,
            spec=spec,
            cache_dir=cache_dir,
            sample_fps=args.sample_fps,
            use_filter=False,
            force=args.force,
        )
        filtered_poses = load_or_extract_poses(
            detector=detector,
            spec=spec,
            cache_dir=cache_dir,
            sample_fps=args.sample_fps,
            use_filter=True,
            force=args.force,
        )
        pose_bank[(spec.video_id, False)] = raw_poses
        pose_bank[(spec.video_id, True)] = filtered_poses
        quality_rows.append(video_quality_row(spec, raw_poses, filtered_poses))

    pair_rows = []
    for left, right in itertools.combinations(specs, 2):
        for config in EVAL_CONFIGS:
            ref_poses = pose_bank[(left.video_id, config.use_filter)]
            user_poses = pose_bank[(right.video_id, config.use_filter)]
            row = evaluate_pair(left, right, ref_poses, user_poses, config)
            pair_rows.append(row)
            reverse_row = evaluate_pair(right, left, user_poses, ref_poses, config)
            pair_rows.append(reverse_row)

    summary_rows = summarize_by_config(pair_rows)
    effect_rows = summarize_effects(summary_rows)
    procrustes_effect_rows = summarize_procrustes_based_effects(summary_rows)

    write_csv(output_dir / "video_pose_quality.csv", quality_rows)
    write_csv(output_dir / "pair_results.csv", pair_rows)
    write_csv(output_dir / "summary_by_config.csv", summary_rows)
    write_csv(output_dir / "technique_effects.csv", effect_rows)
    write_csv(output_dir / "procrustes_based_effects.csv", procrustes_effect_rows)

    print(f"[done] results: {output_dir}")
    print(f"[done] pair rows: {len(pair_rows)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Push-up mp4 영상 nC2 pairwise 성능 평가를 수행합니다."
    )
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="yolo26x-pose.pt")
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--force", action="store_true", help="pose cache를 무시하고 다시 추출합니다.")
    return parser.parse_args()


def load_video_specs(video_dir: Path) -> list[VideoSpec]:
    manifest = video_dir / "manifest.csv"
    if manifest.exists():
        return load_manifest(manifest, video_dir)

    specs = []
    for path in sorted(video_dir.glob("*.mp4")):
        specs.append(VideoSpec(video_id=path.stem, path=path))
    return specs


def load_manifest(manifest: Path, video_dir: Path) -> list[VideoSpec]:
    specs = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"].strip()
            path = video_dir / filename
            start_sec = parse_float(row.get("start_sec"), 0.0)
            end_sec = parse_optional_float(row.get("end_sec"))
            label = (row.get("label") or "unknown").strip()
            specs.append(VideoSpec(video_id=path.stem, path=path, start_sec=start_sec, end_sec=end_sec, label=label))
    return specs


def load_or_extract_poses(
    detector: PoseDetector,
    spec: VideoSpec,
    cache_dir: Path,
    sample_fps: float,
    use_filter: bool,
    force: bool,
) -> list[dict]:
    cache_path = cache_dir / f"{spec.video_id}_{'filtered' if use_filter else 'raw'}_{sample_fps:.2f}fps.pkl"
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            return pickle.load(f)

    loader = VideoLoader()
    if not loader.load_local(str(spec.path)):
        raise RuntimeError(f"영상을 열 수 없습니다: {spec.path}")

    start_frame = int(max(0.0, spec.start_sec) * loader.fps)
    end_sec = spec.end_sec if spec.end_sec is not None else loader.duration
    end_frame = int(max(spec.start_sec, end_sec) * loader.fps)
    if loader.total_frames > 0:
        end_frame = min(end_frame, loader.total_frames)
    sample_step = max(1, int(round(loader.fps / max(sample_fps, 1e-6))))

    stream_id = f"bench_{spec.video_id}_{'filtered' if use_filter else 'raw'}"
    detector.reset_filter(stream_id)
    poses = []
    for offset, frame in enumerate(loader.get_frame_range(start_frame, end_frame)):
        if offset % sample_step != 0:
            continue
        frame_idx = start_frame + offset
        timestamp = frame_idx / max(loader.fps, 1.0)
        pose = detector.detect(
            frame,
            filter_stream=stream_id if use_filter else None,
            timestamp=timestamp,
        )
        if pose is not None:
            poses.append(pose)
    loader.release()

    with cache_path.open("wb") as f:
        pickle.dump(poses, f)
    return poses


def evaluate_pair(
    ref_spec: VideoSpec,
    user_spec: VideoSpec,
    ref_poses: list[dict],
    user_poses: list[dict],
    config: EvalConfig,
) -> dict:
    calc = BenchmarkSimilarity(use_procrustes=config.use_procrustes)
    if not ref_poses or not user_poses:
        return empty_pair_row(ref_spec, user_spec, config)

    if config.use_dtw:
        path, dtw_cost = calc.dtw_match(ref_poses, user_poses)
        match_type = "dtw"
    else:
        path = linear_match(len(ref_poses), len(user_poses))
        dtw_cost = ""
        match_type = "linear"
    if not path:
        return empty_pair_row(ref_spec, user_spec, config)

    similarities = []
    angle_sims = []
    coord_sims = []
    coord_errors = []
    angle_errors = []
    mirror_count = 0
    status_switches = 0
    prev_status = None

    for ref_idx, user_idx in path:
        sim = calc.compute(ref_poses[ref_idx], user_poses[user_idx])
        similarities.append(sim["overall"])
        angle_sims.append(sim["angle_similarity"])
        coord_sims.append(sim["coord_similarity"])
        coord_errors.extend(sim["keypoint_errors"].values())
        angle_errors.extend(sim["angle_errors"].values())
        mirror_count += int(bool(sim.get("mirror_used")))
        status_signature = tuple(sorted(sim["keypoint_status"].items()))
        if prev_status is not None and status_signature != prev_status:
            status_switches += 1
        prev_status = status_signature

    return {
        "config": config.name,
        "ref_video": ref_spec.video_id,
        "user_video": user_spec.video_id,
        "ref_label": ref_spec.label,
        "user_label": user_spec.label,
        "use_filter": int(config.use_filter),
        "use_procrustes": int(config.use_procrustes),
        "use_dtw": int(config.use_dtw),
        "match_type": match_type,
        "ref_pose_count": len(ref_poses),
        "user_pose_count": len(user_poses),
        "matched_frame_count": len(path),
        "avg_overall": mean(similarities),
        "std_overall": stdev(similarities),
        "min_overall": min_or_blank(similarities),
        "avg_angle_similarity": mean(angle_sims),
        "avg_coord_similarity": mean(coord_sims),
        "avg_coord_error": mean(coord_errors),
        "avg_angle_error_deg": mean(angle_errors),
        "dtw_cost": dtw_cost,
        "mirror_ratio": mirror_count / max(len(path), 1),
        "status_switches": status_switches,
        "status_switches_per_frame": status_switches / max(len(path), 1),
    }


def empty_pair_row(ref_spec: VideoSpec, user_spec: VideoSpec, config: EvalConfig) -> dict:
    return {
        "config": config.name,
        "ref_video": ref_spec.video_id,
        "user_video": user_spec.video_id,
        "ref_label": ref_spec.label,
        "user_label": user_spec.label,
        "use_filter": int(config.use_filter),
        "use_procrustes": int(config.use_procrustes),
        "use_dtw": int(config.use_dtw),
        "match_type": "dtw" if config.use_dtw else "linear",
        "ref_pose_count": 0,
        "user_pose_count": 0,
        "matched_frame_count": 0,
        "avg_overall": "",
        "std_overall": "",
        "min_overall": "",
        "avg_angle_similarity": "",
        "avg_coord_similarity": "",
        "avg_coord_error": "",
        "avg_angle_error_deg": "",
        "dtw_cost": "",
        "mirror_ratio": "",
        "status_switches": "",
        "status_switches_per_frame": "",
    }


def video_quality_row(spec: VideoSpec, raw_poses: list[dict], filtered_poses: list[dict]) -> dict:
    raw_kp_jitter = keypoint_jitter(raw_poses)
    filtered_kp_jitter = keypoint_jitter(filtered_poses)
    raw_ang_jitter = angle_jitter(raw_poses)
    filtered_ang_jitter = angle_jitter(filtered_poses)
    return {
        "video": spec.video_id,
        "path": str(spec.path),
        "label": spec.label,
        "raw_pose_count": len(raw_poses),
        "filtered_pose_count": len(filtered_poses),
        "raw_keypoint_jitter": raw_kp_jitter,
        "filtered_keypoint_jitter": filtered_kp_jitter,
        "jitter_delta": subtract_or_blank(filtered_kp_jitter, raw_kp_jitter),
        "jitter_reduction_ratio": reduction_ratio(raw_kp_jitter, filtered_kp_jitter),
        "raw_angle_jitter_deg": raw_ang_jitter,
        "filtered_angle_jitter_deg": filtered_ang_jitter,
        "angle_jitter_delta_deg": subtract_or_blank(filtered_ang_jitter, raw_ang_jitter),
        "angle_jitter_reduction_ratio": reduction_ratio(raw_ang_jitter, filtered_ang_jitter),
    }


def summarize_by_config(rows: list[dict]) -> list[dict]:
    summary = []
    for config in EVAL_CONFIGS:
        selected = [row for row in rows if row["config"] == config.name]
        summary.append(
            {
                "config": config.name,
                "pair_count": len(selected),
                "use_filter": int(config.use_filter),
                "use_procrustes": int(config.use_procrustes),
                "use_dtw": int(config.use_dtw),
                "mean_overall": mean([row["avg_overall"] for row in selected]),
                "mean_coord_similarity": mean([row["avg_coord_similarity"] for row in selected]),
                "mean_angle_similarity": mean([row["avg_angle_similarity"] for row in selected]),
                "mean_coord_error": mean([row["avg_coord_error"] for row in selected]),
                "mean_angle_error_deg": mean([row["avg_angle_error_deg"] for row in selected]),
                "mean_overall_std": mean([row["std_overall"] for row in selected]),
                "mean_status_switches_per_frame": mean([row["status_switches_per_frame"] for row in selected]),
                "mean_dtw_cost": mean([row["dtw_cost"] for row in selected if row["dtw_cost"] != ""]),
            }
        )
    return summary


def summarize_effects(summary_rows: list[dict]) -> list[dict]:
    by_name = {row["config"]: row for row in summary_rows}
    baseline = by_name["baseline"]
    effects = []
    for name in ("procrustes_only", "dtw_only", "filter_only", "all_enabled"):
        row = by_name[name]
        effects.append(
            {
                "comparison": f"{name}_vs_baseline",
                "overall_delta": subtract_or_blank(row["mean_overall"], baseline["mean_overall"]),
                "coord_similarity_delta": subtract_or_blank(
                    row["mean_coord_similarity"], baseline["mean_coord_similarity"]
                ),
                "angle_similarity_delta": subtract_or_blank(
                    row["mean_angle_similarity"], baseline["mean_angle_similarity"]
                ),
                "coord_error_delta": subtract_or_blank(row["mean_coord_error"], baseline["mean_coord_error"]),
                "overall_std_delta": subtract_or_blank(row["mean_overall_std"], baseline["mean_overall_std"]),
                "status_switches_per_frame_delta": (
                    subtract_or_blank(
                        row["mean_status_switches_per_frame"],
                        baseline["mean_status_switches_per_frame"],
                    )
                ),
            }
        )
    return effects


def summarize_procrustes_based_effects(summary_rows: list[dict]) -> list[dict]:
    by_name = {row["config"]: row for row in summary_rows}
    comparisons = [
        (
            "camera_normalization_effect",
            "baseline",
            "procrustes_only",
            "카메라 각도/위치/크기 차이를 Procrustes가 얼마나 보정했는지 확인",
        ),
        (
            "dtw_after_procrustes",
            "procrustes_only",
            "procrustes_dtw",
            "카메라 보정 이후 DTW가 속도 차이를 얼마나 보정했는지 확인",
        ),
        (
            "filter_after_procrustes",
            "procrustes_only",
            "procrustes_filter",
            "카메라 보정 이후 EMA + One Euro Filter가 점수 안정성을 얼마나 개선했는지 확인",
        ),
        (
            "dtw_filter_after_procrustes",
            "procrustes_only",
            "all_enabled",
            "카메라 보정 이후 DTW와 Filter를 함께 적용한 종합 효과 확인",
        ),
        (
            "filter_added_after_procrustes_dtw",
            "procrustes_dtw",
            "all_enabled",
            "Procrustes + DTW 상태에서 Filter 추가 효과 확인",
        ),
        (
            "dtw_added_after_procrustes_filter",
            "procrustes_filter",
            "all_enabled",
            "Procrustes + Filter 상태에서 DTW 추가 효과 확인",
        ),
    ]

    rows = []
    for comparison, base_name, target_name, purpose in comparisons:
        base = by_name[base_name]
        target = by_name[target_name]
        rows.append(
            {
                "comparison": comparison,
                "base_config": base_name,
                "target_config": target_name,
                "purpose": purpose,
                "base_mean_overall": base["mean_overall"],
                "target_mean_overall": target["mean_overall"],
                "overall_delta": subtract_or_blank(target["mean_overall"], base["mean_overall"]),
                "base_coord_similarity": base["mean_coord_similarity"],
                "target_coord_similarity": target["mean_coord_similarity"],
                "coord_similarity_delta": subtract_or_blank(
                    target["mean_coord_similarity"],
                    base["mean_coord_similarity"],
                ),
                "base_coord_error": base["mean_coord_error"],
                "target_coord_error": target["mean_coord_error"],
                "coord_error_delta": subtract_or_blank(target["mean_coord_error"], base["mean_coord_error"]),
                "base_angle_similarity": base["mean_angle_similarity"],
                "target_angle_similarity": target["mean_angle_similarity"],
                "angle_similarity_delta": subtract_or_blank(
                    target["mean_angle_similarity"],
                    base["mean_angle_similarity"],
                ),
                "base_mean_overall_std": base["mean_overall_std"],
                "target_mean_overall_std": target["mean_overall_std"],
                "overall_std_delta": subtract_or_blank(
                    target["mean_overall_std"],
                    base["mean_overall_std"],
                ),
                "base_status_switches_per_frame": base["mean_status_switches_per_frame"],
                "target_status_switches_per_frame": target["mean_status_switches_per_frame"],
                "status_switches_per_frame_delta": subtract_or_blank(
                    target["mean_status_switches_per_frame"],
                    base["mean_status_switches_per_frame"],
                ),
                "target_mean_dtw_cost": target["mean_dtw_cost"],
            }
        )
    return rows


def linear_match(n: int, m: int) -> list[tuple[int, int]]:
    pair_count = min(n, m)
    if pair_count <= 0:
        return []
    return [
        (int(idx * n / pair_count), int(idx * m / pair_count))
        for idx in range(pair_count)
    ]


def keypoint_jitter(poses: list[dict]) -> float:
    values = []
    for prev, cur in zip(poses, poses[1:]):
        valid = prev["valid"] & cur["valid"]
        if np.any(valid):
            distances = np.linalg.norm(cur["keypoints"][valid] - prev["keypoints"][valid], axis=1)
            values.extend(distances.tolist())
    return mean(values)


def angle_jitter(poses: list[dict]) -> float:
    values = []
    for prev, cur in zip(poses, poses[1:]):
        for joint, cur_angle in cur.get("angles", {}).items():
            if joint in prev.get("angles", {}):
                values.append(abs(cur_angle - prev["angles"][joint]))
    return mean(values)


def reduction_ratio(before: float, after: float) -> float:
    if before == "" or before <= 1e-12:
        return ""
    return (before - after) / before


def subtract_or_blank(left, right) -> float | str:
    if left == "" or right == "":
        return ""
    return float(left) - float(right)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value, default: float) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def parse_optional_float(value) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def mean(values) -> float | str:
    clean = [float(v) for v in values if v != ""]
    if not clean:
        return ""
    return float(statistics.fmean(clean))


def stdev(values) -> float | str:
    clean = [float(v) for v in values if v != ""]
    if len(clean) < 2:
        return 0.0 if clean else ""
    return float(statistics.pstdev(clean))


def min_or_blank(values) -> float | str:
    clean = [float(v) for v in values if v != ""]
    return min(clean) if clean else ""


if __name__ == "__main__":
    main()
