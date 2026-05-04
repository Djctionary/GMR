import argparse
import concurrent.futures
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.beat2_processing.common import (
    NAO_UPPER_JOINT_NAMES,
    load_robot_cache,
    load_source_cache,
    read_manifest,
    resolve_repo_path,
    write_csv,
)

_WORKER_SOURCE_CACHE_ROOT = None
_WORKER_ROBOT_CACHE_ROOT = None
_WORKER_ROBOT = None
_WORKER_SCALE = None
_WORKER_JUMP_THRESHOLD = None
_WORKER_ENABLE_SCR = None


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def arm_chain_lengths(positions: np.ndarray) -> np.ndarray:
    left = np.linalg.norm(positions[:, 0] - positions[:, 2], axis=1) + np.linalg.norm(
        positions[:, 2] - positions[:, 4], axis=1
    )
    right = np.linalg.norm(positions[:, 1] - positions[:, 3], axis=1) + np.linalg.norm(
        positions[:, 3] - positions[:, 5], axis=1
    )
    return np.concatenate([left, right])


def compute_mpjpe(source_positions: np.ndarray, robot_positions: np.ndarray, scale: float) -> float:
    num_frames = min(source_positions.shape[0], robot_positions.shape[0])
    if num_frames <= 0:
        return math.nan
    diff = robot_positions[:num_frames] - scale * source_positions[:num_frames]
    return float(np.mean(np.linalg.norm(diff, axis=2)))


def get_upper_qpos_columns(dof_names: list[str], joint_names: tuple[str, ...]) -> list[int]:
    columns = []
    for joint_name in joint_names:
        try:
            columns.append(dof_names.index(joint_name))
        except ValueError as exc:
            raise ValueError(f"Joint not found in robot cache dof_names: {joint_name}") from exc
    return columns


def compute_joint_jump_rate(dof_pos: np.ndarray, upper_columns: list[int], threshold: float) -> tuple[float, float]:
    if dof_pos.shape[0] < 2:
        return math.nan, math.nan
    upper = dof_pos[:, upper_columns]
    max_jump_per_frame = np.max(np.abs(np.diff(upper, axis=0)), axis=1)
    return float(np.mean(max_jump_per_frame > threshold)), float(np.max(max_jump_per_frame))


def init_worker(source_cache_root: str, robot_cache_root: str, robot: str, scale: float, jump_threshold: float, enable_scr: bool) -> None:
    global _WORKER_SOURCE_CACHE_ROOT
    global _WORKER_ROBOT_CACHE_ROOT
    global _WORKER_ROBOT
    global _WORKER_SCALE
    global _WORKER_JUMP_THRESHOLD
    global _WORKER_ENABLE_SCR

    _WORKER_SOURCE_CACHE_ROOT = Path(source_cache_root)
    _WORKER_ROBOT_CACHE_ROOT = Path(robot_cache_root)
    _WORKER_ROBOT = robot
    _WORKER_SCALE = scale
    _WORKER_JUMP_THRESHOLD = jump_threshold
    _WORKER_ENABLE_SCR = enable_scr


def compute_scale_for_row(row: dict, source_cache_root: Path, robot_cache_root: Path, robot: str) -> tuple[float | None, float | None, str | None]:
    clip_id = row["clip_id"]
    try:
        source_cache = load_source_cache(source_cache_root / f"{clip_id}_source_eval.npz")
        robot_cache = load_robot_cache(robot_cache_root / f"{clip_id}_{robot}_eval.npz")
        return (
            float(np.nanmean(arm_chain_lengths(robot_cache["positions"]))),
            float(np.nanmean(arm_chain_lengths(source_cache["positions"]))),
            None,
        )
    except Exception as exc:
        return None, None, f"{clip_id}: {type(exc).__name__}: {exc}"


def process_manifest_row(row: dict) -> tuple[dict | None, str, dict | None]:
    return process_manifest_row_with_args(
        row=row,
        source_cache_root=_WORKER_SOURCE_CACHE_ROOT,
        robot_cache_root=_WORKER_ROBOT_CACHE_ROOT,
        robot=_WORKER_ROBOT,
        scale=_WORKER_SCALE,
        jump_threshold=_WORKER_JUMP_THRESHOLD,
        enable_scr=_WORKER_ENABLE_SCR,
    )


def process_manifest_row_with_args(
    row: dict,
    source_cache_root: Path,
    robot_cache_root: Path,
    robot: str,
    scale: float,
    jump_threshold: float,
    enable_scr: bool,
) -> tuple[dict | None, str, dict | None]:
    clip_id = row["clip_id"]
    try:
        source_cache = load_source_cache(source_cache_root / f"{clip_id}_source_eval.npz")
        robot_cache = load_robot_cache(robot_cache_root / f"{clip_id}_{robot}_eval.npz")
        upper_columns = get_upper_qpos_columns(robot_cache["dof_names"], NAO_UPPER_JOINT_NAMES)

        mpjpe_m = compute_mpjpe(source_cache["positions"], robot_cache["positions"], scale)
        jjr, max_joint_jump = compute_joint_jump_rate(
            robot_cache["dof_pos"],
            upper_columns,
            jump_threshold,
        )
        scr = robot_cache["self_collision_rate"] if enable_scr else math.nan

        log = {}
        if source_cache["num_frames"] != robot_cache["num_frames"]:
            log["frame_mismatch"] = {
                "source_frames": int(source_cache["num_frames"]),
                "robot_frames": int(robot_cache["num_frames"]),
            }
        if enable_scr and robot_cache["collision_pair_counts"]:
            log["collision_pairs"] = robot_cache["collision_pair_counts"]

        feature_row = {
            "clip_id": clip_id,
            "emotion": row["emotion"],
            "speaker_id": row["speaker_id"],
            "source_frames": str(source_cache["num_frames"]),
            "robot_frames": str(robot_cache["num_frames"]),
            "mpjpe_m": fmt(mpjpe_m),
            "mpjpe_mm": fmt(mpjpe_m * 1000.0),
            "joint_jump_rate": fmt(jjr),
            "max_joint_jump_rad": fmt(max_joint_jump),
            "self_collision_rate": fmt(scr),
        }
        return feature_row, clip_id, (log if log else None)
    except Exception as exc:
        return None, clip_id, {"error": f"{type(exc).__name__}: {exc}"}


def write_summary(rows: list[dict], output_path: Path) -> None:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in ("mpjpe_mm", "joint_jump_rate", "max_joint_jump_rad", "self_collision_rate"):
            value = float(row[metric])
            if math.isfinite(value):
                groups[row["emotion"]][metric].append(value)
                groups["ALL"][metric].append(value)

    summary_rows = []
    for emotion in sorted(groups, key=lambda value: (value != "ALL", value)):
        out = {"emotion": emotion}
        for metric in ("mpjpe_mm", "joint_jump_rate", "max_joint_jump_rad", "self_collision_rate"):
            values = np.asarray(groups[emotion][metric], dtype=np.float64)
            out[f"{metric}_count"] = str(values.size)
            out[f"{metric}_mean"] = fmt(float(np.mean(values))) if values.size else "nan"
            out[f"{metric}_median"] = fmt(float(np.median(values))) if values.size else "nan"
            out[f"{metric}_std"] = fmt(float(np.std(values, ddof=1))) if values.size > 1 else "nan"
        summary_rows.append(out)

    fieldnames = ["emotion"]
    for metric in ("mpjpe_mm", "joint_jump_rate", "max_joint_jump_rad", "self_collision_rate"):
        fieldnames.extend(
            [
                f"{metric}_count",
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_std",
            ]
        )
    write_csv(output_path, summary_rows, fieldnames)


def compute_global_scale(args: argparse.Namespace, rows: list[dict], source_cache_root: Path, robot_cache_root: Path) -> float:
    if args.scale is not None:
        return args.scale

    sample_rows = rows if args.scale_sample_limit == 0 else rows[: args.scale_sample_limit]
    robot_lengths = []
    source_lengths = []
    scale_errors = []
    for row in tqdm(sample_rows, desc="Estimating fixed morphology scale"):
        robot_len, source_len, error = compute_scale_for_row(
            row,
            source_cache_root=source_cache_root,
            robot_cache_root=robot_cache_root,
            robot=args.robot,
        )
        if error is not None:
            scale_errors.append(error)
        elif robot_len is not None and source_len is not None:
            robot_lengths.append(robot_len)
            source_lengths.append(source_len)

    if not robot_lengths:
        raise ValueError(
            "Could not estimate morphology scale from any paired clips. "
            f"First errors: {scale_errors[:5]}"
        )
    if scale_errors:
        print(f"[WARN] Ignored {len(scale_errors)} clips during scale estimation")
    return float(np.mean(robot_lengths) / np.mean(source_lengths))


def run_batch(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_repo_path(repo_root, args.manifest).resolve()
    source_cache_root = resolve_repo_path(repo_root, args.source_cache_root).resolve()
    robot_cache_root = resolve_repo_path(repo_root, args.robot_cache_root).resolve()
    output_dir = resolve_repo_path(repo_root, args.output_dir).resolve()

    rows = read_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    scale = compute_global_scale(args, rows, source_cache_root, robot_cache_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nao_metric_config.json").write_text(
        json.dumps(
            {
                "scale": scale,
                "scale_source": "user_provided" if args.scale is not None else "auto_mean_arm_chain_length",
                "scale_sample_limit": args.scale_sample_limit,
                "jump_threshold_rad": args.jump_threshold,
                "scr_enabled": not args.disable_scr,
                "upper_joints": list(NAO_UPPER_JOINT_NAMES),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    feature_rows = []
    logs = {}
    if args.workers == 1:
        for row in tqdm(rows, desc="Evaluating NAO retargeting metrics"):
            feature_row, clip_id, log = process_manifest_row_with_args(
                row=row,
                source_cache_root=source_cache_root,
                robot_cache_root=robot_cache_root,
                robot=args.robot,
                scale=scale,
                jump_threshold=args.jump_threshold,
                enable_scr=not args.disable_scr,
            )
            if feature_row is not None:
                feature_rows.append(feature_row)
            if log is not None:
                logs[clip_id] = log
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(
                str(source_cache_root),
                str(robot_cache_root),
                args.robot,
                scale,
                args.jump_threshold,
                not args.disable_scr,
            ),
        ) as executor:
            futures = [executor.submit(process_manifest_row, row) for row in rows]
            with tqdm(total=len(rows), desc="Evaluating NAO retargeting metrics") as progress:
                for future in concurrent.futures.as_completed(futures):
                    feature_row, clip_id, log = future.result()
                    progress.update(1)
                    if feature_row is not None:
                        feature_rows.append(feature_row)
                    if log is not None:
                        logs[clip_id] = log

    manifest_order = {row["clip_id"]: index for index, row in enumerate(rows)}
    feature_rows.sort(key=lambda row: manifest_order[row["clip_id"]])

    per_clip_path = output_dir / "nao_retarget_metrics_per_clip.csv"
    summary_path = output_dir / "nao_retarget_metrics_summary_by_emotion.csv"
    logs_path = output_dir / "nao_retarget_metrics_logs.json"

    fieldnames = [
        "clip_id",
        "emotion",
        "speaker_id",
        "source_frames",
        "robot_frames",
        "mpjpe_m",
        "mpjpe_mm",
        "joint_jump_rate",
        "max_joint_jump_rad",
        "self_collision_rate",
    ]
    write_csv(per_clip_path, feature_rows, fieldnames)
    write_summary(feature_rows, summary_path)
    logs_path.write_text(json.dumps(logs, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[DONE] per-clip metrics: {per_clip_path}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] logs: {logs_path}")
    print(f"[DONE] rows: {len(feature_rows)} / {len(rows)}")
    print(f"[DONE] fixed morphology scale: {scale:.10g}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NAO retargeting MPJPE, Joint Jump Rate, and Self-Collision Rate from caches."
    )
    parser.add_argument("--manifest", default="motion_data/BEAT2/manifests/beat2_emotion_manifest.csv")
    parser.add_argument("--source_cache_root", default="motion_data/BEAT2/eval_cache/source")
    parser.add_argument("--robot_cache_root", default="motion_data/BEAT2/eval_cache/gmr_baseline")
    parser.add_argument("--output_dir", default="motion_data/BEAT2/retarget_metrics/gmr_baseline")
    parser.add_argument("--robot", default="nao")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--scale_sample_limit", type=int, default=50)
    parser.add_argument("--jump_threshold", type=float, default=0.5)
    parser.add_argument("--disable_scr", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    args.workers = min(args.workers, os.cpu_count() or args.workers)
    run_batch(args)


if __name__ == "__main__":
    main()
