import argparse
import concurrent.futures
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.beat2_processing.common import (
    FEATURE_COLUMNS,
    butter_lowpass_filter,
    compute_laban_features,
    load_source_cache,
    make_feature_row,
    read_manifest,
    resolve_repo_path,
    write_csv,
    write_summary,
)

_WORKER_CACHE_ROOT = None
_WORKER_CUTOFF = None
_WORKER_FILTER_ORDER = None
_WORKER_STATIC_PATH_THRESHOLD_M = None


def init_worker(cache_root: str, cutoff: float, filter_order: int, static_path_threshold_m: float) -> None:
    global _WORKER_CACHE_ROOT
    global _WORKER_CUTOFF
    global _WORKER_FILTER_ORDER
    global _WORKER_STATIC_PATH_THRESHOLD_M

    _WORKER_CACHE_ROOT = Path(cache_root)
    _WORKER_CUTOFF = cutoff
    _WORKER_FILTER_ORDER = filter_order
    _WORKER_STATIC_PATH_THRESHOLD_M = static_path_threshold_m


def extract_cache_features(cache: dict, cutoff: float, filter_order: int, static_path_threshold_m: float) -> tuple[dict, list[str], int]:
    filtered = butter_lowpass_filter(
        cache["positions"],
        fps=cache["fps"],
        cutoff=cutoff,
        order=filter_order,
    )
    features, warnings = compute_laban_features(
        filtered,
        dt=1.0 / cache["fps"],
        static_path_threshold_m=static_path_threshold_m,
    )
    return features, warnings, cache["num_frames"]


def run_synthetic_tests() -> None:
    fps = 30.0
    dt = 1.0 / fps
    frames = 90
    static_positions = np.zeros((frames, 6, 3), dtype=np.float64)
    static_features, static_warnings = compute_laban_features(static_positions, dt)
    print("[SYNTHETIC] static:", static_features, static_warnings)

    t = np.arange(frames, dtype=np.float64) * dt
    straight_positions = np.zeros((frames, 6, 3), dtype=np.float64)
    for joint_idx in range(6):
        straight_positions[:, joint_idx, 0] = 0.1 * t + joint_idx * 0.01
    straight_features, _ = compute_laban_features(straight_positions, dt)

    if abs(static_features["W"]) > 1e-10 or abs(static_features["Ti"]) > 1e-10:
        raise AssertionError("Static synthetic test produced non-zero W or Ti")
    if abs(static_features["F"]) > 1e-10:
        raise AssertionError("Static synthetic test produced non-zero F")
    if not math.isnan(static_features["S"]):
        raise AssertionError("Static synthetic test should produce S=NaN")
    if straight_features["S"] < 0.999:
        raise AssertionError("Straight-line synthetic test should produce S close to 1")


def process_manifest_row(row: dict) -> tuple[dict | None, str, dict | None]:
    clip_id = row["clip_id"]
    cache_path = _WORKER_CACHE_ROOT / f"{clip_id}_source_eval.npz"
    try:
        cache = load_source_cache(cache_path)
        features, warnings, num_frames = extract_cache_features(
            cache,
            cutoff=_WORKER_CUTOFF,
            filter_order=_WORKER_FILTER_ORDER,
            static_path_threshold_m=_WORKER_STATIC_PATH_THRESHOLD_M,
        )
        log = {"warnings": warnings} if warnings else None
        return make_feature_row(row, features, num_frames), clip_id, log
    except Exception as exc:
        return None, clip_id, {"error": f"{type(exc).__name__}: {exc}"}


def run_batch(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_repo_path(repo_root, args.manifest).resolve()
    cache_root = resolve_repo_path(repo_root, args.cache_root).resolve()
    output_dir = resolve_repo_path(repo_root, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "beat2_source_features.csv"
    errors_path = output_dir / "beat2_source_feature_errors.json"
    summary_path = output_dir / "beat2_source_feature_summary_by_emotion.csv"

    manifest_rows = read_manifest(manifest_path)
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]
    if args.smoke_clip:
        manifest_rows = [row for row in manifest_rows if row["clip_id"] == args.smoke_clip]
        if not manifest_rows:
            raise ValueError(f"Smoke clip not found in manifest: {args.smoke_clip}")

    if args.synthetic_test:
        run_synthetic_tests()

    feature_rows = []
    logs = {}

    if args.workers == 1:
        for row in tqdm(manifest_rows, desc="Extracting BEAT2 source Laban features"):
            feature_row, clip_id, log = process_manifest_row_with_args(
                row=row,
                cache_root=cache_root,
                cutoff=args.cutoff,
                filter_order=args.filter_order,
                static_path_threshold_m=args.static_path_threshold_m,
            )
            if feature_row is not None:
                feature_rows.append(feature_row)
            if log is not None:
                logs[clip_id] = log
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(str(cache_root), args.cutoff, args.filter_order, args.static_path_threshold_m),
        ) as executor:
            futures = [executor.submit(process_manifest_row, row) for row in manifest_rows]
            with tqdm(
                total=len(manifest_rows),
                desc=f"Extracting BEAT2 source Laban features ({args.workers} workers)",
            ) as progress:
                for future in concurrent.futures.as_completed(futures):
                    feature_row, clip_id, log = future.result()
                    progress.update(1)
                    if feature_row is not None:
                        feature_rows.append(feature_row)
                    if log is not None:
                        logs[clip_id] = log

    manifest_order = {row["clip_id"]: index for index, row in enumerate(manifest_rows)}
    feature_rows.sort(key=lambda row: manifest_order[row["clip_id"]])

    write_csv(features_path, feature_rows, FEATURE_COLUMNS)
    errors_path.write_text(json.dumps(logs, indent=2, sort_keys=True), encoding="utf-8")
    if feature_rows:
        write_summary(features_path, summary_path)

    print(f"[DONE] features: {features_path}")
    print(f"[DONE] logs: {errors_path}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] rows: {len(feature_rows)} / {len(manifest_rows)}")


def process_manifest_row_with_args(
    row: dict,
    cache_root: Path,
    cutoff: float,
    filter_order: int,
    static_path_threshold_m: float,
) -> tuple[dict | None, str, dict | None]:
    clip_id = row["clip_id"]
    cache_path = cache_root / f"{clip_id}_source_eval.npz"
    try:
        cache = load_source_cache(cache_path)
        features, warnings, num_frames = extract_cache_features(
            cache,
            cutoff=cutoff,
            filter_order=filter_order,
            static_path_threshold_m=static_path_threshold_m,
        )
        log = {"warnings": warnings} if warnings else None
        return make_feature_row(row, features, num_frames), clip_id, log
    except Exception as exc:
        return None, clip_id, {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract source-side Laban Effort features from source evaluation caches."
    )
    parser.add_argument(
        "--manifest",
        default="motion_data/BEAT2/manifests/beat2_emotion_manifest.csv",
        help="Section 1 manifest path, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--cache_root",
        default="motion_data/BEAT2/eval_cache/source",
        help="Source evaluation cache root.",
    )
    parser.add_argument("--output_dir", default="motion_data/BEAT2/features/source")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--static_path_threshold_m", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke_clip", default=None)
    parser.add_argument("--synthetic_test", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > 1:
        args.workers = min(args.workers, os.cpu_count() or args.workers)
    run_batch(args)


if __name__ == "__main__":
    main()
