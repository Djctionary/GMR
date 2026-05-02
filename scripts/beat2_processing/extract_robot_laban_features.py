import argparse
import concurrent.futures
import json
import math
import os
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.data_loader import load_robot_motion
from scripts.beat2_processing.extract_source_laban_features import (
    FEATURE_COLUMNS,
    butter_lowpass_filter,
    compute_laban_features,
    make_feature_row,
    read_manifest,
    write_csv,
    write_summary,
)


NAO_UPPER_BODY_NAMES = ("LShoulder", "RShoulder", "LElbow", "RElbow", "l_wrist", "r_wrist")
NAO_REFERENCE_BODY = "torso"

_WORKER_MODEL = None
_WORKER_BODY_IDS = None
_WORKER_REFERENCE_BODY_ID = None
_WORKER_RETARGETED_ROOT = None
_WORKER_CUTOFF = None
_WORKER_FILTER_ORDER = None
_WORKER_STATIC_PATH_THRESHOLD_M = None


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def load_mujoco_model(robot: str) -> mj.MjModel:
    return mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))


def get_body_ids(model: mj.MjModel, body_names: tuple[str, ...]) -> list[int]:
    body_ids = []
    for body_name in body_names:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found in MuJoCo model: {body_name}")
        body_ids.append(body_id)
    return body_ids


def robot_motion_to_relative_positions(
    motion_path: Path,
    model: mj.MjModel,
    body_ids: list[int],
    reference_body_id: int,
) -> tuple[np.ndarray, int, float]:
    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(motion_path)
    if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError(
            f"Inconsistent motion lengths in {motion_path}: "
            f"root_pos={root_pos.shape}, root_rot={root_rot.shape}, dof_pos={dof_pos.shape}"
        )

    data = mj.MjData(model)
    num_frames = root_pos.shape[0]
    positions = np.zeros((num_frames, len(body_ids), 3), dtype=np.float64)

    for frame_idx in range(num_frames):
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot[frame_idx]
        data.qpos[7 : 7 + dof_pos.shape[1]] = dof_pos[frame_idx]
        mj.mj_forward(model, data)
        reference_pos = data.xpos[reference_body_id].copy()
        positions[frame_idx] = data.xpos[body_ids] - reference_pos

    return positions, num_frames, float(fps)


def extract_clip_features(
    motion_path: Path,
    model: mj.MjModel,
    body_ids: list[int],
    reference_body_id: int,
    cutoff: float,
    filter_order: int,
    static_path_threshold_m: float,
) -> tuple[dict, list[str], int]:
    positions, num_frames, fps = robot_motion_to_relative_positions(
        motion_path=motion_path,
        model=model,
        body_ids=body_ids,
        reference_body_id=reference_body_id,
    )
    filtered = butter_lowpass_filter(positions, fps=fps, cutoff=cutoff, order=filter_order)
    features, warnings = compute_laban_features(
        filtered,
        dt=1.0 / fps,
        static_path_threshold_m=static_path_threshold_m,
    )
    return features, warnings, num_frames


def init_worker(
    robot: str,
    retargeted_root: str,
    cutoff: float,
    filter_order: int,
    static_path_threshold_m: float,
) -> None:
    global _WORKER_MODEL
    global _WORKER_BODY_IDS
    global _WORKER_REFERENCE_BODY_ID
    global _WORKER_RETARGETED_ROOT
    global _WORKER_CUTOFF
    global _WORKER_FILTER_ORDER
    global _WORKER_STATIC_PATH_THRESHOLD_M

    _WORKER_MODEL = load_mujoco_model(robot)
    _WORKER_BODY_IDS = get_body_ids(_WORKER_MODEL, NAO_UPPER_BODY_NAMES)
    _WORKER_REFERENCE_BODY_ID = get_body_ids(_WORKER_MODEL, (NAO_REFERENCE_BODY,))[0]
    _WORKER_RETARGETED_ROOT = Path(retargeted_root)
    _WORKER_CUTOFF = cutoff
    _WORKER_FILTER_ORDER = filter_order
    _WORKER_STATIC_PATH_THRESHOLD_M = static_path_threshold_m


def process_manifest_row(row: dict, robot: str) -> tuple[dict | None, str, dict | None]:
    clip_id = row["clip_id"]
    motion_path = _WORKER_RETARGETED_ROOT / f"{clip_id}_{robot}.pkl"
    try:
        features, warnings, num_frames = extract_clip_features(
            motion_path=motion_path,
            model=_WORKER_MODEL,
            body_ids=_WORKER_BODY_IDS,
            reference_body_id=_WORKER_REFERENCE_BODY_ID,
            cutoff=_WORKER_CUTOFF,
            filter_order=_WORKER_FILTER_ORDER,
            static_path_threshold_m=_WORKER_STATIC_PATH_THRESHOLD_M,
        )
        log = {"warnings": warnings} if warnings else None
        return make_feature_row(row, features, num_frames), clip_id, log
    except Exception as exc:
        return None, clip_id, {"error": f"{type(exc).__name__}: {exc}"}


def run_batch(args: argparse.Namespace) -> None:
    manifest_path = resolve_repo_path(args.manifest).resolve()
    retargeted_root = resolve_repo_path(args.retargeted_root).resolve()
    output_dir = resolve_repo_path(args.output_dir).resolve()

    features_path = output_dir / f"beat2_{args.robot}_features.csv"
    errors_path = output_dir / f"beat2_{args.robot}_feature_errors.json"
    summary_path = output_dir / f"beat2_{args.robot}_feature_summary_by_emotion.csv"

    manifest_rows = read_manifest(manifest_path)
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]
    if args.smoke_clip:
        manifest_rows = [row for row in manifest_rows if row["clip_id"] == args.smoke_clip]
        if not manifest_rows:
            raise ValueError(f"Smoke clip not found in manifest: {args.smoke_clip}")

    feature_rows = []
    logs = {}

    if args.workers == 1:
        model = load_mujoco_model(args.robot)
        body_ids = get_body_ids(model, NAO_UPPER_BODY_NAMES)
        reference_body_id = get_body_ids(model, (NAO_REFERENCE_BODY,))[0]
        iterator = tqdm(manifest_rows, desc=f"Extracting BEAT2 {args.robot} Laban features")
        for row in iterator:
            clip_id = row["clip_id"]
            motion_path = retargeted_root / f"{clip_id}_{args.robot}.pkl"
            try:
                features, warnings, num_frames = extract_clip_features(
                    motion_path=motion_path,
                    model=model,
                    body_ids=body_ids,
                    reference_body_id=reference_body_id,
                    cutoff=args.cutoff,
                    filter_order=args.filter_order,
                    static_path_threshold_m=args.static_path_threshold_m,
                )
                feature_rows.append(make_feature_row(row, features, num_frames))
                if warnings:
                    logs[clip_id] = {"warnings": warnings}
            except Exception as exc:
                logs[clip_id] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=(
                args.robot,
                str(retargeted_root),
                args.cutoff,
                args.filter_order,
                args.static_path_threshold_m,
            ),
        ) as executor:
            futures = [
                executor.submit(process_manifest_row, row, args.robot) for row in manifest_rows
            ]
            with tqdm(
                total=len(manifest_rows),
                desc=f"Extracting BEAT2 {args.robot} Laban features ({args.workers} workers)",
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

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(features_path, feature_rows, FEATURE_COLUMNS)
    errors_path.write_text(json.dumps(logs, indent=2, sort_keys=True), encoding="utf-8")

    if feature_rows:
        write_summary(features_path, summary_path)

    print(f"[DONE] features: {features_path}")
    print(f"[DONE] logs: {errors_path}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] rows: {len(feature_rows)} / {len(manifest_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract robot-side Laban Effort features from retargeted BEAT2 motions."
    )
    parser.add_argument(
        "--manifest",
        default="motion_data/BEAT2/manifests/beat2_emotion_manifest.csv",
        help="Section 1 manifest path, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--retargeted_root",
        default="motion_data/BEAT2/retargeted",
        help="Folder containing retargeted robot .pkl files.",
    )
    parser.add_argument("--output_dir", default="motion_data/BEAT2/features")
    parser.add_argument("--robot", default="nao")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--static_path_threshold_m", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke_clip", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use >1 for multi-CPU extraction.",
    )
    args = parser.parse_args()
    if args.robot != "nao":
        raise ValueError("This script currently defines body mappings only for --robot nao")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > 1:
        args.workers = min(args.workers, os.cpu_count() or args.workers)
    run_batch(args)


if __name__ == "__main__":
    main()
