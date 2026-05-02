import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.data_loader import load_robot_motion
from scripts.beat2_processing.extract_robot_laban_features import (
    NAO_REFERENCE_BODY,
    NAO_UPPER_BODY_NAMES,
    get_body_ids,
    load_mujoco_model,
    robot_motion_to_relative_positions,
)
from scripts.beat2_processing.extract_source_laban_features import (
    load_smplx_joints,
    load_smplx_model,
    read_manifest,
    to_pelvis_relative,
    write_csv,
)


UPPER_JOINT_NAMES = (
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw",
)

SCR_BODY_NAMES = (
    "torso",
    "LShoulder",
    "LBicep",
    "LForeArm",
    "l_wrist",
    "RShoulder",
    "RBicep",
    "RForeArm",
    "r_wrist",
)

SCR_EXCLUDED_ADJACENT_PAIRS = {
    tuple(sorted(pair))
    for pair in (
        ("torso", "LShoulder"),
        ("torso", "LBicep"),
        ("LShoulder", "LBicep"),
        ("LBicep", "LForeArm"),
        ("LForeArm", "l_wrist"),
        ("torso", "RShoulder"),
        ("torso", "RBicep"),
        ("RShoulder", "RBicep"),
        ("RBicep", "RForeArm"),
        ("RForeArm", "r_wrist"),
    )
}

_WORKER_SMPLX_MODEL = None
_WORKER_MJ_MODEL = None
_WORKER_BODY_IDS = None
_WORKER_REFERENCE_BODY_ID = None
_WORKER_UPPER_QPOS_COLUMNS = None
_WORKER_SCR_BODY_IDS = None
_WORKER_CONVERTED_ROOT = None
_WORKER_RETARGETED_ROOT = None
_WORKER_SCALE = None
_WORKER_JUMP_THRESHOLD = None
_WORKER_ENABLE_SCR = None


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def load_source_positions(converted_path: Path, body_model) -> np.ndarray:
    joints_6, pelvis, _ = load_smplx_joints(converted_path, body_model)
    return to_pelvis_relative(joints_6, pelvis)


def arm_chain_lengths(positions: np.ndarray) -> np.ndarray:
    left = np.linalg.norm(positions[:, 0] - positions[:, 2], axis=1) + np.linalg.norm(
        positions[:, 2] - positions[:, 4], axis=1
    )
    right = np.linalg.norm(positions[:, 1] - positions[:, 3], axis=1) + np.linalg.norm(
        positions[:, 3] - positions[:, 5], axis=1
    )
    return np.concatenate([left, right])


def compute_scale_for_row(row: dict) -> tuple[float, float]:
    clip_id = row["clip_id"]
    converted_path = _WORKER_CONVERTED_ROOT / f"{clip_id}_amass_compat.npz"
    motion_path = _WORKER_RETARGETED_ROOT / f"{clip_id}_nao.pkl"
    source_positions = load_source_positions(converted_path, _WORKER_SMPLX_MODEL)
    robot_positions, _, _ = robot_motion_to_relative_positions(
        motion_path=motion_path,
        model=_WORKER_MJ_MODEL,
        body_ids=_WORKER_BODY_IDS,
        reference_body_id=_WORKER_REFERENCE_BODY_ID,
    )
    return float(np.nanmean(arm_chain_lengths(robot_positions))), float(
        np.nanmean(arm_chain_lengths(source_positions))
    )


def process_scale_row(row: dict) -> tuple[float | None, float | None, str | None]:
    try:
        robot_len, source_len = compute_scale_for_row(row)
        if math.isfinite(robot_len) and math.isfinite(source_len) and source_len > 0:
            return robot_len, source_len, None
        return None, None, f"invalid scale lengths for {row['clip_id']}"
    except Exception as exc:
        return None, None, f"{row['clip_id']}: {type(exc).__name__}: {exc}"


def get_upper_qpos_columns(model: mj.MjModel, joint_names: tuple[str, ...]) -> list[int]:
    columns = []
    for joint_name in joint_names:
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {joint_name}")
        qpos_addr = int(model.jnt_qposadr[joint_id])
        column = qpos_addr - 7
        if column < 0:
            raise ValueError(f"Joint is not in saved dof_pos after free joint: {joint_name}")
        columns.append(column)
    return columns


def compute_mpjpe(source_positions: np.ndarray, robot_positions: np.ndarray, scale: float) -> float:
    num_frames = min(source_positions.shape[0], robot_positions.shape[0])
    if num_frames <= 0:
        return math.nan
    diff = robot_positions[:num_frames] - scale * source_positions[:num_frames]
    return float(np.mean(np.linalg.norm(diff, axis=2)))


def compute_joint_jump_rate(dof_pos: np.ndarray, upper_columns: list[int], threshold: float) -> tuple[float, float]:
    if dof_pos.shape[0] < 2:
        return math.nan, math.nan
    max_col = max(upper_columns)
    if max_col >= dof_pos.shape[1]:
        raise ValueError(f"Upper joint column {max_col} exceeds dof_pos width {dof_pos.shape[1]}")
    upper = dof_pos[:, upper_columns]
    max_jump_per_frame = np.max(np.abs(np.diff(upper, axis=0)), axis=1)
    return float(np.mean(max_jump_per_frame > threshold)), float(np.max(max_jump_per_frame))


def is_valid_scr_contact(model: mj.MjModel, contact) -> tuple[bool, tuple[str, str] | None]:
    body1 = int(model.geom_bodyid[contact.geom1])
    body2 = int(model.geom_bodyid[contact.geom2])
    if body1 == body2:
        return False, None
    if body1 not in _WORKER_SCR_BODY_IDS or body2 not in _WORKER_SCR_BODY_IDS:
        return False, None

    name1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body1)
    name2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body2)
    pair = tuple(sorted((name1, name2)))
    if pair in SCR_EXCLUDED_ADJACENT_PAIRS:
        return False, None
    return True, pair


def compute_self_collision_rate(motion_path: Path, model: mj.MjModel) -> tuple[float, dict[str, int]]:
    _, _, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(motion_path)
    data = mj.MjData(model)
    collision_frames = 0
    pair_counts = defaultdict(int)

    for frame_idx in range(root_pos.shape[0]):
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot[frame_idx]
        data.qpos[7 : 7 + dof_pos.shape[1]] = dof_pos[frame_idx]
        mj.mj_forward(model, data)

        frame_has_collision = False
        frame_pairs = set()
        for contact_idx in range(data.ncon):
            valid, pair = is_valid_scr_contact(model, data.contact[contact_idx])
            if valid and pair is not None:
                frame_has_collision = True
                frame_pairs.add(pair)

        if frame_has_collision:
            collision_frames += 1
        for pair in frame_pairs:
            pair_counts["--".join(pair)] += 1

    total_frames = root_pos.shape[0]
    return (float(collision_frames / total_frames) if total_frames else math.nan), dict(pair_counts)


def init_worker(
    model_root: str,
    converted_root: str,
    retargeted_root: str,
    scale: float,
    jump_threshold: float,
    enable_scr: bool,
) -> None:
    global _WORKER_SMPLX_MODEL
    global _WORKER_MJ_MODEL
    global _WORKER_BODY_IDS
    global _WORKER_REFERENCE_BODY_ID
    global _WORKER_UPPER_QPOS_COLUMNS
    global _WORKER_SCR_BODY_IDS
    global _WORKER_CONVERTED_ROOT
    global _WORKER_RETARGETED_ROOT
    global _WORKER_SCALE
    global _WORKER_JUMP_THRESHOLD
    global _WORKER_ENABLE_SCR

    torch.set_num_threads(1)
    _WORKER_SMPLX_MODEL = load_smplx_model(Path(model_root))
    _WORKER_SMPLX_MODEL.eval()
    _WORKER_MJ_MODEL = load_mujoco_model("nao")
    _WORKER_BODY_IDS = get_body_ids(_WORKER_MJ_MODEL, NAO_UPPER_BODY_NAMES)
    _WORKER_REFERENCE_BODY_ID = get_body_ids(_WORKER_MJ_MODEL, (NAO_REFERENCE_BODY,))[0]
    _WORKER_UPPER_QPOS_COLUMNS = get_upper_qpos_columns(_WORKER_MJ_MODEL, UPPER_JOINT_NAMES)
    _WORKER_SCR_BODY_IDS = set(get_body_ids(_WORKER_MJ_MODEL, SCR_BODY_NAMES))
    _WORKER_CONVERTED_ROOT = Path(converted_root)
    _WORKER_RETARGETED_ROOT = Path(retargeted_root)
    _WORKER_SCALE = scale
    _WORKER_JUMP_THRESHOLD = jump_threshold
    _WORKER_ENABLE_SCR = enable_scr


def process_manifest_row(row: dict) -> tuple[dict | None, str, dict | None]:
    clip_id = row["clip_id"]
    converted_path = _WORKER_CONVERTED_ROOT / f"{clip_id}_amass_compat.npz"
    motion_path = _WORKER_RETARGETED_ROOT / f"{clip_id}_nao.pkl"
    try:
        source_positions = load_source_positions(converted_path, _WORKER_SMPLX_MODEL)
        robot_positions, robot_frames, _ = robot_motion_to_relative_positions(
            motion_path=motion_path,
            model=_WORKER_MJ_MODEL,
            body_ids=_WORKER_BODY_IDS,
            reference_body_id=_WORKER_REFERENCE_BODY_ID,
        )
        _, _, _, _, dof_pos, _, _ = load_robot_motion(motion_path)

        mpjpe_m = compute_mpjpe(source_positions, robot_positions, _WORKER_SCALE)
        jjr, max_joint_jump = compute_joint_jump_rate(
            dof_pos, _WORKER_UPPER_QPOS_COLUMNS, _WORKER_JUMP_THRESHOLD
        )
        if _WORKER_ENABLE_SCR:
            scr, collision_pairs = compute_self_collision_rate(motion_path, _WORKER_MJ_MODEL)
        else:
            scr, collision_pairs = math.nan, {}

        log = {}
        if source_positions.shape[0] != robot_frames:
            log["frame_mismatch"] = {
                "source_frames": int(source_positions.shape[0]),
                "robot_frames": int(robot_frames),
            }
        if collision_pairs:
            log["collision_pairs"] = collision_pairs

        feature_row = {
            "clip_id": clip_id,
            "emotion": row["emotion"],
            "speaker_id": row["speaker_id"],
            "source_frames": str(source_positions.shape[0]),
            "robot_frames": str(robot_frames),
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


def compute_global_scale(args: argparse.Namespace, rows: list[dict]) -> float:
    if args.scale is not None:
        return args.scale

    sample_rows = rows if args.scale_sample_limit == 0 else rows[: args.scale_sample_limit]
    robot_lengths = []
    source_lengths = []
    scale_errors = []

    worker_initargs = (
        str(resolve_repo_path(args.model_root)),
        str(resolve_repo_path(args.converted_root)),
        str(resolve_repo_path(args.retargeted_root)),
        1.0,
        args.jump_threshold,
        False,
    )

    if args.workers == 1:
        init_worker(*worker_initargs)
        iterator = tqdm(sample_rows, desc="Estimating fixed morphology scale")
        for row in iterator:
            robot_len, source_len, error = process_scale_row(row)
            if error is not None:
                scale_errors.append(error)
            elif robot_len is not None and source_len is not None:
                robot_lengths.append(robot_len)
                source_lengths.append(source_len)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=worker_initargs,
        ) as executor:
            futures = [executor.submit(process_scale_row, row) for row in sample_rows]
            with tqdm(
                total=len(sample_rows),
                desc=f"Estimating fixed morphology scale ({args.workers} workers)",
            ) as progress:
                for future in concurrent.futures.as_completed(futures):
                    robot_len, source_len, error = future.result()
                    progress.update(1)
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
    manifest_path = resolve_repo_path(args.manifest).resolve()
    converted_root = resolve_repo_path(args.converted_root).resolve()
    retargeted_root = resolve_repo_path(args.retargeted_root).resolve()
    model_root = resolve_repo_path(args.model_root).resolve()
    output_dir = resolve_repo_path(args.output_dir).resolve()

    rows = read_manifest(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    scale = compute_global_scale(args, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nao_metric_config.json").write_text(
        json.dumps(
            {
                "scale": scale,
                "scale_source": "user_provided" if args.scale is not None else "auto_mean_arm_chain_length",
                "scale_sample_limit": args.scale_sample_limit,
                "jump_threshold_rad": args.jump_threshold,
                "scr_enabled": not args.disable_scr,
                "upper_joints": UPPER_JOINT_NAMES,
                "scr_bodies": SCR_BODY_NAMES,
                "scr_excluded_adjacent_pairs": sorted("--".join(pair) for pair in SCR_EXCLUDED_ADJACENT_PAIRS),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    feature_rows = []
    logs = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(
            str(model_root),
            str(converted_root),
            str(retargeted_root),
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
        description="Evaluate NAO retargeting MPJPE, Joint Jump Rate, and Self-Collision Rate."
    )
    parser.add_argument("--manifest", default="motion_data/BEAT2/manifests/beat2_emotion_manifest.csv")
    parser.add_argument("--converted_root", default="motion_data/BEAT2/converted")
    parser.add_argument("--retargeted_root", default="motion_data/BEAT2/retargeted")
    parser.add_argument("--model_root", default="assets/body_models")
    parser.add_argument("--output_dir", default="motion_data/BEAT2/retarget_metrics")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument(
        "--scale_sample_limit",
        type=int,
        default=50,
        help="Number of clips used to estimate fixed global scale. Use 0 for all selected clips.",
    )
    parser.add_argument("--jump_threshold", type=float, default=0.5)
    parser.add_argument("--disable_scr", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    args.workers = min(args.workers, os.cpu_count() or args.workers)
    run_batch(args)


if __name__ == "__main__":
    main()
