import argparse
import concurrent.futures
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from tqdm import tqdm


UPPER_BODY_JOINT_INDICES = [16, 17, 18, 19, 20, 21]
FEATURE_COLUMNS = ["clip_id", "emotion", "speaker_id", "num_frames", "W", "Ti", "S", "F"]
SPACE_WINDOW_FRAMES = 90
SPACE_STRIDE_FRAMES = 45

_WORKER_BODY_MODEL = None
_WORKER_BEAT2_ENGLISH_ROOT = None
_WORKER_CUTOFF = None
_WORKER_FILTER_ORDER = None
_WORKER_STATIC_PATH_THRESHOLD_M = None


def read_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_smplx_model(model_root: Path):
    import smplx

    return smplx.create(
        model_root,
        "smplx",
        gender="neutral",
        use_pca=False,
        num_betas=16,
    )


def _scalar(value) -> float:
    array = np.asarray(value)
    if array.shape == ():
        return float(array.item())
    return float(array.reshape(-1)[0])


def load_smplx_joints(npz_path: Path, body_model) -> tuple[np.ndarray, np.ndarray, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        if "poses" in data:
            poses = data["poses"].astype(np.float32)
            if poses.ndim != 2 or poses.shape[1] < 66:
                raise ValueError(f"Invalid poses shape {poses.shape}")
            global_orient = poses[:, :3]
            body_pose = poses[:, 3:66]
        elif {"root_orient", "pose_body"}.issubset(data.files):
            global_orient = data["root_orient"].astype(np.float32)
            body_pose = data["pose_body"].astype(np.float32)
        else:
            raise ValueError("Missing SMPL-X pose keys: expected poses or root_orient/pose_body")

        trans = data["trans"].astype(np.float32)
        betas_raw = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
        betas = np.zeros(16, dtype=np.float32)
        betas[: min(16, betas_raw.shape[0])] = betas_raw[:16]
        fps = _scalar(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30.0

    num_frames = body_pose.shape[0]
    if trans.shape != (num_frames, 3):
        raise ValueError(f"Invalid trans shape {trans.shape}, expected ({num_frames}, 3)")

    device = next(body_model.parameters()).device
    with torch.no_grad():
        output = body_model(
            betas=torch.from_numpy(betas).float().view(1, -1).to(device),
            global_orient=torch.from_numpy(global_orient).float().to(device),
            body_pose=torch.from_numpy(body_pose).float().to(device),
            transl=torch.from_numpy(trans).float().to(device),
            left_hand_pose=torch.zeros(num_frames, 45, dtype=torch.float32, device=device),
            right_hand_pose=torch.zeros(num_frames, 45, dtype=torch.float32, device=device),
            jaw_pose=torch.zeros(num_frames, 3, dtype=torch.float32, device=device),
            leye_pose=torch.zeros(num_frames, 3, dtype=torch.float32, device=device),
            reye_pose=torch.zeros(num_frames, 3, dtype=torch.float32, device=device),
            return_verts=False,
        )

    joints = output.joints.detach().cpu().numpy()
    if joints.shape[1] <= max(UPPER_BODY_JOINT_INDICES):
        raise ValueError(f"SMPL-X output has too few joints: {joints.shape}")

    joints_6 = joints[:, UPPER_BODY_JOINT_INDICES, :].astype(np.float64)
    pelvis = joints[:, 0, :].astype(np.float64)
    return joints_6, pelvis, fps


def to_pelvis_relative(joints_6: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
    if joints_6.ndim != 3 or joints_6.shape[1:] != (6, 3):
        raise ValueError(f"Expected joints_6 shape (T, 6, 3), got {joints_6.shape}")
    if pelvis.shape != (joints_6.shape[0], 3):
        raise ValueError(f"Expected pelvis shape ({joints_6.shape[0]}, 3), got {pelvis.shape}")
    return joints_6 - pelvis[:, None, :]


def butter_lowpass_filter(
    positions: np.ndarray, fps: float = 30.0, cutoff: float = 6.0, order: int = 4
) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")
    nyquist = fps / 2.0
    if cutoff >= nyquist:
        raise ValueError(f"Cutoff {cutoff} Hz must be lower than Nyquist {nyquist} Hz")

    b, a = butter(order, cutoff / nyquist, btype="low")
    padlen = 3 * max(len(a), len(b))
    if positions.shape[0] <= padlen:
        raise ValueError(
            f"Clip is too short for filtfilt: frames={positions.shape[0]}, padlen={padlen}"
        )
    return filtfilt(b, a, positions, axis=0)


def compute_windowed_space(
    positions: np.ndarray,
    static_path_threshold_m: float = 0.01,
    window_frames: int = SPACE_WINDOW_FRAMES,
    stride_frames: int = SPACE_STRIDE_FRAMES,
) -> tuple[float, list[str]]:
    """Compute clip-level Space as mean directness over overlapping gesture windows.

    Directness is geometric, so it is defined per keypoint first. Each 3-second
    window produces one S(w) by averaging valid keypoint directness values; the
    clip-level S is the mean of all valid S(w). Long clips are not penalized by a
    single endpoint-to-endpoint denominator.
    """
    num_frames = positions.shape[0]
    if num_frames < 2:
        return math.nan, ["space_nan_too_few_frames"]

    if num_frames < window_frames:
        windows = [(0, num_frames)]
    else:
        windows = [
            (start, start + window_frames)
            for start in range(0, num_frames - window_frames + 1, stride_frames)
        ]

    warnings = []
    window_scores = []
    excluded_keypoints = 0
    for start, end in windows:
        window = positions[start:end]
        endpoint_distance = np.linalg.norm(window[-1] - window[0], axis=1)
        path_length = np.sum(np.linalg.norm(np.diff(window, axis=0), axis=2), axis=0)
        valid_space = path_length >= static_path_threshold_m
        excluded_keypoints += int(np.count_nonzero(~valid_space))
        if not np.any(valid_space):
            continue

        directness = endpoint_distance[valid_space] / path_length[valid_space]
        window_scores.append(float(np.mean(np.clip(directness, 0.0, 1.0))))

    if excluded_keypoints:
        warnings.append(f"static_keypoints_excluded_in_space_windows:{excluded_keypoints}")
    if not window_scores:
        return math.nan, warnings + ["space_nan_all_windows_static"]

    return float(np.mean(window_scores)), warnings


def compute_laban_features(
    positions: np.ndarray, dt: float, static_path_threshold_m: float = 0.01
) -> tuple[dict, list[str]]:
    if positions.ndim != 3 or positions.shape[1:] != (6, 3):
        raise ValueError(f"Expected positions shape (T, 6, 3), got {positions.shape}")
    if positions.shape[0] < 5:
        raise ValueError("At least 5 frames are required for five-point jerk")
    if dt <= 0:
        raise ValueError(f"Invalid dt: {dt}")

    warnings = []
    velocity = (positions[2:] - positions[:-2]) / (2.0 * dt)
    acceleration = (positions[2:] - 2.0 * positions[1:-1] + positions[:-2]) / (dt**2)
    jerk = (
        positions[4:]
        - 2.0 * positions[3:-1]
        + 2.0 * positions[1:-3]
        - positions[:-4]
    ) / (2.0 * dt**3)

    kinetic_energy = 0.5 * np.sum(np.linalg.norm(velocity, axis=2) ** 2, axis=1)
    suddenness = np.sum(np.linalg.norm(acceleration, axis=2), axis=1)
    jerk_energy = np.linalg.norm(jerk, axis=2) ** 2

    space, space_warnings = compute_windowed_space(
        positions,
        static_path_threshold_m=static_path_threshold_m,
    )
    warnings.extend(space_warnings)

    features = {
        "W": float(np.max(kinetic_energy)),
        "Ti": float(np.max(suddenness)),
        "S": space,
        "F": float(np.sqrt(np.mean(jerk_energy))),
    }
    return features, warnings


def extract_clip_features(
    npz_path: Path,
    body_model,
    cutoff: float,
    filter_order: int,
    static_path_threshold_m: float,
) -> tuple[dict, list[str], int]:
    joints_6, pelvis, fps = load_smplx_joints(npz_path, body_model)
    positions = to_pelvis_relative(joints_6, pelvis)
    filtered = butter_lowpass_filter(positions, fps=fps, cutoff=cutoff, order=filter_order)
    features, warnings = compute_laban_features(
        filtered,
        dt=1.0 / fps,
        static_path_threshold_m=static_path_threshold_m,
    )
    return features, warnings, positions.shape[0]


def init_worker(
    model_root: str,
    beat2_english_root: str,
    cutoff: float,
    filter_order: int,
    static_path_threshold_m: float,
) -> None:
    global _WORKER_BODY_MODEL
    global _WORKER_BEAT2_ENGLISH_ROOT
    global _WORKER_CUTOFF
    global _WORKER_FILTER_ORDER
    global _WORKER_STATIC_PATH_THRESHOLD_M

    torch.set_num_threads(1)
    _WORKER_BODY_MODEL = load_smplx_model(Path(model_root))
    _WORKER_BODY_MODEL.eval()
    _WORKER_BEAT2_ENGLISH_ROOT = Path(beat2_english_root)
    _WORKER_CUTOFF = cutoff
    _WORKER_FILTER_ORDER = filter_order
    _WORKER_STATIC_PATH_THRESHOLD_M = static_path_threshold_m


def process_manifest_row(row: dict) -> tuple[dict | None, str, dict | None]:
    clip_id = row["clip_id"]
    npz_path = _WORKER_BEAT2_ENGLISH_ROOT / "smplxflame_30" / row["npz_filename"]
    try:
        features, warnings, num_frames = extract_clip_features(
            npz_path=npz_path,
            body_model=_WORKER_BODY_MODEL,
            cutoff=_WORKER_CUTOFF,
            filter_order=_WORKER_FILTER_ORDER,
            static_path_threshold_m=_WORKER_STATIC_PATH_THRESHOLD_M,
        )
        log = {"warnings": warnings} if warnings else None
        return make_feature_row(row, features, num_frames), clip_id, log
    except Exception as exc:
        return None, clip_id, {"error": f"{type(exc).__name__}: {exc}"}


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
    straight_features, straight_warnings = compute_laban_features(straight_positions, dt)
    print("[SYNTHETIC] straight:", straight_features, straight_warnings)

    if abs(static_features["W"]) > 1e-10 or abs(static_features["Ti"]) > 1e-10:
        raise AssertionError("Static synthetic test produced non-zero W or Ti")
    if abs(static_features["F"]) > 1e-10:
        raise AssertionError("Static synthetic test produced non-zero F")
    if not math.isnan(static_features["S"]):
        raise AssertionError("Static synthetic test should produce S=NaN")
    if straight_features["S"] < 0.999:
        raise AssertionError("Straight-line synthetic test should produce S close to 1")


def make_feature_row(manifest_row: dict, features: dict, num_frames: int) -> dict:
    return {
        "clip_id": manifest_row["clip_id"],
        "emotion": manifest_row["emotion"],
        "speaker_id": manifest_row["speaker_id"],
        "num_frames": str(num_frames),
        "W": f"{features['W']:.10g}",
        "Ti": f"{features['Ti']:.10g}",
        "S": "nan" if math.isnan(features["S"]) else f"{features['S']:.10g}",
        "F": f"{features['F']:.10g}",
    }


def write_summary(features_path: Path, summary_path: Path) -> None:
    groups = defaultdict(lambda: defaultdict(list))
    with features_path.open(newline="") as file:
        for row in csv.DictReader(file):
            for feature_name in ("W", "Ti", "S", "F"):
                value = float(row[feature_name])
                if not math.isnan(value):
                    groups[row["emotion"]][feature_name].append(value)

    rows = []
    for emotion in sorted(groups):
        row = {"emotion": emotion}
        for feature_name in ("W", "Ti", "S", "F"):
            values = np.asarray(groups[emotion][feature_name], dtype=np.float64)
            row[f"{feature_name}_count"] = str(values.size)
            row[f"{feature_name}_mean"] = f"{np.mean(values):.10g}" if values.size else "nan"
            row[f"{feature_name}_median"] = f"{np.median(values):.10g}" if values.size else "nan"
            row[f"{feature_name}_std"] = f"{np.std(values, ddof=1):.10g}" if values.size > 1 else "nan"
        rows.append(row)

    fieldnames = ["emotion"]
    for feature_name in ("W", "Ti", "S", "F"):
        fieldnames.extend(
            [
                f"{feature_name}_count",
                f"{feature_name}_mean",
                f"{feature_name}_median",
                f"{feature_name}_std",
            ]
        )
    write_csv(summary_path, rows, fieldnames)


def run_batch(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (repo_root / args.manifest).resolve()
    beat2_english_root = Path(args.beat2_english_root).expanduser().resolve()
    model_root = (repo_root / args.model_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
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
        body_model = load_smplx_model(model_root)
        body_model.eval()
        iterator = tqdm(manifest_rows, desc="Extracting BEAT2 source Laban features")
        for row in iterator:
            clip_id = row["clip_id"]
            npz_path = beat2_english_root / "smplxflame_30" / row["npz_filename"]
            try:
                features, warnings, num_frames = extract_clip_features(
                    npz_path=npz_path,
                    body_model=body_model,
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
                str(model_root),
                str(beat2_english_root),
                args.cutoff,
                args.filter_order,
                args.static_path_threshold_m,
            ),
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
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(json.dumps(logs, indent=2, sort_keys=True), encoding="utf-8")

    if feature_rows:
        write_summary(features_path, summary_path)

    print(f"[DONE] features: {features_path}")
    print(f"[DONE] logs: {errors_path}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] rows: {len(feature_rows)} / {len(manifest_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract source-side Laban Effort features from BEAT2 SMPL-X clips."
    )
    parser.add_argument(
        "--manifest",
        default="motion_data/BEAT2/manifests/beat2_emotion_manifest.csv",
        help="Section 1 manifest path, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--beat2_english_root",
        default="/home/vergil/dataset/BEAT2/beat_english_v2.0.0",
        help="BEAT2 English root containing smplxflame_30.",
    )
    parser.add_argument(
        "--model_root",
        default="assets/body_models",
        help="SMPL-X model root, relative to the repository root unless absolute.",
    )
    parser.add_argument("--output_dir", default="motion_data/BEAT2/features")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--filter_order", type=int, default=4)
    parser.add_argument("--static_path_threshold_m", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke_clip", default=None)
    parser.add_argument("--synthetic_test", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Use >1 for multi-CPU extraction.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.workers > 1:
        args.workers = min(args.workers, os.cpu_count() or args.workers)
    run_batch(args)


if __name__ == "__main__":
    main()
