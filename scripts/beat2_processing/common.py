import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from scipy.signal import butter, filtfilt

HUMAN_UPPER_BODY_JOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
HUMAN_UPPER_BODY_JOINT_INDICES = [16, 17, 18, 19, 20, 21]

NAO_UPPER_BODY_NAMES = (
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "l_wrist",
    "r_wrist",
)
NAO_REFERENCE_BODY = "torso"
NAO_UPPER_JOINT_NAMES = (
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
NAO_SCR_BODY_NAMES = (
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
NAO_SCR_EXCLUDED_ADJACENT_PAIRS = {
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

FEATURE_COLUMNS = ["clip_id", "emotion", "speaker_id", "num_frames", "W", "Ti", "S", "F"]
SPACE_WINDOW_FRAMES = 90
SPACE_STRIDE_FRAMES = 45


def resolve_repo_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


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
    smplx_data, smplx_output, _, fps = load_smplx_data_and_output(
        npz_path,
        body_model,
        return_full_pose=False,
    )
    del smplx_data
    joints_6, pelvis = upper_joints_from_smplx_output(smplx_output)
    return joints_6, pelvis, fps


def load_smplx_data_and_output(
    npz_path: Path,
    body_model,
    return_full_pose: bool = True,
) -> tuple[dict, object, float, float]:
    with np.load(npz_path, allow_pickle=True) as data:
        if {"root_orient", "pose_body"}.issubset(data.files):
            global_orient = data["root_orient"].astype(np.float32)
            body_pose = data["pose_body"].astype(np.float32)
        elif "poses" in data:
            poses = data["poses"].astype(np.float32)
            if poses.ndim != 2 or poses.shape[1] < 66:
                raise ValueError(f"Invalid poses shape {poses.shape}")
            global_orient = poses[:, :3]
            body_pose = poses[:, 3:66]
        else:
            raise ValueError("Missing SMPL-X pose keys: expected poses or root_orient/pose_body")

        trans = data["trans"].astype(np.float32)
        betas_raw = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
        betas = np.zeros(16, dtype=np.float32)
        betas[: min(16, betas_raw.shape[0])] = betas_raw[:16]
        fps = _scalar(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30.0
        gender = data["gender"] if "gender" in data else np.array("neutral")

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
            return_full_pose=return_full_pose,
        )

    smplx_data = {
        "pose_body": body_pose,
        "root_orient": global_orient,
        "trans": trans,
        "betas": betas,
        "gender": gender,
        "mocap_frame_rate": np.array(fps, dtype=np.float32),
    }
    if betas.ndim == 1:
        actual_human_height = float(1.66 + 0.1 * betas[0])
    else:
        actual_human_height = float(1.66 + 0.1 * betas[0, 0])

    return smplx_data, output, actual_human_height, fps


def upper_joints_from_smplx_output(smplx_output) -> tuple[np.ndarray, np.ndarray]:
    joints = smplx_output.joints.detach().cpu().numpy()
    if joints.shape[1] <= max(HUMAN_UPPER_BODY_JOINT_INDICES):
        raise ValueError(f"SMPL-X output has too few joints: {joints.shape}")

    joints_6 = joints[:, HUMAN_UPPER_BODY_JOINT_INDICES, :].astype(np.float64)
    pelvis = joints[:, 0, :].astype(np.float64)
    return joints_6, pelvis


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


def save_source_cache(cache_path: Path, manifest_row: dict, positions: np.ndarray, fps: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        clip_id=np.array(manifest_row["clip_id"]),
        emotion=np.array(manifest_row["emotion"]),
        speaker_id=np.array(manifest_row["speaker_id"]),
        fps=np.array(float(fps), dtype=np.float32),
        num_frames=np.array(int(positions.shape[0]), dtype=np.int32),
        reference_name=np.array("pelvis"),
        joint_names=np.asarray(HUMAN_UPPER_BODY_JOINT_NAMES),
        positions=positions.astype(np.float32),
    )


def save_robot_cache(
    cache_path: Path,
    manifest_row: dict,
    backend: str,
    positions: np.ndarray,
    fps: float,
    dof_names: list[str],
    dof_pos: np.ndarray,
    self_collision_rate: float,
    collision_frame_mask: np.ndarray,
    collision_pair_counts: dict[str, int],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        clip_id=np.array(manifest_row["clip_id"]),
        emotion=np.array(manifest_row["emotion"]),
        speaker_id=np.array(manifest_row["speaker_id"]),
        backend=np.array(backend),
        robot=np.array("nao"),
        fps=np.array(float(fps), dtype=np.float32),
        num_frames=np.array(int(positions.shape[0]), dtype=np.int32),
        reference_name=np.array(NAO_REFERENCE_BODY),
        body_names=np.asarray(NAO_UPPER_BODY_NAMES),
        positions=positions.astype(np.float32),
        dof_names=np.asarray(dof_names),
        dof_pos=dof_pos.astype(np.float32),
        self_collision_rate=np.array(float(self_collision_rate), dtype=np.float32),
        collision_frame_mask=collision_frame_mask.astype(np.uint8),
        collision_pair_counts_json=np.array(
            json.dumps(collision_pair_counts, sort_keys=True),
            dtype=np.str_,
        ),
    )


def load_source_cache(cache_path: Path) -> dict:
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "clip_id": str(data["clip_id"].item()),
            "emotion": str(data["emotion"].item()),
            "speaker_id": str(data["speaker_id"].item()),
            "fps": float(data["fps"].item()),
            "num_frames": int(data["num_frames"].item()),
            "reference_name": str(data["reference_name"].item()),
            "joint_names": [str(name) for name in data["joint_names"].tolist()],
            "positions": data["positions"].astype(np.float64),
        }


def load_robot_cache(cache_path: Path) -> dict:
    with np.load(cache_path, allow_pickle=False) as data:
        collision_pair_counts_json = str(data["collision_pair_counts_json"].item())
        dof_pos = data["dof_pos"].astype(np.float64)
        dof_names = align_dof_names_to_dof_pos(
            [str(name) for name in data["dof_names"].tolist()],
            dof_pos,
        )
        return {
            "clip_id": str(data["clip_id"].item()),
            "emotion": str(data["emotion"].item()),
            "speaker_id": str(data["speaker_id"].item()),
            "backend": str(data["backend"].item()),
            "robot": str(data["robot"].item()),
            "fps": float(data["fps"].item()),
            "num_frames": int(data["num_frames"].item()),
            "reference_name": str(data["reference_name"].item()),
            "body_names": [str(name) for name in data["body_names"].tolist()],
            "positions": data["positions"].astype(np.float64),
            "dof_names": dof_names,
            "dof_pos": dof_pos,
            "self_collision_rate": float(data["self_collision_rate"].item()),
            "collision_frame_mask": data["collision_frame_mask"].astype(bool),
            "collision_pair_counts": json.loads(collision_pair_counts_json),
        }


def get_body_ids(model: mj.MjModel, body_names: tuple[str, ...]) -> list[int]:
    body_ids = []
    for body_name in body_names:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found in MuJoCo model: {body_name}")
        body_ids.append(body_id)
    return body_ids


def get_dof_names(model: mj.MjModel) -> list[str]:
    names = []
    for i in range(model.nv):
        names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, model.dof_jntid[i]))
    return names


def align_dof_names_to_dof_pos(dof_names: list[str], dof_pos: np.ndarray) -> list[str]:
    if len(dof_names) == dof_pos.shape[1]:
        return dof_names

    leading_count = len(dof_names) - dof_pos.shape[1]
    if leading_count > 0:
        return dof_names[leading_count:]

    raise ValueError(
        f"dof_names shorter than dof_pos columns: names={len(dof_names)} columns={dof_pos.shape[1]}"
    )


def is_valid_scr_contact(model: mj.MjModel, contact, scr_body_ids: set[int]) -> tuple[bool, tuple[str, str] | None]:
    body1 = int(model.geom_bodyid[contact.geom1])
    body2 = int(model.geom_bodyid[contact.geom2])
    if body1 == body2:
        return False, None
    if body1 not in scr_body_ids or body2 not in scr_body_ids:
        return False, None

    name1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body1)
    name2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body2)
    pair = tuple(sorted((name1, name2)))
    if pair in NAO_SCR_EXCLUDED_ADJACENT_PAIRS:
        return False, None
    return True, pair


def build_robot_cache_from_motion(
    model: mj.MjModel,
    motion_data: dict,
) -> tuple[np.ndarray, list[str], np.ndarray, float, np.ndarray, dict[str, int]]:
    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot_wxyz"]
    dof_pos = motion_data["dof_pos"]
    fps = motion_data["fps"]

    body_ids = get_body_ids(model, NAO_UPPER_BODY_NAMES)
    reference_body_id = get_body_ids(model, (NAO_REFERENCE_BODY,))[0]
    scr_body_ids = set(get_body_ids(model, NAO_SCR_BODY_NAMES))
    dof_names = align_dof_names_to_dof_pos(get_dof_names(model), dof_pos)

    data = mj.MjData(model)
    num_frames = root_pos.shape[0]
    positions = np.zeros((num_frames, len(body_ids), 3), dtype=np.float64)
    collision_frame_mask = np.zeros(num_frames, dtype=bool)
    collision_pair_counts = defaultdict(int)

    for frame_idx in range(num_frames):
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot[frame_idx]
        data.qpos[7 : 7 + dof_pos.shape[1]] = dof_pos[frame_idx]
        mj.mj_forward(model, data)

        reference_pos = data.xpos[reference_body_id].copy()
        positions[frame_idx] = data.xpos[body_ids] - reference_pos

        frame_pairs = set()
        for contact_idx in range(data.ncon):
            valid, pair = is_valid_scr_contact(model, data.contact[contact_idx], scr_body_ids)
            if valid and pair is not None:
                collision_frame_mask[frame_idx] = True
                frame_pairs.add(pair)

        for pair in frame_pairs:
            collision_pair_counts["--".join(pair)] += 1

    self_collision_rate = float(np.mean(collision_frame_mask)) if num_frames else math.nan
    return positions, dof_names, dof_pos, self_collision_rate, collision_frame_mask, dict(collision_pair_counts)
