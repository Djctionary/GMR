from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .motion_retarget import GeneralMotionRetargeting
from .utils.smpl import load_smplx_file, get_smplx_data_offline_fast


@dataclass
class RetargetedMotion:
    fps: float
    root_pos: np.ndarray
    root_rot_xyzw: np.ndarray
    root_rot_wxyz: np.ndarray
    dof_pos: np.ndarray


def retarget_smplx_data_to_motion(
    smplx_data,
    body_model,
    smplx_output,
    actual_human_height: float,
    robot: str,
    backend: str = "gmr_baseline",
    quiet: bool = False,
) -> RetargetedMotion:
    supported_backends = {
        "gmr_baseline",
        "gmr_velocity",
        "gmr_velocity_stage3_wrist",
    }
    if backend not in supported_backends:
        raise ValueError(f"Unsupported retarget backend: {backend}")

    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=30,
    )

    retarget = GeneralMotionRetargeting(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=not quiet,
        use_velocity_tracking=backend == "gmr_velocity",
        use_velocity_stage3=backend == "gmr_velocity_stage3_wrist",
    )

    qpos_list = []
    i = -1
    while True:
        i += 1
        if i >= len(smplx_data_frames):
            break
        qpos_list.append(retarget.retarget(smplx_data_frames[i]))

    root_pos = np.asarray([qpos[:3] for qpos in qpos_list], dtype=np.float32)
    root_rot_wxyz = np.asarray([qpos[3:7] for qpos in qpos_list], dtype=np.float32)
    root_rot_xyzw = np.asarray([quat[[1, 2, 3, 0]] for quat in root_rot_wxyz], dtype=np.float32)
    dof_pos = np.asarray([qpos[7:] for qpos in qpos_list], dtype=np.float32)

    return RetargetedMotion(
        fps=float(aligned_fps),
        root_pos=root_pos,
        root_rot_xyzw=root_rot_xyzw,
        root_rot_wxyz=root_rot_wxyz,
        dof_pos=dof_pos,
    )


def retarget_smplx_file_to_motion(
    smplx_file: str | Path,
    robot: str,
    model_root: str | Path,
    backend: str = "gmr_baseline",
    quiet: bool = False,
) -> RetargetedMotion:
    supported_backends = {
        "gmr_baseline",
        "gmr_velocity",
        "gmr_velocity_stage3_wrist",
    }
    if backend not in supported_backends:
        raise ValueError(f"Unsupported retarget backend: {backend}")

    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        str(smplx_file), Path(model_root)
    )
    return retarget_smplx_data_to_motion(
        smplx_data,
        body_model,
        smplx_output,
        actual_human_height,
        robot=robot,
        backend=backend,
        quiet=quiet,
    )


def save_retargeted_motion(path: str | Path, motion: RetargetedMotion) -> None:
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    motion_data = {
        "fps": motion.fps,
        "root_pos": motion.root_pos,
        "root_rot": motion.root_rot_xyzw,
        "dof_pos": motion.dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with path.open("wb") as file:
        pickle.dump(motion_data, file)
