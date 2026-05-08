#!/usr/bin/env python3
"""Viser viewer for comparing BEAT2 retargeting backends.

This script is intentionally read-only: it loads retargeted .pkl files from
motion_data/BEAT2/retargeted/<backend>/ and visualizes FK-derived upper-body
keypoints. It does not modify pkl/cache/result files.
"""

from __future__ import annotations

import argparse
import time
import threading
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
import viser

from general_motion_retargeting import ROBOT_XML_DICT, load_robot_motion


DEFAULT_BACKENDS = (
    "gmr_baseline",
    "gmr_velocity",
    "gmr_velocity_stage3_wrist",
)

BACKEND_COLORS = {
    "gmr_baseline": (70, 130, 255),
    "gmr_velocity": (255, 150, 35),
    "gmr_velocity_stage3_wrist": (35, 190, 110),
}

BODY_NAMES = (
    "torso",
    "Neck",
    "Head",
    "LPelvis",
    "LHip",
    "LThigh",
    "LTibia",
    "LAnklePitch",
    "l_ankle",
    "RPelvis",
    "RHip",
    "RThigh",
    "RTibia",
    "RAnklePitch",
    "r_ankle",
    "LShoulder",
    "LBicep",
    "LElbow",
    "LForeArm",
    "l_wrist",
    "RShoulder",
    "RBicep",
    "RElbow",
    "RForeArm",
    "r_wrist",
)

SKELETON_EDGES = (
    ("torso", "Neck"),
    ("Neck", "Head"),
    ("torso", "LPelvis"),
    ("LPelvis", "LHip"),
    ("LHip", "LThigh"),
    ("LThigh", "LTibia"),
    ("LTibia", "LAnklePitch"),
    ("LAnklePitch", "l_ankle"),
    ("torso", "RPelvis"),
    ("RPelvis", "RHip"),
    ("RHip", "RThigh"),
    ("RThigh", "RTibia"),
    ("RTibia", "RAnklePitch"),
    ("RAnklePitch", "r_ankle"),
    ("torso", "LShoulder"),
    ("LShoulder", "LBicep"),
    ("LBicep", "LElbow"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LForeArm"),
    ("LForeArm", "l_wrist"),
    ("LElbow", "l_wrist"),
    ("torso", "RShoulder"),
    ("RShoulder", "RBicep"),
    ("RBicep", "RElbow"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RForeArm"),
    ("RForeArm", "r_wrist"),
    ("RElbow", "r_wrist"),
)


@dataclass
class MotionPositions:
    fps: float
    positions: np.ndarray  # [T, B, 3]


@dataclass
class BackendState:
    enabled: object
    x_offset: object
    y_offset: object
    point_handle: object | None = None
    line_handle: object | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare BEAT2 retargeting backends in Viser."
    )
    parser.add_argument("--robot", default="nao")
    parser.add_argument(
        "--data_root",
        default="motion_data/BEAT2/retargeted",
        help="Root containing backend subfolders with <clip>_<robot>.pkl files.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        help="Backend to include. Can be repeated. Defaults to baseline/velocity/stage3.",
    )
    parser.add_argument("--clip", default=None, help="Initial clip id, without _<robot>.pkl.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.015)
    parser.add_argument("--line_width", type=float, default=4.0)
    parser.add_argument("--default_spacing", type=float, default=0.0)
    return parser.parse_args()


def clip_id_from_path(path: Path, robot: str) -> str:
    suffix = f"_{robot}.pkl"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected robot motion filename: {path}")
    return name[: -len(suffix)]


def collect_clip_ids(data_root: Path, backends: tuple[str, ...], robot: str) -> list[str]:
    per_backend: list[set[str]] = []
    for backend in backends:
        backend_dir = data_root / backend
        ids = {
            clip_id_from_path(path, robot)
            for path in backend_dir.glob(f"*_{robot}.pkl")
            if path.is_file()
        }
        per_backend.append(ids)
    if not per_backend:
        return []
    common = set.intersection(*per_backend)
    return sorted(common)


def body_ids_for_model(model: mj.MjModel, body_names: tuple[str, ...]) -> list[int]:
    ids = []
    for name in body_names:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Robot model is missing body: {name}")
        ids.append(body_id)
    return ids


def compute_body_positions(
    model: mj.MjModel,
    body_ids: list[int],
    motion_path: Path,
) -> MotionPositions:
    (
        _motion_data,
        motion_fps,
        root_pos,
        root_rot,
        dof_pos,
        _local_body_pos,
        _link_body_list,
    ) = load_robot_motion(str(motion_path))

    data = mj.MjData(model)
    positions = np.zeros((root_pos.shape[0], len(body_ids), 3), dtype=np.float32)
    for frame_idx in range(root_pos.shape[0]):
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot[frame_idx]
        dof_count = min(dof_pos.shape[1], data.qpos.shape[0] - 7)
        data.qpos[7 : 7 + dof_count] = dof_pos[frame_idx, :dof_count]
        mj.mj_forward(model, data)
        positions[frame_idx] = data.xpos[body_ids]
    return MotionPositions(fps=float(motion_fps), positions=positions)


class CompareViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data_root = Path(args.data_root)
        self.backends = tuple(args.backends or DEFAULT_BACKENDS)
        self.robot = args.robot
        self.model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[self.robot]))
        self.body_ids = body_ids_for_model(self.model, BODY_NAMES)
        self.body_index = {name: idx for idx, name in enumerate(BODY_NAMES)}
        self.edge_indices = np.asarray(
            [(self.body_index[a], self.body_index[b]) for a, b in SKELETON_EDGES],
            dtype=np.int32,
        )

        self.clip_ids = collect_clip_ids(self.data_root, self.backends, self.robot)
        if not self.clip_ids:
            raise FileNotFoundError(
                f"No common *_{self.robot}.pkl clips found for backends {self.backends} "
                f"under {self.data_root}"
            )
        if args.clip is not None and args.clip not in self.clip_ids:
            raise ValueError(f"Clip not found in all selected backends: {args.clip}")

        self.lock = threading.RLock()
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.add_grid(
            "/ground",
            width=6.0,
            height=6.0,
            plane="xy",
            cell_size=0.25,
            section_size=1.0,
            cell_color=(90, 90, 90),
            section_color=(140, 140, 140),
        )

        initial_clip = args.clip or self.clip_ids[0]
        self.motions: dict[str, MotionPositions] = {}
        self.current_clip = initial_clip
        self.frame_idx = 0
        self.num_frames = 1
        self.fps = 30.0
        self.playing = True

        self.clip_dropdown = self.server.gui.add_dropdown(
            "clip",
            self.clip_ids,
            initial_value=initial_clip,
        )
        self.play_button = self.server.gui.add_button("play / pause")
        self.frame_slider = self.server.gui.add_slider("frame", 0, 1, 1, 0)
        self.speed_slider = self.server.gui.add_slider("speed", 0.05, 2.0, 0.05, 1.0)
        self.reload_button = self.server.gui.add_button("reload clip")

        self.backend_states: dict[str, BackendState] = {}
        for idx, backend in enumerate(self.backends):
            color = BACKEND_COLORS.get(backend, (220, 220, 220))
            with self.server.gui.add_folder("", expand_by_default=True):
                self.server.gui.add_html(
                    "<div style='font-weight: 700; white-space: nowrap; "
                    "padding-left: 0.25rem; "
                    f"color: rgb({color[0]}, {color[1]}, {color[2]});'>{backend}</div>"
                )
                enabled = self.server.gui.add_checkbox("show", True)
                x_offset = self.server.gui.add_number(
                    "x offset",
                    initial_value=idx * args.default_spacing,
                    step=0.05,
                )
                y_offset = self.server.gui.add_number(
                    "y offset",
                    initial_value=0.0,
                    step=0.05,
                )
            self.backend_states[backend] = BackendState(enabled, x_offset, y_offset)

        self.clip_dropdown.on_update(lambda _event: self.load_clip(str(self.clip_dropdown.value)))
        self.reload_button.on_click(lambda _event: self.load_clip(str(self.clip_dropdown.value)))
        self.play_button.on_click(lambda _event: self.toggle_playback())
        self.frame_slider.on_update(lambda _event: self.set_frame(int(self.frame_slider.value)))
        for state in self.backend_states.values():
            state.enabled.on_update(lambda _event: self.render_frame())
            state.x_offset.on_update(lambda _event: self.render_frame())
            state.y_offset.on_update(lambda _event: self.render_frame())

        self.load_clip(initial_clip)

    def motion_path(self, backend: str, clip_id: str) -> Path:
        return self.data_root / backend / f"{clip_id}_{self.robot}.pkl"

    def clear_scene_handles(self) -> None:
        for state in self.backend_states.values():
            if state.point_handle is not None:
                state.point_handle.remove()
                state.point_handle = None
            if state.line_handle is not None:
                state.line_handle.remove()
                state.line_handle = None

    def load_clip(self, clip_id: str) -> None:
        with self.lock:
            self.current_clip = clip_id
            self.motions = {
                backend: compute_body_positions(
                    self.model,
                    self.body_ids,
                    self.motion_path(backend, clip_id),
                )
                for backend in self.backends
            }
            self.num_frames = min(motion.positions.shape[0] for motion in self.motions.values())
            self.fps = min(motion.fps for motion in self.motions.values())
            self.frame_idx = min(self.frame_idx, self.num_frames - 1)
            self.frame_slider.max = max(0, self.num_frames - 1)
            self.frame_slider.value = self.frame_idx
            self.clear_scene_handles()
            self.render_frame()
            print(f"[loaded] {clip_id}: frames={self.num_frames}, fps={self.fps:.3f}")

    def set_frame(self, frame_idx: int) -> None:
        with self.lock:
            self.frame_idx = max(0, min(frame_idx, self.num_frames - 1))
            self.render_frame()

    def toggle_playback(self) -> None:
        with self.lock:
            self.playing = not self.playing

    def backend_positions(self, backend: str) -> np.ndarray:
        state = self.backend_states[backend]
        offset = np.array([state.x_offset.value, state.y_offset.value, 0.0], dtype=np.float32)
        return self.motions[backend].positions[self.frame_idx] + offset

    def render_frame(self) -> None:
        with self.lock:
            for backend, state in self.backend_states.items():
                if state.point_handle is not None:
                    state.point_handle.remove()
                    state.point_handle = None
                if state.line_handle is not None:
                    state.line_handle.remove()
                    state.line_handle = None

                if not state.enabled.value:
                    continue

                points = self.backend_positions(backend)
                color = BACKEND_COLORS.get(backend, (220, 220, 220))
                colors = np.tile(np.asarray(color, dtype=np.uint8), (points.shape[0], 1))
                segments = points[self.edge_indices]
                segment_colors = np.tile(
                    np.asarray(color, dtype=np.uint8),
                    (segments.shape[0], 2, 1),
                )

                state.point_handle = self.server.scene.add_point_cloud(
                    f"/{backend}/points",
                    points=points,
                    colors=colors,
                    point_size=self.args.point_size,
                    point_shape="circle",
                )
                state.line_handle = self.server.scene.add_line_segments(
                    f"/{backend}/skeleton",
                    points=segments,
                    colors=segment_colors,
                    line_width=self.args.line_width,
                )

    def run(self) -> None:
        print(f"[viser] Open http://localhost:{self.args.port}")
        last_time = time.time()
        while True:
            time.sleep(0.005)
            with self.lock:
                if not self.playing:
                    last_time = time.time()
                    continue
                now = time.time()
                speed = float(self.speed_slider.value)
                if now - last_time < (1.0 / max(self.fps * speed, 1e-6)):
                    continue
                last_time = now
                self.frame_idx = (self.frame_idx + 1) % self.num_frames
                self.frame_slider.value = self.frame_idx
                self.render_frame()


def main() -> None:
    viewer = CompareViewer(parse_args())
    viewer.run()


if __name__ == "__main__":
    main()
