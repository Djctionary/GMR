"""Viser viewer for a single retargeted robot motion pkl.

Usage:
    python scripts/vis_robot_motion_viser.py --robot nao \
        --robot_motion_path motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_30/example_nao.pkl
"""

from __future__ import annotations

import argparse
import time
import threading
from pathlib import Path

import mujoco as mj
import numpy as np
import viser

from general_motion_retargeting import ROBOT_XML_DICT, load_robot_motion

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
    ("LElbow", "LForeArm"),
    ("LForeArm", "l_wrist"),
    ("torso", "RShoulder"),
    ("RShoulder", "RBicep"),
    ("RBicep", "RElbow"),
    ("RElbow", "RForeArm"),
    ("RForeArm", "r_wrist"),
)

POINT_COLOR = (35, 190, 110)
LINE_COLOR  = (35, 190, 110)


def rotation_matrix_xyz_degrees(xyz_degrees: list[float]) -> np.ndarray:
    """Return a world-space XYZ Euler rotation matrix."""
    rx, ry, rz = np.deg2rad(xyz_degrees)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=np.float32,
    )
    rot_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float32,
    )
    rot_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return rot_z @ rot_y @ rot_x


def body_positions_from_pkl(
    model: mj.MjModel,
    body_ids: list[int],
    pkl_path: Path,
    global_rotation_xyz_deg: list[float],
) -> tuple[np.ndarray, float]:
    """Return positions [T, B, 3] and fps from a retargeted motion pkl."""
    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(str(pkl_path))
    data = mj.MjData(model)
    T = root_pos.shape[0]
    positions = np.zeros((T, len(body_ids), 3), dtype=np.float32)
    for t in range(T):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot[t]
        n = min(dof_pos.shape[1], data.qpos.shape[0] - 7)
        data.qpos[7:7 + n] = dof_pos[t, :n]
        mj.mj_forward(model, data)
        positions[t] = data.xpos[body_ids]
    global_rotation = rotation_matrix_xyz_degrees(global_rotation_xyz_deg)
    positions = positions @ global_rotation.T
    return positions, float(fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="nao")
    parser.add_argument("--robot_motion_path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.015)
    parser.add_argument("--line_width", type=float, default=4.0)
    parser.add_argument(
        "--global_rotation_xyz_deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Rotate visualized skeleton points in world XYZ Euler degrees.",
    )
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    global_rotation_xyz_deg = (
        [90.0, 0.0, 0.0]
        if args.global_rotation_xyz_deg is None and args.robot == "nao"
        else (args.global_rotation_xyz_deg or [0.0, 0.0, 0.0])
    )

    # resolve which body names actually exist in this model
    all_edge_bodies: list[str] = []
    for a, b in SKELETON_EDGES:
        all_edge_bodies += [a, b]
    body_names = list(dict.fromkeys(all_edge_bodies))  # deduplicated, ordered

    body_ids: list[int] = []
    valid_body_names: list[str] = []
    for name in body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_ids.append(bid)
            valid_body_names.append(name)
        else:
            print(f"[warn] body '{name}' not found in model, skipping")

    body_index = {name: i for i, name in enumerate(valid_body_names)}
    edge_indices = []
    for a, b in SKELETON_EDGES:
        if a in body_index and b in body_index:
            edge_indices.append((body_index[a], body_index[b]))
    edge_indices_np = np.asarray(edge_indices, dtype=np.int32)

    print(f"Loading {args.robot_motion_path} ...")
    positions, fps = body_positions_from_pkl(
        model, body_ids, Path(args.robot_motion_path), global_rotation_xyz_deg
    )
    T = positions.shape[0]
    print(f"Loaded {T} frames @ {fps:.1f} fps")
    print(f"Global visual rotation xyz_deg={global_rotation_xyz_deg}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid(
        "/ground", width=4.0, height=4.0, plane="xy",
        cell_size=0.2, section_size=1.0,
        cell_color=(90, 90, 90), section_color=(140, 140, 140),
    )

    play_btn   = server.gui.add_button("play / pause")
    frame_sl   = server.gui.add_slider("frame", 0, T - 1, 1, 0)
    speed_sl   = server.gui.add_slider("speed", 0.05, 2.0, 0.05, 1.0)

    state = {"playing": True, "frame": 0}
    lock = threading.RLock()
    point_handle = [None]
    line_handle  = [None]

    def render(frame_idx: int) -> None:
        pts = positions[frame_idx]
        segs = pts[edge_indices_np]
        pt_colors  = np.tile(np.array(POINT_COLOR, dtype=np.uint8), (pts.shape[0], 1))
        seg_colors = np.tile(np.array(LINE_COLOR,  dtype=np.uint8), (segs.shape[0], 2, 1))

        if point_handle[0] is not None:
            point_handle[0].remove()
        if line_handle[0] is not None:
            line_handle[0].remove()

        point_handle[0] = server.scene.add_point_cloud(
            "/robot/points", points=pts, colors=pt_colors,
            point_size=args.point_size, point_shape="circle",
        )
        line_handle[0] = server.scene.add_line_segments(
            "/robot/skeleton", points=segs, colors=seg_colors,
            line_width=args.line_width,
        )

    play_btn.on_click(lambda _: state.update({"playing": not state["playing"]}))
    frame_sl.on_update(lambda _: (state.update({"frame": int(frame_sl.value)}), render(state["frame"])))

    render(0)
    print(f"[viser] Open http://localhost:{args.port}")

    last = time.time()
    while True:
        time.sleep(0.005)
        with lock:
            if not state["playing"]:
                last = time.time()
                continue
            now = time.time()
            interval = 1.0 / max(fps * float(speed_sl.value), 1e-6)
            if now - last < interval:
                continue
            last = now
            state["frame"] = (state["frame"] + 1) % T
            frame_sl.value = state["frame"]
            render(state["frame"])


if __name__ == "__main__":
    main()
