import argparse
import os
import pathlib
import sys
import time

import numpy as np

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import IK_CONFIG_DICT, ROBOT_XML_DICT, RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print


def multiply_wxyz_quats(left, right):
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.result_type(left, right),
    )


def yaw_wxyz_quat(yaw_degrees):
    half_yaw = np.deg2rad(yaw_degrees) * 0.5
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])


def rotate_human_data_around_z(human_data, yaw_degrees):
    if yaw_degrees == 0:
        return human_data
    yaw = np.deg2rad(yaw_degrees)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot_z = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    yaw_quat = yaw_wxyz_quat(yaw_degrees)
    rotated_human_data = {}
    for body_name, (pos, quat) in human_data.items():
        pos = np.asarray(pos)
        quat = np.asarray(quat)
        rotated_human_data[body_name] = [
            rot_z @ pos,
            multiply_wxyz_quats(yaw_quat, quat),
        ]
    return rotated_human_data


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent
    smplx_robot_choices = sorted(set(IK_CONFIG_DICT["smplx"]) & set(ROBOT_XML_DICT))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gvhmr_pred_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/GVHMR/outputs/demo/tennis/hmr4d_results.pt",
    )
    
    parser.add_argument(
        "--robot",
        choices=smplx_robot_choices,
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    parser.add_argument(
        "--headless",
        default=False,
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--source_yaw",
        type=float,
        default=0.0,
        help="Rotate the source human motion around the z axis before retargeting, e.g. 180.",
    )

    args = parser.parse_args()
    headless = args.headless or (args.record_video and not os.environ.get("DISPLAY"))
    if headless and args.record_video and os.environ.get("MUJOCO_GL") == "egl":
        print("DISPLAY is not set; recording video with MuJoCo EGL offscreen rendering.")


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        args.gvhmr_pred_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.gvhmr_pred_file.split('/')[-1].split('.')[0]}.mp4",
                                            headless=headless,)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]
        smplx_data = rotate_human_data_around_z(smplx_data, args.source_yaw)

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
