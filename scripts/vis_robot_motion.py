from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


NAO_VISUAL_ROOT_BASE_HEIGHT = 0.33
NAO_VISUAL_ROOT_ROTATION = R.from_euler("z", 180, degrees=True) * R.from_euler("x", 90, degrees=True)


def prepare_visual_root_motion(robot_type, root_pos, root_rot):
    """Apply display-only root pose fixes. The loaded motion arrays are left unchanged."""
    if robot_type != "nao":
        return root_pos, root_rot

    visual_root_pos = NAO_VISUAL_ROOT_ROTATION.apply(root_pos)
    target_initial_z = root_pos[0, 2] + NAO_VISUAL_ROOT_BASE_HEIGHT
    visual_root_pos[:, 2] += target_initial_z - visual_root_pos[0, 2]

    visual_root_rot = (
        NAO_VISUAL_ROOT_ROTATION * R.from_quat(root_rot, scalar_first=True)
    ).as_quat(scalar_first=True)

    return visual_root_pos, visual_root_rot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
    parser.add_argument("--loop", action="store_true",
                        help="Loop motion playback indefinitely")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    visual_motion_root_pos, visual_motion_root_rot = prepare_visual_root_motion(
        robot_type,
        motion_root_pos,
        motion_root_rot,
    )
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            camera_follow=False,
                            record_video=args.record_video, video_path=args.video_path)
    
    frame_idx = 0
    if args.loop:
        # Loop motion playback indefinitely
        while True:
            env.step(visual_motion_root_pos[frame_idx],
                     visual_motion_root_rot[frame_idx],
                     motion_dof_pos[frame_idx],
                     rate_limit=True)
            frame_idx += 1
            if frame_idx >= len(motion_root_pos):
                frame_idx = 0
    else:
        # Play motion once (useful for clean video export)
        for frame_idx in range(len(motion_root_pos)):
            env.step(visual_motion_root_pos[frame_idx],
                     visual_motion_root_rot[frame_idx],
                     motion_dof_pos[frame_idx],
                     rate_limit=True)
        env.close()
