import argparse
import pathlib

from general_motion_retargeting import (
    RobotMotionViewer,
    retarget_smplx_file_to_motion,
    save_retargeted_motion,
)


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", type=str, required=True, help="SMPLX motion file to load.")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "unitree_h1",
            "unitree_h1_2",
            "booster_t1",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "kuavo_s45",
            "hightorque_hi",
            "galaxea_r1pro",
            "berkeley_humanoid_lite",
            "booster_k1",
            "pnd_adam_lite",
            "openloong",
            "tienkung",
            "fourier_gr3",
            "nao",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion.")
    parser.add_argument("--loop", action="store_true", help="Loop the motion.")
    parser.add_argument("--record_video", action="store_true", help="Record the video.")
    parser.add_argument("--rate_limit", action="store_true", help="Rate limit playback.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose retargeting diagnostics.")
    parser.add_argument(
        "--backend",
        default="gmr_baseline",
        help="Retarget backend name. Current implementation supports gmr_baseline.",
    )
    args = parser.parse_args()

    model_root = here / ".." / "assets" / "body_models"
    motion = retarget_smplx_file_to_motion(
        smplx_file=args.smplx_file,
        robot=args.robot,
        model_root=model_root,
        backend=args.backend,
        quiet=args.quiet,
    )

    if args.save_path is not None:
        save_retargeted_motion(args.save_path, motion)

    if not args.headless:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=motion.fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=f"videos/{args.robot}_{pathlib.Path(args.smplx_file).stem}.mp4",
        )
        frame_idx = 0
        while True:
            viewer.step(
                root_pos=motion.root_pos[frame_idx],
                root_rot=motion.root_rot_wxyz[frame_idx],
                dof_pos=motion.dof_pos[frame_idx],
                rate_limit=args.rate_limit,
                follow_camera=False,
            )
            frame_idx += 1
            if frame_idx >= motion.root_pos.shape[0]:
                if args.loop:
                    frame_idx = 0
                else:
                    break
        viewer.close()
