"""Batch-convert GVHMR hmr4d_results.pt files to retargeted robot motions.

Examples:
    python scripts/gvhmr_batch_to_robot.py \
        --gvhmr_root /path/to/GVHMR/outputs/demo \
        --model_root assets/body_models \
        --robot nao

    python scripts/gvhmr_batch_to_robot.py \
        --gvhmr_file /path/to/Musk/hmr4d_results.pt \
        --gvhmr_file /path/to/Jensen/hmr4d_results.pt \
        --model_root assets/body_models \
        --robot nao \
        --backend gmr_velocity_stage3_wrist
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich import print

from general_motion_retargeting import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting.retarget_pipeline import (
    retarget_gvhmr_file_to_motion,
    save_retargeted_motion,
)


def resolve_path(path: pathlib.Path) -> pathlib.Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def discover_gvhmr_files(root: pathlib.Path | None, explicit_files: list[pathlib.Path]) -> list[pathlib.Path]:
    files = [resolve_path(path) for path in explicit_files]
    if root is not None:
        files.extend(sorted(resolve_path(root).rglob("hmr4d_results.pt")))

    unique_files = []
    seen = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)
    return unique_files


def backend_label(backend: str, velocity_stage3_cost: float, output_backend: str | None) -> str:
    if output_backend:
        return output_backend
    if backend == "gmr_velocity_stage3_wrist" and velocity_stage3_cost != 30.0:
        cost = f"{velocity_stage3_cost:g}".replace(".", "p")
        return f"{backend}_{cost}"
    return backend


def main() -> None:
    smplx_robot_choices = sorted(set(IK_CONFIG_DICT["smplx"]) & set(ROBOT_XML_DICT))

    parser = argparse.ArgumentParser(
        description="Batch retarget GVHMR hmr4d_results.pt files to robot motion pkl files."
    )
    parser.add_argument(
        "--gvhmr_root",
        type=pathlib.Path,
        default=None,
        help="Root directory searched recursively for hmr4d_results.pt files.",
    )
    parser.add_argument(
        "--gvhmr_file",
        type=pathlib.Path,
        action="append",
        default=[],
        help="Specific GVHMR hmr4d_results.pt file. Can be passed multiple times.",
    )
    parser.add_argument("--model_root", type=pathlib.Path, default=REPO_ROOT / "assets" / "body_models")
    parser.add_argument("--robot", default="nao", choices=smplx_robot_choices)
    parser.add_argument(
        "--backend",
        default="gmr_velocity_stage3_wrist",
        choices=["gmr_baseline", "gmr_velocity", "gmr_velocity_stage3_wrist"],
    )
    parser.add_argument("--velocity_stage3_cost", type=float, default=30.0)
    parser.add_argument(
        "--output_backend",
        default=None,
        help="Optional label used in output filenames. Defaults to backend/cost label.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=REPO_ROOT / "outputs" / "gvhmr_demo",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose retargeting diagnostics.")
    args = parser.parse_args()

    gvhmr_files = discover_gvhmr_files(args.gvhmr_root, args.gvhmr_file)
    if not gvhmr_files:
        raise FileNotFoundError("No hmr4d_results.pt files found. Pass --gvhmr_root or --gvhmr_file.")

    model_root = resolve_path(args.model_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = backend_label(args.backend, args.velocity_stage3_cost, args.output_backend)

    for pt_file in gvhmr_files:
        if not pt_file.exists():
            raise FileNotFoundError(pt_file)

        sample_name = pt_file.parent.name
        save_path = output_dir / f"{sample_name}_{args.robot}_{label}.pkl"
        print(f"\n[bold cyan]Processing {sample_name}[/bold cyan] -> {save_path}")
        motion = retarget_gvhmr_file_to_motion(
            gvhmr_pred_file=pt_file,
            robot=args.robot,
            model_root=model_root,
            backend=args.backend,
            velocity_stage3_cost=args.velocity_stage3_cost,
            quiet=args.quiet,
        )
        save_retargeted_motion(save_path, motion)
        print(f"[green]Saved[/green] {len(motion.dof_pos)} frames @ {motion.fps:.1f} fps -> {save_path}")


if __name__ == "__main__":
    main()
