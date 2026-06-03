# GMR-BEAT2-NAO

This repository is a research fork of [GMR: General Motion Retargeting](https://github.com/YanjieZe/GMR). It keeps the original GMR inverse-kinematics retargeting interface, then adds a BEAT2 English Speech -> NAO workflow, NAO-specific assets/configuration, wrist-velocity retargeting variants, and an emotion-preservation evaluation pipeline.

The intended use is:

```text
BEAT2 SMPL-X/FLAME motion
-> AMASS-compatible SMPL-X conversion
-> GMR/NAO retargeting
-> retargeted robot motion pkl
-> Laban features, ANOVA, EFPR, MPJPE/JJR/SCR
-> optional CSV export for whole-body tracking pipelines
```

## What Changed From Original GMR

- Added NAO support through `assets/nao/nao_scene.xml`, NAO meshes/textures, and `general_motion_retargeting/ik_configs/smplx_to_nao.json`.
- Added BEAT2 conversion scripts so BEAT2 English Speech `.npz` clips can be converted into the SMPL-X format expected by GMR.
- Added batch NAO retargeting and evaluation caches for reproducible backend comparisons.
- Added three retarget backends:
  - `gmr_baseline`: vanilla two-stage GMR IK.
  - `gmr_velocity`: adds per-frame wrist velocity-derived `FrameTask`s during retargeting.
  - `gmr_velocity_stage3_wrist`: appends a third IK stage for left/right wrist velocity tracking while keeping the root pose fixed.
- Added Stage3 wrist cost search with separate result names such as `gmr_velocity_stage3_wrist_1`, `gmr_velocity_stage3_wrist_5`, `gmr_velocity_stage3_wrist_10`, and `gmr_velocity_stage3_wrist_30`.
- Added EFPR, bootstrap confidence intervals, and geometric-quality metrics for measuring how much emotion-related motion structure survives retargeting.
- Added CSV export for downstream tracking/control experiments, including projects such as [whole_body_tracking](https://github.com/Djctionary/whole_body_tracking).

## Repository Map

```text
general_motion_retargeting/
  motion_retarget.py              # GMR IK core plus velocity/stage3 variants
  retarget_pipeline.py            # SMPL-X/GVHMR -> RetargetedMotion API
  ik_configs/smplx_to_nao.json    # NAO retarget map

scripts/
  beat2_to_robot.py               # single/folder BEAT2 -> robot entry
  smplx_to_robot.py               # original-style SMPL-X -> robot entry
  batch_gmr_pkl_to_csv.py         # pkl -> CSV export
  beat2_processing/
    run_backend_pipeline.sh       # full BEAT2 -> NAO evaluation pipeline
    run_stage3_cost_search.sh     # Stage3 wrist cost sweep
    batch_retarget_nao.py         # batch retarget + cache generation
    compute_efpr.py               # EFPR from source/robot ANOVA tables
    bootstrap_efpr_ci.py          # paired stratified bootstrap EFPR CI

assets/nao/                       # NAO MuJoCo/URDF/USD assets
BEAT2_PIPELINE.md                 # detailed pipeline command reference
README_NAO.md                     # NAO quickstart
```

## Setup

Tested environment follows the upstream GMR setup.

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

Place SMPL-X body models under:

```text
assets/body_models/smplx/
  SMPLX_NEUTRAL.pkl
  SMPLX_FEMALE.pkl
  SMPLX_MALE.pkl
```

BEAT2 data is expected to contain:

```text
/path/to/BEAT2/
  beat_english_v2.0.0/
    smplxflame_30/
      *.npz
```

## Quick NAO Retargeting

Retarget one BEAT2 clip:

```bash
conda activate gmr
python scripts/beat2_to_robot.py \
  --src /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30/10_kieks_0_103_103.npz \
  --robot nao \
  --headless
```

Retarget a folder:

```bash
python scripts/beat2_to_robot.py \
  --src /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --robot nao \
  --headless
```

Outputs are written under `motion_data/BEAT2/converted/` and `motion_data/BEAT2/retargeted/`.

Visualize a saved NAO motion:

```bash
python scripts/vis_robot_motion.py \
  --robot nao \
  --robot_motion_path motion_data/BEAT2/retargeted/10_kieks_0_103_103_nao.pkl \
  --loop
```

Visualization requires a GUI display. Retargeting can run headlessly.

## Full BEAT2 Evaluation Pipeline

Use the wrapper for complete backend runs:

```bash
conda activate gmr
bash scripts/beat2_processing/run_backend_pipeline.sh \
  --workers 16 \
  --backend gmr_velocity_stage3_wrist \
  --output_backend gmr_velocity_stage3_wrist_30 \
  --velocity_stage3_cost 30 \
  --robot nao \
  --source_up_axis y \
  --beat2_root /path/to/BEAT2 \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30
```

The pipeline runs:

```text
manifest
-> converted motions
-> source/robot evaluation caches
-> retargeted NAO pkl files
-> source/robot Laban features
-> source/robot ANOVA
-> EFPR and bootstrap CI
-> MPJPE, JJR, SCR
```

`--backend` selects the algorithm implementation. `--output_backend` selects the result directory name, which is useful for parameter sweeps.

## Stage3 Wrist Cost Search

Run the configured Stage3 cost sweep:

```bash
bash scripts/beat2_processing/run_stage3_cost_search.sh \
  --workers 16 \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --beat2_root /path/to/BEAT2
```

The final comparison set uses:

```text
gmr_baseline
gmr_velocity_stage3_wrist_1
gmr_velocity_stage3_wrist_5
gmr_velocity_stage3_wrist_10
gmr_velocity_stage3_wrist_30
```

## EFPR

EFPR means Emotion Feature Preservation Rate. It compares emotion-related separability before and after retargeting:

```text
EFPR(feature, effect) = robot_effect_size(feature) / source_effect_size(feature)
```

The current implementation uses Laban features `W`, `Ti`, and `F`, computes ANOVA effect sizes `eta_squared` and `omega_squared`, and reports:

- per-dimension EFPR in `efpr_dimension_table.csv`
- aggregate EFPR as the geometric mean across dimensions in `efpr_summary.json`
- paired stratified bootstrap 95% CI in `efpr_bootstrap_ci.csv`

This makes the evaluation different from pure geometry metrics: MPJPE/JJR/SCR measure motion quality and feasibility, while EFPR measures whether the retargeted NAO motion preserves emotion-discriminative structure from the BEAT2 human motion.

## Output Layout

```text
motion_data/BEAT2/
  manifests/
  converted/
  retargeted/<backend>/
  eval_cache/source/
  eval_cache/<backend>/
  features/source/
  features/<backend>/
  anova/source/
  anova/<backend>/
  efpr/<backend>/
  retarget_metrics/<backend>/
```

Retargeted `.pkl` files contain:

```text
fps
root_pos
root_rot
dof_pos
```

## Downstream Whole-Body Tracking

This repo does not train the final tracking policy. It produces robot reference motions that can be exported and consumed by downstream control/tracking projects.

For CSV export:

```bash
python scripts/batch_gmr_pkl_to_csv.py \
  --folder motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_30
```

Use the generated CSV motions as references for whole-body tracking workflows, including the referenced [whole_body_tracking](https://github.com/Djctionary/whole_body_tracking) project.

## Detailed Docs

- `README_NAO.md`: concise NAO setup, single-clip retargeting, and visualization.
- `BEAT2_PIPELINE.md`: complete BEAT2 backend pipeline commands and output layout.
- `BETA2_Experiment_Log.md`: experiment notes and metric definitions.
- `BEAT2_Final_Stage3_Cost_Search_Results.md`: final Stage3 comparison report.

## License

This fork follows the upstream GMR license. See `LICENSE`.
