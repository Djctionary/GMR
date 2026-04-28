# NAO Retargeting Quickstart

This note records the working commands and setup for retargeting BEAT2 motions to NAO with GMR.

## 1. Environment

Activate the conda environment:

```bash
conda activate gmr
```

Install the project if needed:

```bash
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

## 2. SMPL-X Body Models

The `smplx` Python package in the `gmr` environment is already configured to load `.pkl` body model files.

Place the SMPL-X body model directory somewhere persistent, for example outside the repo, with files such as:

```bash
smplx/
  SMPLX_NEUTRAL.pkl
  SMPLX_FEMALE.pkl
  SMPLX_MALE.pkl
```

Create the repo link:

```bash
mkdir -p assets/body_models
rm -rf assets/body_models/smplx
ln -s ../../../../data/jiacdong/smplx assets/body_models/smplx
```

Verify:

```bash
ls -l assets/body_models
ls -l assets/body_models/smplx
```

## 3. BEAT2 Dataset

Expected dataset layout:

```bash
../datasets/BEAT2/
  beat_english_v2.0.0/
    smplxflame_30/
```

The retarget script accepts either a single `.npz` motion or a whole folder.

## 4. Retarget One Motion to NAO

Use headless mode on servers without a GUI:

```bash
python scripts/beat2_to_robot.py \
  --src ../datasets/BEAT2/beat_english_v2.0.0/smplxflame_30/10_kieks_0_103_103.npz \
  --robot nao \
  --headless
```

Optional:

```bash
--rate_limit
```

Outputs:

```bash
motion_data/BEAT2/converted/
motion_data/BEAT2/retargeted/
```

Example output file:

```bash
motion_data/BEAT2/retargeted/10_kieks_0_103_103_nao.pkl
```

## 5. Retarget a Whole Folder

```bash
python scripts/beat2_to_robot.py \
  --src ../datasets/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --robot nao \
  --headless
```

## 6. Visualize Saved NAO Motion

Replay a saved `.pkl` motion:

```bash
python scripts/vis_robot_motion.py \
  --robot nao \
  --robot_motion_path motion_data/BEAT2/retargeted/10_kieks_0_103_103_nao.pkl
```

Loop playback:

```bash
python scripts/vis_robot_motion.py \
  --robot nao \
  --robot_motion_path motion_data/BEAT2/retargeted/10_kieks_0_103_103_nao.pkl \
  --loop
```

Record video during playback:

```bash
python scripts/vis_robot_motion.py \
  --robot nao \
  --robot_motion_path motion_data/BEAT2/retargeted/10_kieks_0_103_103_nao.pkl \
  --record_video \
  --video_path videos/10_kieks_0_103_103_nao.mp4
```

Note: visualization requires a GUI `DISPLAY`. On a headless server, use `--headless` for retargeting and run visualization only in a graphical session.

## 7. Inspect NAO Retarget Frames

This is for frame debugging, not motion playback:

```bash
python scripts/vis_nao_frames.py
```

## 8. Notes

- `scripts/beat2_to_robot.py` converts BEAT2 motions into the SMPL-X format expected by GMR, then calls `scripts/smplx_to_robot.py`.
- `--headless` was added so retargeting can run on servers without X11.
- `scripts/vis_nao_frames.py` does not replay `.pkl` motion files. Use `scripts/vis_robot_motion.py` for that.
