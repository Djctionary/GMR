# BEAT2 Pipeline V2

本文件记录当前 BEAT2 English Speech -> NAO 的实验管线定义。本文档只描述当前有效流程，不和旧版流程做对比。

## Scope

- 数据源：BEAT2 English Speech clips
- 目标机器人：NAO
- 当前 retarget backends：
  - `gmr_baseline`：vanilla GMR baseline
  - `gmr_velocity`：在 baseline 位置任务之外，对双肘和双腕加入 per-frame velocity-derived FrameTask
- 输出目标：
  - `retargeted pkl`，供可视化、RL mimic、下游控制复用
  - `source cache` / `robot cache`，供全部评估复用
  - Laban / ANOVA / EFPR
  - MPJPE / JJR / SCR

## Pipeline Flow

```text
BEAT2 English Speech raw npz
-> emotion manifest
-> precompute pipeline
   -> converted AMASS-compatible npz
   -> source evaluation cache
   -> retarget backend
   -> retargeted robot pkl
   -> robot evaluation cache
-> source Laban features
-> robot Laban features
-> source ANOVA
-> robot ANOVA
-> EFPR + bootstrap CI
-> retarget metrics: MPJPE / JJR / SCR
```

更细的执行图如下：

```text
Section 1
BEAT2 raw npz
-> scripts/beat2_processing/build_emotion_manifest.py
-> motion_data/BEAT2/manifests/beat2_emotion_manifest.csv

Section 2
manifest
-> scripts/beat2_processing/batch_retarget_nao.py
   -> motion_data/BEAT2/converted/<clip_id>_amass_compat.npz
   -> motion_data/BEAT2/eval_cache/source/<clip_id>_source_eval.npz
   -> motion_data/BEAT2/retargeted/<backend>/<clip_id>_nao.pkl
   -> motion_data/BEAT2/eval_cache/<backend>/<clip_id>_nao_eval.npz

Section 3
source cache
-> scripts/beat2_processing/extract_source_laban_features.py
-> motion_data/BEAT2/features/source/beat2_source_features.csv

Section 4
robot cache
-> scripts/beat2_processing/extract_robot_laban_features.py
-> motion_data/BEAT2/features/<backend>/beat2_nao_features.csv

Section 5
source features
-> scripts/beat2_processing/run_anova.py
-> motion_data/BEAT2/anova/source/

robot features
-> scripts/beat2_processing/run_anova.py
-> motion_data/BEAT2/anova/<backend>/

Section 6
source anova + robot anova
-> scripts/beat2_processing/compute_efpr.py
-> motion_data/BEAT2/efpr/<backend>/

source features + robot features
-> scripts/beat2_processing/bootstrap_efpr_ci.py
-> motion_data/BEAT2/efpr/<backend>/

Section 7
source cache + robot cache
-> scripts/beat2_processing/evaluate_nao_retargeting_metrics.py
-> motion_data/BEAT2/retarget_metrics/<backend>/
```

## Commands

### 1. Build Emotion Manifest

```bash
conda activate gmr
python scripts/beat2_processing/build_emotion_manifest.py
```

### 2. Precompute Converted Motion, Caches, and Retargeted PKL

Baseline:

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_baseline \
  --robot nao \
  --source_up_axis y
```

Velocity backend:

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_velocity \
  --robot nao \
  --source_up_axis y
```

`batch_retarget_nao.py` 会根据 `--backend` 自动设置默认输出目录：

```text
motion_data/BEAT2/retargeted/<backend>/
motion_data/BEAT2/eval_cache/<backend>/
```

如果当前机器的 BEAT2 raw 数据不在脚本默认路径，需要显式传入：

```bash
--src_root /path/to/beat_english_v2.0.0/smplxflame_30
```

### 3. Extract Source-side Laban Features

```bash
conda activate gmr
python scripts/beat2_processing/extract_source_laban_features.py \
  --workers 16 \
  --cache_root motion_data/BEAT2/eval_cache/source \
  --output_dir motion_data/BEAT2/features/source
```

### 4. Extract Robot-side Laban Features

Baseline:

```bash
conda activate gmr
python scripts/beat2_processing/extract_robot_laban_features.py \
  --workers 16 \
  --robot nao \
  --cache_root motion_data/BEAT2/eval_cache/gmr_baseline \
  --output_dir motion_data/BEAT2/features/gmr_baseline
```

Velocity backend:

```bash
conda activate gmr
python scripts/beat2_processing/extract_robot_laban_features.py \
  --workers 16 \
  --robot nao \
  --cache_root motion_data/BEAT2/eval_cache/gmr_velocity \
  --output_dir motion_data/BEAT2/features/gmr_velocity
```

### 5. Run Source-side ANOVA

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/source/beat2_source_features.csv \
  --output_dir motion_data/BEAT2/anova/source
```

### 6. Run Robot-side ANOVA

Baseline:

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/anova/gmr_baseline
```

Velocity backend:

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/gmr_velocity/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/anova/gmr_velocity
```

### 7. Compute EFPR

Baseline:

```bash
conda activate gmr
python scripts/beat2_processing/compute_efpr.py \
  --human_anova motion_data/BEAT2/anova/source/anova_main_table.csv \
  --robot_anova motion_data/BEAT2/anova/gmr_baseline/anova_main_table.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_baseline
```

Velocity backend:

```bash
conda activate gmr
python scripts/beat2_processing/compute_efpr.py \
  --human_anova motion_data/BEAT2/anova/source/anova_main_table.csv \
  --robot_anova motion_data/BEAT2/anova/gmr_velocity/anova_main_table.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_velocity
```

### 8. Compute Bootstrap EFPR CI

Baseline:

```bash
conda activate gmr
python scripts/beat2_processing/bootstrap_efpr_ci.py \
  --source_features motion_data/BEAT2/features/source/beat2_source_features.csv \
  --robot_features motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_baseline \
  --n_bootstrap 1000
```

Velocity backend:

```bash
conda activate gmr
python scripts/beat2_processing/bootstrap_efpr_ci.py \
  --source_features motion_data/BEAT2/features/source/beat2_source_features.csv \
  --robot_features motion_data/BEAT2/features/gmr_velocity/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_velocity \
  --n_bootstrap 1000
```

### 9. Evaluate MPJPE / JJR / SCR

Baseline:

```bash
conda activate gmr
python scripts/beat2_processing/evaluate_nao_retargeting_metrics.py \
  --workers 8 \
  --robot nao \
  --source_cache_root motion_data/BEAT2/eval_cache/source \
  --robot_cache_root motion_data/BEAT2/eval_cache/gmr_baseline \
  --output_dir motion_data/BEAT2/retarget_metrics/gmr_baseline \
  --scale_sample_limit 0
```

Velocity backend:

```bash
conda activate gmr
python scripts/beat2_processing/evaluate_nao_retargeting_metrics.py \
  --workers 8 \
  --robot nao \
  --source_cache_root motion_data/BEAT2/eval_cache/source \
  --robot_cache_root motion_data/BEAT2/eval_cache/gmr_velocity \
  --output_dir motion_data/BEAT2/retarget_metrics/gmr_velocity \
  --scale_sample_limit 0
```

## Notes

### Dataset Notes

- 本流程只使用 BEAT2 English Speech，不纳入 English Conversation。
- 情感映射规则以 `build_emotion_manifest.py` 为准。
- 当前 `neutral` question id 区间为 `0-64`。

### Cache Notes

- `source cache` 是 source-side evaluation 的唯一输入。
- `robot cache` 是 robot-side evaluation 的唯一输入。
- 评估阶段不再进行任何 SMPL-X FK、MuJoCo FK 或 contact recovery。
- 当前 cache 和所有结果文件默认覆盖生成。

### Retarget Notes

- `gmr_baseline` 是 vanilla GMR baseline。
- `gmr_velocity` 在 baseline 位置任务之外增加 velocity-derived FrameTask：
  - 作用 body：`left_elbow`、`right_elbow`、`left_wrist`、`right_wrist`
  - 默认 weight：`velocity_tracking_cost = 3.0`
  - velocity scale：直接来自 GMR 已缩放、已 offset 的原始 target trajectory，不额外使用 MPJPE morphology scale
  - 当前实现形式：`target_position = current_robot_body_position + (target_source_position[t] - target_source_position[t-1])`
- `retargeted pkl` 保留，作为 motion 主结果，不因评估 cache 的存在而取消。
- `retargeted pkl` 继续服务于：
  - 可视化
  - RL mimic
  - 下游导出 / 控制

### Source Cache Definition

`motion_data/BEAT2/eval_cache/source/<clip_id>_source_eval.npz`

固定包含：

- `clip_id`
- `emotion`
- `speaker_id`
- `fps`
- `num_frames`
- `reference_name = pelvis`
- `joint_names`
- `positions[T,6,3]`

其中 `positions` 已经是：

- source-up-axis 修正后的坐标
- `pelvis-relative`

### Robot Cache Definition

`motion_data/BEAT2/eval_cache/<backend>/<clip_id>_nao_eval.npz`

固定包含：

- `clip_id`
- `emotion`
- `speaker_id`
- `backend`
- `robot`
- `fps`
- `num_frames`
- `reference_name = torso`
- `body_names`
- `positions[T,6,3]`
- `dof_names`
- `dof_pos[T,D]`
- `self_collision_rate`
- `collision_frame_mask[T]`
- `collision_pair_counts_json`

其中：

- `positions` 已经是 `torso-relative`
- `dof_pos` 供 JJR 使用
- `SCR` 相关结果已经在 precompute 阶段写入 cache

### Metrics Notes

- Laban 特征使用：
  - `W`
  - `Ti`
  - `S`
  - `F`
- MPJPE 使用 source cache 与 robot cache 的 6 点轨迹
- JJR 使用 robot cache 的 `dof_pos`
- SCR 使用 robot cache 中已经写好的碰撞统计结果

### Directory Notes

当前默认目录结构为：

```text
motion_data/BEAT2/
  manifests/
  converted/
  retargeted/
    gmr_baseline/
    gmr_velocity/
  eval_cache/
    source/
    gmr_baseline/
    gmr_velocity/
  features/
    source/
    gmr_baseline/
    gmr_velocity/
  anova/
    source/
    gmr_baseline/
    gmr_velocity/
  efpr/
    gmr_baseline/
    gmr_velocity/
  retarget_metrics/
    gmr_baseline/
    gmr_velocity/
```

### Future Backend Extension

后续若新增新 IK 约束版本，例如：

- `gmr_velocity_acc`

则只替换 precompute 阶段的 backend，并把输出目录切换到新的 backend 名称。`batch_retarget_nao.py` 会自动把 retargeted motion 与 robot cache 写到 `<backend>` 目录；Laban / ANOVA / EFPR / MPJPE / JJR / SCR 脚本应保持不变，只更换输入目录。
