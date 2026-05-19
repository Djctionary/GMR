# BEAT2 Pipeline

本文件记录当前 BEAT2 English Speech -> NAO 的实验管线命令和简要使用说明。算法结构、数据处理逻辑和指标定义见 `BETA2_Experiment_Log.md`。

本文档不记录本地数据路径、实验数值或结果分析。命令中的 raw dataset 路径均使用占位符。

## Scope

- 数据源：BEAT2 English Speech clips
- 目标机器人：NAO
- 当前 retarget backends：
  - `gmr_baseline`：vanilla GMR baseline
  - `gmr_velocity`：在 baseline 第二阶段位置任务之外，对双腕加入 per-frame velocity-derived FrameTask
  - `gmr_velocity_stage3_wrist`：baseline 两阶段 IK 后追加双腕 velocity stage3，stage3 锁定 root pose 并允许 non-root DOF 更新
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

### 0. Run Whole Backend Pipeline

推荐使用完整 backend wrapper。它会按本文档的 Section 1-7 顺序运行：

- 如果 `motion_data/BEAT2/manifests/beat2_emotion_manifest.csv` 不存在，则先运行 `build_emotion_manifest.py`
- retarget/cache precompute 默认重跑并覆盖输出，以确保 retarget 代码改动会反映到结果中
- 然后继续生成 source/robot Laban features、ANOVA、EFPR、bootstrap CI、MPJPE/JJR/SCR

Baseline:

```bash
conda activate gmr
bash scripts/beat2_processing/run_backend_pipeline.sh \
  --workers 16 \
  --backend gmr_baseline \
  --robot nao \
  --source_up_axis y \
  --beat2_root /path/to/BEAT2 \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30
```

Velocity backend:

```bash
conda activate gmr
bash scripts/beat2_processing/run_backend_pipeline.sh \
  --workers 16 \
  --backend gmr_velocity \
  --robot nao \
  --source_up_axis y \
  --beat2_root /path/to/BEAT2 \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30
```

Velocity stage3 wrist backend:

```bash
conda activate gmr
bash scripts/beat2_processing/run_backend_pipeline.sh \
  --workers 16 \
  --backend gmr_velocity_stage3_wrist \
  --robot nao \
  --source_up_axis y \
  --beat2_root /path/to/BEAT2 \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30
```

如果只是从部分已有 pkl/cache 断点续跑，可传入：

```bash
--skip_existing
```

旧入口 `scripts/beat2_processing/run_backend_from_retarget.sh` 仍保留兼容，但会转发到 `run_backend_pipeline.sh`。

### 1. Build Emotion Manifest

```bash
conda activate gmr
python scripts/beat2_processing/build_emotion_manifest.py \
  --beat2_root /path/to/BEAT2 \
  --output_dir motion_data/BEAT2/manifests
```

### 2. Precompute Converted Motion, Caches, and Retargeted PKL

Baseline:

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_baseline \
  --robot nao \
  --source_up_axis y \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30
```

Velocity backend:

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_velocity \
  --robot nao \
  --source_up_axis y \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30
```

Velocity stage3 wrist backend:

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_velocity_stage3_wrist \
  --robot nao \
  --source_up_axis y \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30
```

`batch_retarget_nao.py` 会根据 `--backend` 自动设置默认输出目录：

```text
motion_data/BEAT2/retargeted/<backend>/
motion_data/BEAT2/eval_cache/<backend>/
```

正式运行建议显式传入 raw dataset 路径：

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

其他 backend 使用同一命令，将 `gmr_velocity` 替换为目标 backend，例如 `gmr_velocity_stage3_wrist`。

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

其他 backend 使用同一命令，将 backend-specific 路径替换为目标 backend。

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

其他 backend 使用同一命令，将 backend-specific 路径替换为目标 backend。

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

其他 backend 使用同一命令，将 backend-specific 路径替换为目标 backend。

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

其他 backend 使用同一命令，将 backend-specific 路径替换为目标 backend。

## Optional Visualization

`viser_compare_retarget.py` 只读取 retargeted pkl，不修改任何 pipeline 输出：

```bash
conda activate gmr
python scripts/beat2_processing/viser_compare_retarget.py \
  --robot nao \
  --data_root motion_data/BEAT2/retargeted \
  --backend gmr_baseline \
  --backend gmr_velocity_stage3_wrist_1 \
  --backend gmr_velocity_stage3_wrist_5 \
  --backend gmr_velocity_stage3_wrist_10 \
  --backend gmr_velocity_stage3_wrist_30 \
  --host 0.0.0.0 \
  --port 8080
```

可通过 `--clip <clip_id>` 指定初始 clip。

## Output Layout

默认目录结构：

```text
motion_data/BEAT2/
  manifests/
  converted/
  retargeted/
    gmr_baseline/
    gmr_velocity/
    gmr_velocity_stage3_wrist_1/
    gmr_velocity_stage3_wrist_5/
    gmr_velocity_stage3_wrist_10/
    gmr_velocity_stage3_wrist_30/
  eval_cache/
    source/
    gmr_baseline/
    gmr_velocity/
    gmr_velocity_stage3_wrist_1/
    gmr_velocity_stage3_wrist_5/
    gmr_velocity_stage3_wrist_10/
    gmr_velocity_stage3_wrist_30/
  features/
    source/
    gmr_baseline/
    gmr_velocity/
    gmr_velocity_stage3_wrist_1/
    gmr_velocity_stage3_wrist_5/
    gmr_velocity_stage3_wrist_10/
    gmr_velocity_stage3_wrist_30/
  anova/
    source/
    gmr_baseline/
    gmr_velocity/
    gmr_velocity_stage3_wrist_1/
    gmr_velocity_stage3_wrist_5/
    gmr_velocity_stage3_wrist_10/
    gmr_velocity_stage3_wrist_30/
  efpr/
    gmr_baseline/
    gmr_velocity/
    gmr_velocity_stage3_wrist_1/
    gmr_velocity_stage3_wrist_5/
    gmr_velocity_stage3_wrist_10/
    gmr_velocity_stage3_wrist_30/
  retarget_metrics/
    gmr_baseline/
    gmr_velocity/
    gmr_velocity_stage3_wrist_1/
    gmr_velocity_stage3_wrist_5/
    gmr_velocity_stage3_wrist_10/
    gmr_velocity_stage3_wrist_30/
```

## Final Stage3 Cost Search

最后一组实验固定算法 backend 为 `gmr_velocity_stage3_wrist`，只改变
`velocity_stage3_cost`，并使用 `--output_backend` 将不同参数写入独立结果目录。
旧的无后缀 `gmr_velocity_stage3_wrist` 结果目录已统一重命名为
`gmr_velocity_stage3_wrist_30`。

参数搜索入口：

```bash
conda activate gmr
bash scripts/beat2_processing/run_stage3_cost_search.sh \
  --workers 16 \
  --src_root /path/to/BEAT2/beat_english_v2.0.0/smplxflame_30 \
  --beat2_root /path/to/BEAT2
```

默认搜索参数：

```text
1 5 10
```

本次最终结果文档只纳入已完成并用于论文整理的：

```text
gmr_baseline
gmr_velocity_stage3_wrist_1
gmr_velocity_stage3_wrist_5
gmr_velocity_stage3_wrist_10
gmr_velocity_stage3_wrist_30
```

相关代码改动：

- `general_motion_retargeting/retarget_pipeline.py`：向 `GeneralMotionRetargeting` 透传 `velocity_stage3_cost`。
- `scripts/beat2_processing/batch_retarget_nao.py`：新增 `--velocity_stage3_cost` 与 `--output_backend`。
- `scripts/beat2_processing/run_backend_pipeline.sh`：新增 `--velocity_stage3_cost` 与 `--output_backend`，结果目录按 output backend 写入。
- `scripts/beat2_processing/run_stage3_cost_search.sh`：新增参数搜索 wrapper。
- `scripts/beat2_processing/viser_compare_retarget.py`：默认可视化 backend 更新为 baseline 与 Stage3 cost-search 结果目录。

## Usage Notes

- 本流程只使用 BEAT2 English Speech，不纳入 English Conversation。
- `source cache` 和 `robot cache` 是下游 features、ANOVA、EFPR、metrics 的输入。
- 除 `batch_retarget_nao.py --skip_existing` 外，pipeline 默认覆盖同名输出。
- 新增普通 backend 时，通常只需要替换 `--backend` 和 backend-specific 输入输出目录。
- 参数搜索时，`--backend` 表示算法实现，`--output_backend` 表示结果目录名。
