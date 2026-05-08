# BEAT2 实验实现说明

本文档只记录当前代码中可确认的 BEAT2 English Speech -> NAO 实现流程、脚本入口、默认参数、产物路径和指标定义。

本文档不包含实验数值、统计解释、运行记录或由运行输出触发的方法判断。需要查看实际产物时，以 `motion_data/BEAT2` 下最新生成的 CSV/JSON 文件为准。

## Scope

- 数据源：BEAT2 English Speech clips。
- 排除范围：English Conversation clips。
- 目标机器人：`nao`。
- 当前 retarget backend：`gmr_baseline`、`gmr_velocity`。
- 主要代码目录：`scripts/beat2_processing/`。
- 共用实现：`scripts/beat2_processing/common.py`。

整体流程：

```text
BEAT2 raw npz
-> emotion manifest
-> precompute pipeline
   -> AMASS-compatible converted npz
   -> source evaluation cache
   -> retarget backend
   -> retargeted robot pkl
   -> robot evaluation cache
-> source / robot Laban features
-> source / robot ANOVA
-> EFPR + bootstrap CI
-> MPJPE / JJR / SCR retargeting metrics
```

与 `BEAT2_PIPELINE.md` 的结构对应关系：

```text
Experiment Log Section 1 -> Pipeline Section 1: emotion manifest
Experiment Log Section 2 -> Pipeline Section 2: precompute pipeline
Experiment Log Section 3 -> Pipeline Sections 3-4: source / robot Laban features
Experiment Log Section 4 -> Pipeline Section 5: source / robot ANOVA
Experiment Log Section 5 -> Pipeline Section 6: EFPR + bootstrap CI
Experiment Log Section 6 -> Pipeline Section 7: MPJPE / JJR / SCR
Experiment Log Section 7 -> Pipeline Commands: minimal execution order
```

## 1. Emotion Manifest

脚本：`scripts/beat2_processing/build_emotion_manifest.py`

默认输入：

```text
--beat2_root /home/vergil/dataset/BEAT2
```

脚本会读取：

```text
<beat2_root>/beat_english_v2.0.0/smplxflame_30
<beat2_root>/beat_english_v2.0.0/wave16k
```

只保留文件名中 recording type 为 `0` 的 English Speech clip。文件名解析规则由 `parse_clip_id()` 定义，例如：

```text
10_kieks_0_103_103.npz
speaker_id = 10
speaker_name = kieks
recording_type = 0
start_id = 103
end_id = 103
```

情感映射由 `EMOTION_RANGES` 定义：

```text
0-64    neutral
65-72   happiness
73-80   anger
81-86   sadness
87-94   contempt
95-102  surprise
103-110 fear
111-118 disgust
```

Manifest 构建会检查：

- 必需字段：`poses`、`trans`、`mocap_frame_rate`
- `poses` / `trans` shape
- `poses` 和 `trans` 帧数一致性
- finite values
- translation drift，默认阈值 `--max_trans_drift_m 5.0`
- 最小帧数，默认 `--min_frames 150`
- 对应音频是否存在

运行命令：

```bash
conda activate gmr
python scripts/beat2_processing/build_emotion_manifest.py
```

默认输出目录：

```text
motion_data/BEAT2/manifests
```

输出文件：

```text
beat2_emotion_manifest.csv
beat2_emotion_group_stats.csv
beat2_emotion_speaker_distribution.csv
beat2_emotion_spot_check_samples.csv
beat2_emotion_problematic_clips.json
```

`beat2_emotion_manifest.csv` 字段：

```text
clip_id,speaker_id,speaker_name,emotion,num_frames,duration_sec,
npz_filename,audio_filename,has_audio,audio_duration_sec,trans_drift_m
```

## 2. Precompute And Retarget

脚本：`scripts/beat2_processing/batch_retarget_nao.py`

该脚本按 manifest 逐 clip 完成四类产物：

```text
motion_data/BEAT2/converted/<clip_id>_amass_compat.npz
motion_data/BEAT2/eval_cache/source/<clip_id>_source_eval.npz
motion_data/BEAT2/retargeted/<backend>/<clip_id>_nao.pkl
motion_data/BEAT2/eval_cache/<backend>/<clip_id>_nao_eval.npz
```

默认参数：

```text
--manifest motion_data/BEAT2/manifests/beat2_emotion_manifest.csv
--src_root /home/vergil/dataset/BEAT2/beat_english_v2.0.0/smplxflame_30
--converted_root motion_data/BEAT2/converted
--source_cache_root motion_data/BEAT2/eval_cache/source
--robot nao
--backend gmr_baseline
--source_up_axis y
--workers 1
```

如果 `--retargeted_root` 和 `--robot_cache_root` 保持默认，脚本会根据 `--backend` 自动写入：

```text
motion_data/BEAT2/retargeted/<backend>
motion_data/BEAT2/eval_cache/<backend>
```

默认行为会覆盖同名 converted npz、source cache、retargeted pkl 和 robot cache。只有显式传入 `--skip_existing` 时，`batch_retarget_nao.py` 才会跳过同时已有 retargeted pkl 与 robot cache 的 clip。

Baseline：

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_baseline \
  --robot nao \
  --source_up_axis y
```

Velocity backend：

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_velocity \
  --robot nao \
  --source_up_axis y
```

### AMASS-compatible Conversion

转换函数：`scripts/beat2_to_robot.py::build_amass_compatible_file`

BEAT2-like 输入使用：

```text
pose_body = poses[:, 3:66]
root_orient = poses[:, :3]
trans = trans
betas = betas[:16]，不足 16 维时补零
gender = gender，缺失时使用 neutral
mocap_frame_rate = mocap_frame_rate，缺失时使用 30
```

当 `--source_up_axis y` 时，`convert_up_axis_to_z_up()` 会将 Y-up 坐标转换为管线使用的 Z-up 坐标。

### Source Cache Definition

`source cache` 由 `common.py::save_source_cache()` 写入：

```text
motion_data/BEAT2/eval_cache/source/<clip_id>_source_eval.npz
```

固定字段：

```text
clip_id
emotion
speaker_id
fps
num_frames
reference_name = pelvis
joint_names
positions[T,6,3]
```

`positions` 是 source-up-axis 修正后的 SMPL-X 上肢 6 点，并已转为 pelvis-relative。

Source cache 的 SMPL-X FK 使用 `common.py::load_smplx_model()` 创建的 neutral SMPL-X model：

```text
gender = neutral
use_pca = False
num_betas = 16
```

SMPL-X 上肢点：

```text
left_shoulder, right_shoulder, left_elbow,
right_elbow, left_wrist, right_wrist
```

对应 joint indices：

```text
16, 17, 18, 19, 20, 21
```

### Retargeted PKL

Retarget 入口：`general_motion_retargeting/retarget_pipeline.py::retarget_smplx_file_to_motion`

支持 backend：

```text
gmr_baseline
gmr_velocity
```

`retarget_smplx_file_to_motion()` 会调用：

```text
load_smplx_file(...)
get_smplx_data_offline_fast(..., tgt_fps=30)
GeneralMotionRetargeting(...)
```

当前 retarget loop 从 frame `0` 开始处理全部 `smplx_data_frames`。

保存函数：`general_motion_retargeting/retarget_pipeline.py::save_retargeted_motion`

PKL 字段：

```text
fps
root_pos
root_rot
dof_pos
local_body_pos
link_body_list
```

其中 `root_rot` 保存为 xyzw 顺序。

### Velocity Backend

实现位置：`general_motion_retargeting/motion_retarget.py::GeneralMotionRetargeting`

`gmr_velocity` 在 `retarget_smplx_file_to_motion()` 中通过以下条件启用：

```text
use_velocity_tracking = backend == "gmr_velocity"
```

默认 velocity tracking bodies：

```text
left_elbow
right_elbow
left_wrist
right_wrist
```

默认 `velocity_tracking_cost = 3.0`。

实现方式：在第二阶段 IK task 中为匹配 body 额外加入 position-only `mink.FrameTask`。每帧目标位置为当前 robot body 世界位置加上相邻源侧目标位置差分：

```text
target_position =
  current_robot_body_position
  + human_data2[body_name][0]
  - previous_velocity_human_data2[body_name][0]
```

首帧没有 previous source target 时，velocity task 目标保持当前 robot body position。

该差分来自 `GeneralMotionRetargeting.update_targets()` 中已缩放并应用 offset / ground 处理后的 `human_data`，不使用 metrics 脚本中的 MPJPE morphology scale。

### Robot Cache Definition

Robot cache 由 `common.py::build_robot_cache_from_motion()` 和 `save_robot_cache()` 写入：

```text
motion_data/BEAT2/eval_cache/<backend>/<clip_id>_nao_eval.npz
```

固定字段：

```text
clip_id
emotion
speaker_id
backend
robot = nao
fps
num_frames
reference_name = torso
body_names
positions[T,6,3]
dof_names
dof_pos[T,D]
self_collision_rate
collision_frame_mask[T]
collision_pair_counts_json
```

NAO 上肢 body：

```text
LShoulder, RShoulder, LElbow, RElbow, l_wrist, r_wrist
```

`positions` 通过 MuJoCo FK 得到，并转为 torso-relative。

Precompute 阶段构建 robot cache 时不从 pkl 反读 `root_rot`，而是直接使用内存中的 `RetargetedMotion.root_rot_wxyz` 写入 MuJoCo `qpos[3:7]`。

Self-collision cache 在同一 MuJoCo pass 中计算，统计范围和过滤规则见本文档第 6 节。

## 3. Laban Feature Extraction

对应 `BEAT2_PIPELINE.md` 的 Section 3 和 Section 4。本节合并说明 source-side 与 robot-side 的共用实现。

源侧脚本：`scripts/beat2_processing/extract_source_laban_features.py`

Robot 侧脚本：`scripts/beat2_processing/extract_robot_laban_features.py`

两侧均从 evaluation cache 读取 `positions[T,6,3]`，并共用 `common.py` 中的以下函数：

```text
butter_lowpass_filter
compute_laban_features
compute_windowed_space
make_feature_row
write_summary
```

默认滤波参数：

```text
--cutoff 6.0
--filter_order 4
```

滤波实现：

```text
scipy.signal.butter
scipy.signal.filtfilt
```

如果 clip 帧数不满足 `filtfilt` padlen 要求，脚本会为该 clip 记录 error。

特征字段：

```text
clip_id,emotion,speaker_id,num_frames,W,Ti,S,F
```

运行源侧：

```bash
conda activate gmr
python scripts/beat2_processing/extract_source_laban_features.py \
  --workers 16 \
  --cache_root motion_data/BEAT2/eval_cache/source \
  --output_dir motion_data/BEAT2/features/source
```

运行 baseline robot 侧：

```bash
conda activate gmr
python scripts/beat2_processing/extract_robot_laban_features.py \
  --workers 16 \
  --robot nao \
  --cache_root motion_data/BEAT2/eval_cache/gmr_baseline \
  --output_dir motion_data/BEAT2/features/gmr_baseline
```

运行 velocity robot 侧：

```bash
conda activate gmr
python scripts/beat2_processing/extract_robot_laban_features.py \
  --workers 16 \
  --robot nao \
  --cache_root motion_data/BEAT2/eval_cache/gmr_velocity \
  --output_dir motion_data/BEAT2/features/gmr_velocity
```

默认输出：

```text
motion_data/BEAT2/features/source/beat2_source_features.csv
motion_data/BEAT2/features/source/beat2_source_feature_summary_by_emotion.csv
motion_data/BEAT2/features/source/beat2_source_feature_errors.json

motion_data/BEAT2/features/<backend>/beat2_nao_features.csv
motion_data/BEAT2/features/<backend>/beat2_nao_feature_summary_by_emotion.csv
motion_data/BEAT2/features/<backend>/beat2_nao_feature_errors.json
```

### Feature Definitions

`compute_laban_features()` 要求输入至少 5 帧。

中心差分：

```text
v(t) = [p(t+1) - p(t-1)] / (2 dt)
a(t) = [p(t+1) - 2p(t) + p(t-1)] / dt^2
jerk(t) = [p(t+2) - 2p(t+1) + 2p(t-1) - p(t-2)] / (2 dt^3)
```

Weight：

```text
W = max_t sum_j 0.5 * ||v_j(t)||^2
```

Time：

```text
Ti = max_t sum_j ||a_j(t)||
```

Flow：

```text
F = sqrt(mean_{t,j} ||jerk_j(t)||^2)
```

Space 使用滑窗 directness：

```text
S_j(w) = ||p_j(t_end) - p_j(t_start)|| /
         sum_t ||p_j(t+1) - p_j(t)||
S(w) = mean_j S_j(w)
S = mean_w S(w)
```

Space 默认参数在 `common.py` 中定义：

```text
SPACE_WINDOW_FRAMES = 90
SPACE_STRIDE_FRAMES = 45
--static_path_threshold_m 0.01
```

短于 90 帧的 clip 使用整段作为一个窗口。长度不短于 90 帧时，窗口起点为 `range(0, num_frames - 90 + 1, 45)`；末尾不足一个完整窗口的 remainder 不单独追加。窗口内 path length 低于阈值的 keypoint 会被排除；directness 会被裁剪到 `[0, 1]`；如果所有窗口都无有效 keypoint，`S` 写为 `nan` 并记录 warning。

## 4. ANOVA

对应 `BEAT2_PIPELINE.md` 的 Section 5。

脚本：`scripts/beat2_processing/run_anova.py`

输入要求：

```text
clip_id
emotion
W
Ti
S
F
```

每个 feature 分别计算：

```text
one-way ANOVA: scipy.stats.f_oneway
Welch ANOVA: pingouin.welch_anova
Kruskal-Wallis: scipy.stats.kruskal
Levene: scipy.stats.levene
Shapiro-Wilk by group: scipy.stats.shapiro
Tukey HSD: statsmodels.stats.multicomp.pairwise_tukeyhsd
eta_squared / omega_squared: statsmodels OLS + typ=2 ANOVA table
```

源侧：

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/source/beat2_source_features.csv \
  --output_dir motion_data/BEAT2/anova/source
```

Baseline robot 侧：

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/anova/gmr_baseline
```

Velocity robot 侧：

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/gmr_velocity/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/anova/gmr_velocity
```

输出文件：

```text
anova_main_table.csv
anova_shapiro_by_group.csv
anova_tukey_hsd.csv
anova_diagnostics.json
```

`anova_main_table.csv` 字段：

```text
feature
F_oneway
p_oneway
F_welch
p_welch
H_kruskal
p_kruskal
levene_p
eta_squared
omega_squared
n_significant_pairs_tukey
```

## 5. EFPR

对应 `BEAT2_PIPELINE.md` 的 Section 6。

脚本：

```text
scripts/beat2_processing/compute_efpr.py
scripts/beat2_processing/bootstrap_efpr_ci.py
```

默认 EFPR dimensions：

```text
W Ti F
```

可通过 `--dimensions` 修改。

Dimension-wise EFPR：

```text
EFPR_d = effect_size_robot,d / effect_size_human,d
```

支持 effect size：

```text
eta_squared
omega_squared
```

Aggregate EFPR 使用几何平均：

```text
EFPR = geometric_mean(EFPR_W, EFPR_Ti, EFPR_F)
```

`compute_efpr.py` 中，如果 human effect size `<= 0`，对应维度 EFPR 写为 `nan` 且不进入 aggregate；如果进入 aggregate 的 EFPR 中存在负值，几何平均会报错，存在 `0` 时 aggregate 为 `0`。`bootstrap_efpr_ci.py` 中 aggregate 只在所有维度 EFPR 都 finite 且 `> 0` 时返回数值，否则返回 `nan`。

Baseline：

```bash
conda activate gmr
python scripts/beat2_processing/compute_efpr.py \
  --human_anova motion_data/BEAT2/anova/source/anova_main_table.csv \
  --robot_anova motion_data/BEAT2/anova/gmr_baseline/anova_main_table.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_baseline
```

Velocity：

```bash
conda activate gmr
python scripts/beat2_processing/compute_efpr.py \
  --human_anova motion_data/BEAT2/anova/source/anova_main_table.csv \
  --robot_anova motion_data/BEAT2/anova/gmr_velocity/anova_main_table.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_velocity
```

输出文件：

```text
efpr_dimension_table.csv
efpr_summary.json
```

### Bootstrap EFPR CI

Bootstrap 脚本从 per-clip feature table 读取 source/robot paired clips，以 `clip_id` 对齐，并检查两侧 `emotion` 一致。

采样方法：

```text
paired stratified bootstrap by emotion
percentile 95% CI
```

默认参数：

```text
--n_bootstrap 1000
--seed 20260502
--dimensions W Ti F
```

Baseline：

```bash
conda activate gmr
python scripts/beat2_processing/bootstrap_efpr_ci.py \
  --source_features motion_data/BEAT2/features/source/beat2_source_features.csv \
  --robot_features motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_baseline \
  --n_bootstrap 1000
```

Velocity：

```bash
conda activate gmr
python scripts/beat2_processing/bootstrap_efpr_ci.py \
  --source_features motion_data/BEAT2/features/source/beat2_source_features.csv \
  --robot_features motion_data/BEAT2/features/gmr_velocity/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_velocity \
  --n_bootstrap 1000
```

输出文件：

```text
efpr_bootstrap_ci.csv
efpr_bootstrap_samples.csv
efpr_bootstrap_summary.json
```

## 6. Retargeting Metrics

对应 `BEAT2_PIPELINE.md` 的 Section 7。

脚本：`scripts/beat2_processing/evaluate_nao_retargeting_metrics.py`

该脚本只读取 source cache 和 robot cache，不重新执行 SMPL-X FK、retargeting 或 MuJoCo cache recovery。

默认参数：

```text
--manifest motion_data/BEAT2/manifests/beat2_emotion_manifest.csv
--source_cache_root motion_data/BEAT2/eval_cache/source
--robot_cache_root motion_data/BEAT2/eval_cache/gmr_baseline
--output_dir motion_data/BEAT2/retarget_metrics/gmr_baseline
--robot nao
--workers 4
--scale_sample_limit 50
--jump_threshold 0.5
```

Baseline：

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

Velocity：

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

输出文件：

```text
nao_metric_config.json
nao_retarget_metrics_per_clip.csv
nao_retarget_metrics_summary_by_emotion.csv
nao_retarget_metrics_logs.json
```

### MPJPE

MPJPE 使用 source cache 与 robot cache 中相同顺序的 6 个上肢点。

帧数对齐：

```text
num_frames = min(source_frames, robot_frames)
```

Morphology scale 默认自动估计：

```text
scale = mean(robot shoulder-elbow-wrist chain length) /
        mean(source shoulder-elbow-wrist chain length)
```

可通过 `--scale` 直接指定固定 scale。`--scale_sample_limit 0` 表示用全部 manifest rows 估计 scale。

MPJPE：

```text
MPJPE = mean_t mean_j ||p_robot_j(t) - scale * p_source_j(t)||
```

脚本同时输出 meter 和 millimeter：

```text
mpjpe_m
mpjpe_mm
```

### Joint Jump Rate

JJR 使用 NAO 上肢 10 个关节：

```text
LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw,
RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw
```

列索引通过 robot cache 中的 `dof_names` 查找，不硬编码 index。

默认阈值：

```text
--jump_threshold 0.5
```

公式：

```text
JJR = mean_t [max_j |q_j(t+1) - q_j(t)| > threshold]
```

同时输出：

```text
max_joint_jump_rad
```

### Self-Collision Rate

SCR 直接使用 robot cache 中的：

```text
self_collision_rate
collision_frame_mask
collision_pair_counts_json
```

如需禁用 SCR 输出，可传入：

```bash
--disable_scr
```

SCR body set：

```text
torso,
LShoulder, LBicep, LForeArm, l_wrist,
RShoulder, RBicep, RForeArm, r_wrist
```

排除的结构性相邻 body pairs：

```text
torso-LShoulder
torso-LBicep
LShoulder-LBicep
LBicep-LForeArm
LForeArm-l_wrist
torso-RShoulder
torso-RBicep
RShoulder-RBicep
RBicep-RForeArm
RForeArm-r_wrist
```

公式：

```text
SCR = mean_t [exists valid non-adjacent upper-body contact at frame t]
```

## 7. Minimal Execution Order

对应 `BEAT2_PIPELINE.md` 的 Commands。

Baseline 完整链路：

```bash
conda activate gmr

python scripts/beat2_processing/build_emotion_manifest.py

python scripts/beat2_processing/batch_retarget_nao.py \
  --workers 16 \
  --backend gmr_baseline \
  --robot nao \
  --source_up_axis y

python scripts/beat2_processing/extract_source_laban_features.py \
  --workers 16 \
  --cache_root motion_data/BEAT2/eval_cache/source \
  --output_dir motion_data/BEAT2/features/source

python scripts/beat2_processing/extract_robot_laban_features.py \
  --workers 16 \
  --robot nao \
  --cache_root motion_data/BEAT2/eval_cache/gmr_baseline \
  --output_dir motion_data/BEAT2/features/gmr_baseline

python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/source/beat2_source_features.csv \
  --output_dir motion_data/BEAT2/anova/source

python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/anova/gmr_baseline

python scripts/beat2_processing/compute_efpr.py \
  --human_anova motion_data/BEAT2/anova/source/anova_main_table.csv \
  --robot_anova motion_data/BEAT2/anova/gmr_baseline/anova_main_table.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_baseline

python scripts/beat2_processing/bootstrap_efpr_ci.py \
  --source_features motion_data/BEAT2/features/source/beat2_source_features.csv \
  --robot_features motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/efpr/gmr_baseline \
  --n_bootstrap 1000

python scripts/beat2_processing/evaluate_nao_retargeting_metrics.py \
  --workers 8 \
  --robot nao \
  --source_cache_root motion_data/BEAT2/eval_cache/source \
  --robot_cache_root motion_data/BEAT2/eval_cache/gmr_baseline \
  --output_dir motion_data/BEAT2/retarget_metrics/gmr_baseline \
  --scale_sample_limit 0
```

Velocity backend 可复用同一链路，将 backend-specific 路径中的 `gmr_baseline` 替换为 `gmr_velocity`，并在 precompute 阶段传入：

```bash
--backend gmr_velocity
```

## 8. Current Code Assumptions

- `batch_retarget_nao.py` 默认 BEAT2 raw path 是 `/home/vergil/dataset/BEAT2/beat_english_v2.0.0/smplxflame_30`；如果本机路径不同，需要显式传入 `--src_root`。
- `build_emotion_manifest.py` 默认 BEAT2 root 是 `/home/vergil/dataset/BEAT2`；如果本机路径不同，需要显式传入 `--beat2_root`。该脚本的相对 `--output_dir` 按当前工作目录解析，因此建议从仓库根目录运行。
- SMPL-X model root 默认是 `assets/body_models`。
- Source cache 使用 `pelvis` 作为 reference。
- Robot cache 使用 NAO `torso` 作为 reference。
- 当前 feature extraction、ANOVA、EFPR 和 metric 脚本均以 cache / CSV 为输入，不会自动触发上游步骤。
- 除 `batch_retarget_nao.py --skip_existing` 这一种显式跳过模式外，当前各脚本写输出时使用 `np.savez` / `np.savez_compressed`、CSV `open("w")` 或 `Path.write_text()`，同名输出默认覆盖。
