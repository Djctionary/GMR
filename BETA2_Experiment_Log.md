# BEAT2 情感保留实验记录（Section 1-5）

本文档记录当前 BEAT2 English 源侧与 NAO baseline 侧实验的核心流程、命令、路径、结果和由结果触发的方法决策。目标是为后续 Human-to-NAO retargeting 与 EFPR 计算建立稳定的数据与统计基础。

说明：本文档中的数值结果当前只作为流程参考。由于后续会基于首帧对齐修复重新跑一版全量结果，下面各节中的统计值、均值表和 EFPR 数字暂不作为最终定稿结果引用。

## Section 1：情感标签恢复与 Manifest 构建

### 数据源与范围

本阶段只使用 BEAT2 English Speech clips，不纳入 English Conversation。

- BEAT2 根目录：`/home/vergil/dataset/BEAT2/beat_english_v2.0.0`
- SMPL-X/FLAME 动作目录：`/home/vergil/dataset/BEAT2/beat_english_v2.0.0/smplxflame_30`
- 音频目录：`/home/vergil/dataset/BEAT2/beat_english_v2.0.0/wave16k`
- Manifest 脚本：`scripts/beat2_processing/build_emotion_manifest.py`

English Conversation 在 BEAT 协议中统一标为 neutral；为避免“对话场景”与“情感类别”混淆，当前 manifest 只保留 recording type `0` 的 English Speech。

### 情感映射规则

文件名形如 `10_kieks_0_103_103.npz`，其中：

- `10_kieks`：speaker
- `0`：English Speech
- `103_103`：question start/end id

采用 BEAT 官方协议恢复 8 类情感：

- `0-64`：neutral
- `65-72`：happiness
- `73-80`：anger
- `81-86`：sadness
- `87-94`：contempt
- `95-102`：surprise
- `103-110`：fear
- `111-118`：disgust

### 执行命令

```bash
python3 scripts/beat2_processing/build_emotion_manifest.py
```

### 输出文件

- `motion_data/BEAT2/manifests/beat2_emotion_manifest.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_group_stats.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_speaker_distribution.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_spot_check_samples.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_problematic_clips.json`

Manifest 保存 `clip_id`、`emotion`、`speaker_id`、`npz_filename`、`audio_filename`、`num_frames`、`duration_sec` 等下游所需字段；其中 `npz_filename` 可直接拼接到 BEAT2 `smplxflame_30` 目录定位原始动作。

### 核心结果

通过 sanity check 的 English Speech clips 共 `1464` 个，problematic clips 为 `0`。

| emotion | clips | avg duration (s) | speakers |
|---|---:|---:|---:|
| neutral | 756 | 69.151 | 25 |
| happiness | 104 | 58.425 | 25 |
| anger | 102 | 53.619 | 24 |
| sadness | 86 | 67.645 | 25 |
| contempt | 104 | 69.987 | 25 |
| surprise | 104 | 58.130 | 25 |
| fear | 104 | 64.340 | 25 |
| disgust | 104 | 62.606 | 25 |

最短 clip 为 `9_miranda_0_34_34`，`17.8s`，长于原先设定的 `5s / 150 frames` 过滤阈值，因此时长过滤没有删除任何样本。

### 决策记录

本机 `/home/vergil/dataset/BEAT` 是较旧/较小的 BEAT English v0.2.1 子集；BEAT2 English 中存在许多 BEAT 目录没有的 clips。因此后续实验以 BEAT2 为唯一数据源，不要求 clip 同时存在于旧 BEAT 目录。

## Section 2：SMPL-X 源侧 Laban 特征提取

### 目标

对每个 clip 提取 4 个 per-clip 标量：`W`、`Ti`、`S`、`F`。这些标量作为 Section 3 ANOVA 和后续 EFPR 分母的源侧特征。

### 实现路径

- 特征脚本：`scripts/beat2_processing/extract_source_laban_features.py`
- 输入 manifest：`motion_data/BEAT2/manifests/beat2_emotion_manifest.csv`
- 当前输出：
  - `motion_data/BEAT2/features/beat2_source_features.csv`
  - `motion_data/BEAT2/features/beat2_source_feature_summary_by_emotion.csv`
  - `motion_data/BEAT2/features/beat2_source_feature_errors.json`
- 旧整段 Space 版本备份：
  - `motion_data/BEAT2/features/beat2_source_features_fullclip_space_backup.csv`
  - `motion_data/BEAT2/features/beat2_source_feature_summary_by_emotion_fullclip_space_backup.csv`

### SMPL-X 轨迹恢复

BEAT2 `.npz` 使用 `poses` 字段：

- `global_orient = poses[:, :3]`
- `body_pose = poses[:, 3:66]`
- `transl = trans`
- `betas = betas[:16]`
- body model：统一使用 `neutral` SMPL-X body model

通过项目已有 `assets/body_models/smplx` 下的 `neutral` SMPL-X body model 做 forward，得到 `joints[T, 55, 3]`。当前实现不按 clip 单独切换 male/female body model，而是统一使用 neutral body model 配合该 clip 的 `betas`。当前只取上肢 6 个关节：

```text
16, 17, 18, 19, 20, 21
left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
```

参考系只做 pelvis 平移消除：

```text
p_rel_j(t) = p_j(t) - p_pelvis(t)
```

未做 pelvis 旋转矫正。原因是 BEAT2 说话者大多正面坐立，pelvis 朝向变化较小；旋转矫正可能引入额外噪声。

### 滤波与差分

在计算速度、加速度和 jerk 前，对 `[T, 6, 3]` 轨迹做 Butterworth 低通滤波：

- order：`4`
- cutoff：`6 Hz`
- fps：clip 内 `mocap_frame_rate`，通常 `30`
- 方法：`scipy.signal.filtfilt` 零相位滤波

差分公式：

```text
v(t) = [p(t+1) - p(t-1)] / (2 dt)
a(t) = [p(t+1) - 2p(t) + p(t-1)] / dt^2
jerk(t) = [p(t+2) - 2p(t+1) + 2p(t-1) - p(t-2)] / (2 dt^3)
```

### 四维特征公式

Weight 使用路径 1：先空间聚合，再取时间峰值。

```text
W = max_t sum_j 0.5 * ||v_j(t)||^2
```

Time 使用路径 1：先空间聚合，再取时间峰值。

```text
Ti = max_t sum_j ||a_j(t)||
```

Flow 使用 jerk RMS：

```text
F = sqrt(mean_{t,j} ||jerk_j(t)||^2)
```

Space 最初采用整段 directness：

```text
S_j = ||p_j(T) - p_j(0)|| / sum_t ||p_j(t+1) - p_j(t)||
S = mean_j S_j
```

但整段 directness 与 clip 时长显著负相关：

```text
S vs duration: r = -0.370, p = 1.36e-48
```

因此改为滑窗 directness：

```text
S_j(w) = ||p_j(t_end) - p_j(t_start)|| /
         sum_{t=t_start}^{t_end-1} ||p_j(t+1) - p_j(t)||

S(w) = mean_j S_j(w)
S = mean_w S(w)
```

窗口参数：

- window：`90` frames，即 `3s @ 30fps`
- stride：`45` frames，即 50% overlap
- 单窗口内 path length `< 0.01m` 的关键点剔除
- 短于 90 帧时整段作为一个窗口

滑窗改动后，`S` 与时长相关性降为：

```text
S vs duration: r = +0.019, p = 4.73e-01
```

说明长度耦合基本消除。

### 执行命令

全量提取使用：

```bash
conda activate gmr
python scripts/beat2_processing/extract_source_laban_features.py --workers 8
```

并行实现使用 `ProcessPoolExecutor + initializer + as_completed`。每个 worker 初始化一次 SMPL-X 模型，并设置 `torch.set_num_threads(1)`，避免 PyTorch 线程与多进程抢占 CPU。

### 源侧特征摘要

当前滑窗 Space 版本在一次全量运行中对 `1464` 个 clips 无错误。分情感均值如下，当前仅作参考：

| emotion | W mean | Ti mean | S mean | F mean |
|---|---:|---:|---:|---:|
| anger | 6.344 | 70.336 | 0.224 | 46.414 |
| contempt | 3.091 | 41.860 | 0.236 | 27.810 |
| disgust | 3.025 | 42.609 | 0.233 | 30.480 |
| fear | 3.892 | 53.874 | 0.238 | 31.577 |
| happiness | 4.508 | 54.650 | 0.226 | 37.905 |
| neutral | 2.718 | 41.029 | 0.235 | 27.295 |
| sadness | 2.030 | 37.709 | 0.233 | 22.286 |
| surprise | 3.864 | 45.559 | 0.232 | 32.718 |

### 决策记录

`W`、`Ti`、`F` 保留为峰值或 RMS 型强度指标，能反映 anger、happiness 等高能量情感与 sadness/neutral 的差异。`S` 改为滑窗版本后不再主要反映 clip 长短，后续可作为更干净的 Space 维度进入统计检验。

## Section 3：源侧情感可分性 ANOVA

### 目标

验证源侧 SMPL-X Laban 特征是否包含情感分组可解释的运动学差异。该步骤为后续 EFPR 的分母有效性提供依据。

### 实现路径

- ANOVA 脚本：`scripts/beat2_processing/run_anova.py`
- 输入：`motion_data/BEAT2/features/beat2_source_features.csv`
- 输出目录：`motion_data/BEAT2/anova`

输出文件：

- `motion_data/BEAT2/anova/anova_main_table.csv`
- `motion_data/BEAT2/anova/anova_shapiro_by_group.csv`
- `motion_data/BEAT2/anova/anova_tukey_hsd.csv`
- `motion_data/BEAT2/anova/anova_diagnostics.json`

### 统计项目

每个特征 `W`、`Ti`、`S`、`F` 分别计算：

- one-way ANOVA：`scipy.stats.f_oneway`
- Welch's ANOVA：`pingouin.welch_anova`
- Kruskal-Wallis：`scipy.stats.kruskal`
- Levene 方差齐性：`scipy.stats.levene`
- Shapiro-Wilk 分组正态性：`scipy.stats.shapiro`
- Tukey HSD：`statsmodels.stats.multicomp.pairwise_tukeyhsd`
- η² 与 ω²：基于 `statsmodels.formula.api.ols` 和 `sm.stats.anova_lm(..., typ=2)` 的 SS 手动计算

### 执行命令

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py
```

### 主结果

`anova_main_table.csv` 的核心结果如下，当前仅作参考：

| feature | p one-way | p Welch | p Kruskal | η² | ω² | Tukey significant pairs |
|---|---:|---:|---:|---:|---:|---:|
| W | 6.95e-16 | 4.83e-11 | 1.96e-10 | 0.0575 | 0.0529 | 10 |
| Ti | 4.92e-15 | 5.69e-07 | 2.69e-14 | 0.0548 | 0.0502 | 11 |
| S | 6.96e-02 | 9.76e-02 | 1.12e-01 | 0.0089 | 0.0042 | 0 |
| F | 1.19e-41 | 6.37e-20 | 2.35e-22 | 0.1337 | 0.1295 | 17 |

诊断文件 `anova_diagnostics.json` 显示四个特征均使用 `1464` 个样本，分组数量与 manifest 一致：

```text
neutral 756, happiness 104, anger 102, sadness 86,
contempt 104, surprise 104, fear 104, disgust 104
```

### 解释与后续决策

`W`、`Ti`、`F` 在 one-way、Welch 和 Kruskal-Wallis 三类检验中均显著，说明源侧 BEAT2 SMPL-X 上肢运动确实存在情感可分的强度与动态特征结构。其中 `F` 的效应量最大，`ω² = 0.1295`，后续可作为情感运动学保留的重要维度。

`S` 在滑窗修正后不显著，`p_welch = 0.0976`，且 Tukey 显著对数为 `0`。这说明原整段 Space 显著性很可能受 clip 长度耦合影响；当前版本将 `S` 保留为 EFPR 四维之一，但在论文中应说明其源侧情感判别性弱于 `W`、`Ti`、`F`。

Levene 检验在四个维度上均显著，Shapiro-Wilk 分组检验也显示明显非正态。因此后续报告应同时呈现 Welch ANOVA 和 Kruskal-Wallis，不只依赖 classic one-way ANOVA。

## Section 4：NAO Baseline Laban 特征提取

### 目标

对 Section 1 manifest 中的 `1464` 个 English Speech clips 运行 vanilla GMR 到 NAO 的 retargeting，并通过 MuJoCo FK 恢复 NAO 端上肢 6 点笛卡尔轨迹，计算与源侧完全同构的 `W`、`Ti`、`S`、`F` 四维 Laban 特征。

### Retargeting 路径

- 批量脚本：`scripts/beat2_processing/batch_retarget_nao.py`
- 输入 manifest：`motion_data/BEAT2/manifests/beat2_emotion_manifest.csv`
- BEAT2 SMPL-X 输入目录：`/home/vergil/dataset/BEAT2/beat_english_v2.0.0/smplxflame_30`
- converted 输出：`motion_data/BEAT2/converted`
- NAO motion 输出：`motion_data/BEAT2/retargeted`

执行命令：

```bash
conda activate gmr
python scripts/beat2_processing/batch_retarget_nao.py --workers 8
```

该脚本只按 manifest 处理 English Speech clips，不处理 English Conversation clips。BEAT2 坐标转换固定使用 `--source_up_axis y`，避免 NAO 可视化时整体躺倒。

当前 GMR fork 已包含 `use_velocity_limit` patch，但 `GeneralMotionRetargeting` 默认 `use_velocity_limit=False`，批量脚本也没有启用 velocity limit。因此本节结果定义为：

```text
patched GMR fork, velocity limit disabled, vanilla NAO baseline
```

### MuJoCo FK 与预处理

- robot-side 特征脚本：`scripts/beat2_processing/extract_robot_laban_features.py`
- 输入：`motion_data/BEAT2/retargeted/*_nao.pkl`
- 输出：
  - `motion_data/BEAT2/features/beat2_nao_features.csv`
  - `motion_data/BEAT2/features/beat2_nao_feature_summary_by_emotion.csv`
  - `motion_data/BEAT2/features/beat2_nao_feature_errors.json`

执行命令：

```bash
conda activate gmr
python scripts/beat2_processing/extract_robot_laban_features.py --workers 8
```

NAO 端 6 个 body 点：

```text
LShoulder, RShoulder, LElbow, RElbow, l_wrist, r_wrist
```

参考系采用 torso-relative：

```text
p_rel_j(t) = p_body_j(t) - p_torso(t)
```

参考系一致性判断：源侧当前使用 SMPL-X pelvis 作为 reference，NAO 侧使用 MJCF torso 作为 reference。根据 SMPL-X 参考位置图（https://www.researchgate.net/publication/380819663_DTP_learning_to_estimate_full-body_pose_in_real-time_from_sparse_VR_sensor_measurements/figures）和 NAO URDF/MJCF 中 torso 位置检查，NAO torso 在 Z 轴上略高于 SMPL-X pelvis，二者并非完全相同的解剖锚点。但 BEAT2 共语动作主要聚焦上半身，且说话/坐立姿态下 pelvis/torso 整体运动较小；当前实验只消除全局平移，不做缩放或额外旋转归一化。因此本阶段保留“各自骨架 trunk/root-relative reference”的设定，不额外把源侧 reference 改为 SMPL-X spine3 或 neck。若后续需要回应严格对照问题，可补做源侧 spine3/neck reference 的 sensitivity check。

后续滤波与特征计算直接复用源侧函数：

- `butter_lowpass_filter`
- `compute_laban_features`
- `compute_windowed_space`
- `make_feature_row`
- `write_summary`

因此 NAO 侧与源侧保持相同预处理：

- `6 Hz` Butterworth 低通，order `4`
- 中心差分计算速度 `v`
- 中心二阶差分计算加速度 `a`
- 五点中心差分计算 jerk
- Space 使用 `90` 帧滑窗、`45` stride、静止路径阈值 `0.01m`

### 输出格式一致性

`beat2_nao_features.csv` 与源侧 `beat2_source_features.csv` 字段完全一致：

```text
clip_id,emotion,speaker_id,num_frames,W,Ti,S,F
```

当前结果：

- feature rows：`1464 / 1464`
- error entries：`0`
- warning entries：`457`

warnings 来自 Space 滑窗中静止 keypoints 被剔除，不影响 `W`、`Ti`、`F`，且 `S` 仍有有效窗口时正常输出。

### NAO 特征摘要

`beat2_nao_feature_summary_by_emotion.csv` 的分情感均值如下，当前仅作参考：

| emotion | W mean | Ti mean | S mean | F mean |
|---|---:|---:|---:|---:|
| anger | 1.162 | 32.916 | 0.258 | 21.401 |
| contempt | 0.826 | 23.725 | 0.273 | 16.008 |
| disgust | 1.083 | 28.212 | 0.268 | 16.892 |
| fear | 0.858 | 26.910 | 0.272 | 15.415 |
| happiness | 1.079 | 29.359 | 0.251 | 20.693 |
| neutral | 0.810 | 25.589 | 0.261 | 16.079 |
| sadness | 0.690 | 21.356 | 0.265 | 12.415 |
| surprise | 0.961 | 27.985 | 0.262 | 17.529 |

### 帧对齐修复

此前 retarget `.pkl` 比源侧 clip 少 `1` 帧。原因是 `scripts/smplx_to_robot.py` 的 retarget loop 从第 `1` 帧开始处理，跳过了第 `0` 帧。例如源侧 `10_kieks_0_103_103` 为 `1913` 帧，旧 NAO 端输出为 `1912` 帧。

现已修复 `scripts/smplx_to_robot.py`：loop index 从 `-1` 初始化，使第一轮处理 frame `0`。因此当前代码版本在重新生成 NAO `.pkl` 后，robot frame 数应与源侧 frame 数一致。

为保证 Section 5 与后续 EFPR 完全可比，修复前生成的 NAO `.pkl`、`beat2_nao_features.csv` 和 `anova_nao` 结果应视为旧版参考结果；需要用 `--overwrite` 重新生成 NAO baseline，再重新提取 NAO Laban 特征并重跑 NAO ANOVA。

## Section 5：NAO Baseline 情感可分性 ANOVA

### 目标

使用与 Section 3 完全相同的 ANOVA 脚本，对 NAO baseline 特征表进行情感分组统计检验，得到 robot-side 的 `η²_robot,d` 与 `ω²_robot,d`。

### 执行命令

```bash
conda activate gmr
python scripts/beat2_processing/run_anova.py \
  --features_csv motion_data/BEAT2/features/beat2_nao_features.csv \
  --output_dir motion_data/BEAT2/anova_nao
```

输出文件：

- `motion_data/BEAT2/anova_nao/anova_main_table.csv`
- `motion_data/BEAT2/anova_nao/anova_shapiro_by_group.csv`
- `motion_data/BEAT2/anova_nao/anova_tukey_hsd.csv`
- `motion_data/BEAT2/anova_nao/anova_diagnostics.json`

### 主结果

`anova_nao/anova_main_table.csv` 的核心结果如下，当前仅作参考：

| feature | p one-way | p Welch | p Kruskal | η² | ω² | Tukey significant pairs |
|---|---:|---:|---:|---:|---:|---:|
| W | 6.18e-08 | 1.47e-08 | 7.12e-06 | 0.0316 | 0.0269 | 8 |
| Ti | 5.20e-08 | 1.61e-10 | 1.16e-07 | 0.0318 | 0.0271 | 7 |
| S | 7.17e-03 | 1.77e-02 | 2.54e-02 | 0.0132 | 0.0084 | 2 |
| F | 5.61e-16 | 2.04e-13 | 4.48e-11 | 0.0578 | 0.0532 | 14 |

诊断文件 `anova_nao/anova_diagnostics.json` 显示四个特征均使用 `1464` 个样本，分组数量与 Section 3 一致：

```text
neutral 756, happiness 104, anger 102, sadness 86,
contempt 104, surprise 104, fear 104, disgust 104
```

### 解释与后续决策

NAO baseline 上 `W`、`Ti`、`S`、`F` 在 one-way、Welch 和 Kruskal-Wallis 三类检验中均达到显著。与源侧相比，NAO 端 `W`、`Ti`、`F` 的效应量整体下降，其中 `F` 仍是 NAO 端最强的情感运动学维度，`ω² = 0.0532`。

`S` 在 NAO 端达到统计显著，但效应量仍小，`ω² = 0.0084`。考虑到源侧 `S` 不显著且效应量很低，后续 EFPR 报告中仍应把 `S` 作为辅助几何维度，而不是主要情感保留指标。

## Section 6：EFPR 计算

### 目标

计算 vanilla GMR NAO baseline 的 Emotion Feature Preservation Rate（EFPR）。主指标只使用 `W`、`Ti`、`F` 三个源侧情感可分性较强的动态维度，不纳入 `S`。

### 实现路径

- EFPR 脚本：`scripts/beat2_processing/compute_efpr.py`
- human/source 输入：`motion_data/BEAT2/anova/anova_main_table.csv`
- robot/NAO 输入：`motion_data/BEAT2/anova_nao/anova_main_table.csv`
- 输出目录：`motion_data/BEAT2/efpr`

执行命令：

```bash
python scripts/beat2_processing/compute_efpr.py
```

输出文件：

- `motion_data/BEAT2/efpr/efpr_dimension_table.csv`
- `motion_data/BEAT2/efpr/efpr_summary.json`
- `motion_data/BEAT2/efpr/efpr_bootstrap_ci.csv`
- `motion_data/BEAT2/efpr/efpr_bootstrap_samples.csv`
- `motion_data/BEAT2/efpr/efpr_bootstrap_summary.json`

### 计算公式

对每个维度 `d ∈ {W, Ti, F}`：

```text
EFPR_d = eta_squared_robot,d / eta_squared_human,d
```

`ω²` 版本同理：

```text
EFPR_d = omega_squared_robot,d / omega_squared_human,d
```

Aggregate EFPR 使用几何平均：

```text
EFPR = (EFPR_W * EFPR_Ti * EFPR_F)^(1/3)
```

使用几何平均的原因是 EFPR 是比值型指标，任一维度接近 `0` 时应显著惩罚 aggregate score，以反映“任一核心动态维度坍塌都会削弱整体情感保持”的语义。

### 当前结果（参考）

基于当前 `anova` 与 `anova_nao` 主表，dimension-wise EFPR 为：

| feature | EFPR η² | EFPR ω² |
|---|---:|---:|
| W | 0.5490 | 0.5081 |
| Ti | 0.5807 | 0.5405 |
| F | 0.4321 | 0.4110 |

Aggregate EFPR：

| effect size | aggregate EFPR |
|---|---:|
| η² | 0.5164 |
| ω² | 0.4833 |

### Bootstrap 95% CI

EFPR 是两个 effect size 的比值，没有简单解析分布，因此需要通过 bootstrap 估计采样不确定性。实现脚本：

```bash
python scripts/beat2_processing/bootstrap_efpr_ci.py --n_bootstrap 1000
```

Bootstrap 输入不是 ANOVA 主表，而是 per-clip 特征表：

- source：`motion_data/BEAT2/features/beat2_source_features.csv`
- robot：`motion_data/BEAT2/features/beat2_nao_features.csv`

方法采用 paired stratified bootstrap：

- 以 `clip_id` 对齐 source/robot，确保每次重采样取同一批 clips 的两侧特征。
- 在每个 emotion 组内有放回重采样，保持原始情感组大小不变。
- 每次 bootstrap 重新计算 source/robot 的 `η²`、`ω²`，再计算 `W`、`Ti`、`F` 以及 aggregate EFPR。
- 重复 `1000` 次，取 `2.5%` 和 `97.5%` 分位数作为 95% CI。

当前 `efpr_bootstrap_ci.csv` 核心结果如下，当前仅作参考：

| metric | point | 95% CI |
|---|---:|---:|
| EFPR_W η² | 0.5490 | [0.3661, 1.0748] |
| EFPR_Ti η² | 0.5807 | [0.4001, 0.9688] |
| EFPR_F η² | 0.4321 | [0.3441, 0.5661] |
| aggregate EFPR η² | 0.5164 | [0.4018, 0.7620] |
| EFPR_W ω² | 0.5081 | [0.3161, 1.0801] |
| EFPR_Ti ω² | 0.5405 | [0.3454, 0.9636] |
| EFPR_F ω² | 0.4110 | [0.3146, 0.5508] |
| aggregate EFPR ω² | 0.4833 | [0.3614, 0.7481] |

### 注意事项

当前 EFPR 与 bootstrap CI 是根据现有 `motion_data/BEAT2/features/beat2_nao_features.csv` 与 `motion_data/BEAT2/anova_nao/anova_main_table.csv` 计算得到。若因 Section 4 的首帧对齐修复而重新生成 NAO `.pkl`、`beat2_nao_features.csv` 和 `anova_nao`，则需要重新运行：

```bash
python scripts/beat2_processing/compute_efpr.py
python scripts/beat2_processing/bootstrap_efpr_ci.py --n_bootstrap 1000
```

以刷新 `motion_data/BEAT2/efpr` 下的 Section 6 结果。

## Section 7：Retargeting 标准指标实现

### 目标

在 EFPR 之外补充 humanoid retargeting 常用客观指标，避免只从情感保留角度评价方法。当前实现三个指标：

- scale-aligned upper-body MPJPE：几何 fidelity
- Joint Jump Rate：关节时间连续性
- Self-Collision Rate：物理可行性

### 实现脚本

统一脚本：

```bash
scripts/beat2_processing/evaluate_nao_retargeting_metrics.py
```

输入：

- manifest：`motion_data/BEAT2/manifests/beat2_emotion_manifest.csv`
- source converted SMPL-X：`motion_data/BEAT2/converted/*_amass_compat.npz`
- robot motion：`motion_data/BEAT2/retargeted/*_nao.pkl`
- SMPL-X body model：`assets/body_models`
- NAO MJCF：`assets/nao/nao_scene.xml`

输出：

- `motion_data/BEAT2/retarget_metrics/nao_retarget_metrics_per_clip.csv`
- `motion_data/BEAT2/retarget_metrics/nao_retarget_metrics_summary_by_emotion.csv`
- `motion_data/BEAT2/retarget_metrics/nao_retarget_metrics_logs.json`
- `motion_data/BEAT2/retarget_metrics/nao_metric_config.json`

建议全量运行命令：

```bash
conda activate gmr
python scripts/beat2_processing/evaluate_nao_retargeting_metrics.py \
  --workers 8 \
  --scale_sample_limit 0
```

如果只想先调试：

```bash
python scripts/beat2_processing/evaluate_nao_retargeting_metrics.py \
  --limit 3 \
  --workers 2 \
  --scale_sample_limit 3 \
  --output_dir motion_data/BEAT2/retarget_metrics_smoke
```

### MPJPE

MPJPE 使用与 Section 4 相同的 6 个上肢点：

```text
source SMPL-X: left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
robot NAO: LShoulder, RShoulder, LElbow, RElbow, l_wrist, r_wrist
```

两侧均转为各自 root-relative：

```text
source: p_j(t) - p_pelvis(t)
robot:  p_j(t) - p_torso(t)
```

坐标轴使用 retarget 前的 converted SMPL-X `.npz`，而不是 BEAT2 原始 `.npz`，确保 source 已应用 `source_up_axis=y` 到 Z-up 的转换。

为处理 SMPL-X 成人身体与 NAO 机器人尺寸差异，MPJPE 使用固定全局 morphology scale：

```text
s = mean(NAO shoulder-elbow-wrist chain length) /
    mean(SMPL-X shoulder-elbow-wrist chain length)
```

该 scale 只估计一次并写入 `nao_metric_config.json`，不做 per-clip 调参。

MPJPE 公式：

```text
MPJPE = mean_t mean_j || p_robot_j(t) - s * p_source_j(t) ||
```

### Joint Jump Rate

JJR 使用 NAO 上肢 10 个关节：

```text
LShoulderPitch, LShoulderRoll, LElbowYaw, LElbowRoll, LWristYaw,
RShoulderPitch, RShoulderRoll, RElbowYaw, RElbowRoll, RWristYaw
```

脚本通过 MuJoCo `jnt_qposadr` 自动映射到 `.pkl` 中的 `dof_pos` 列，不硬编码 index。

阈值采用 NMR 标准：

```text
tau = 0.5 rad
```

公式：

```text
JJR = count_t[max_j |q_j(t+1) - q_j(t)| > 0.5] / (T - 1)
```

同时输出每个 clip 的 `max_joint_jump_rad`，用于定位异常。

### Self-Collision Rate

已确认 NAO MJCF 可用于 contact 检测：

- 上肢 mesh geoms 存在。
- MuJoCo runtime 中相关上肢 geoms 的 `contype=1`、`conaffinity=1`。
- 可通过 `mj_forward` 后读取 `data.contact[:data.ncon]`。

SCR 统计 body 集合：

```text
torso,
LShoulder, LBicep, LForeArm, l_wrist,
RShoulder, RBicep, RForeArm, r_wrist
```

过滤掉结构性相邻接触：

```text
torso-LShoulder, torso-LBicep, LShoulder-LBicep,
LBicep-LForeArm, LForeArm-l_wrist,
torso-RShoulder, torso-RBicep, RShoulder-RBicep,
RBicep-RForeArm, RForeArm-r_wrist
```

公式：

```text
SCR = count_t[exists valid non-adjacent upper-body contact] / T
```

### Smoke Test 结果与注意事项

`--limit 3` smoke test 已通过，输出路径：

```text
motion_data/BEAT2/retarget_metrics_smoke
```

初步结果如下，当前仅作参考：

| clip | MPJPE (mm) | JJR | max joint jump (rad) | SCR |
|---|---:|---:|---:|---:|
| 10_kieks_0_103_103 | 69.67 | 0.000 | 0.244 | 0.449 |
| 10_kieks_0_104_104 | 68.40 | 0.000 | 0.227 | 0.235 |
| 10_kieks_0_10_10 | 64.52 | 0.000 | 0.187 | 0.711 |

MPJPE 与 JJR 路径工作正常，数值范围合理。SCR 能够计算，但由于当前 NAO MJCF 使用 visual mesh 同时作为 collision mesh，`forearm/wrist--torso` 接触可能包含 mesh false positives。SCR 在论文中应标注为 MuJoCo mesh-collision based estimate；正式报告前建议抽样检查 contact pairs 或可视化碰撞帧。

此前 smoke test 使用的是首帧修复前生成的旧 `.pkl`，因此当时日志里会出现 `source_frames = robot_frames + 1`。在当前代码版本下，首帧问题已修复；后续只需用 `batch_retarget_nao.py --overwrite` 重跑 NAO motion，再全量运行本节指标脚本即可刷新为新版本结果。

## 当前阶段结论

Section 1-7 已建立完整的源侧与 NAO baseline 对照链路：

```text
BEAT2 English Speech clips
-> emotion manifest
-> SMPL-X FK upper-body trajectories
-> filtered Laban features
-> source-side ANOVA validation
-> vanilla GMR NAO retargeting
-> MuJoCo FK torso-relative upper-body trajectories
-> NAO-side filtered Laban features
-> NAO-side ANOVA validation
-> EFPR dimension-wise and aggregate computation
-> retargeting standard metrics: MPJPE, Joint Jump Rate, Self-Collision Rate
```

当前结果支持将 EFPR 作为 baseline GMR 的情感运动学保持指标：`W`、`Ti`、`F` 三个动态维度均保留了部分源侧情感效应，但 robot-side 效应量相对 source-side 明显下降。后续若引入 velocity-constrained retargeting，可复用同一套 Section 4-7 脚本链路，同时比较 baseline 与改进版的 per-dimension EFPR、aggregate EFPR、MPJPE、JJR 与 SCR。
