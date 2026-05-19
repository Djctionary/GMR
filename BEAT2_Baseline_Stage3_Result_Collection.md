# BEAT2 Baseline 与 Stage3 Wrist 结果收集

生成日期：2026-05-09  
整理范围：当前 workspace 中 `motion_data/BEAT2` 已生成产物。  
包含 backend：`gmr_baseline`、`gmr_velocity_stage3_wrist_30`。  
排除 backend：`gmr_velocity`。

本文只做结果收集和重点摘录，不做新的机制解释或深入分析。算法流程与指标定义见 `BETA2_Experiment_Log.md` 和 `BEAT2_PIPELINE.md`。

## 1. 产物覆盖

| 产物 | 路径 | 数量 |
| --- | --- | ---: |
| emotion manifest | `motion_data/BEAT2/manifests/beat2_emotion_manifest.csv` | 1464 rows |
| converted AMASS-compatible npz | `motion_data/BEAT2/converted/*.npz` | 1464 |
| source eval cache | `motion_data/BEAT2/eval_cache/source/*.npz` | 1464 |
| baseline retargeted pkl | `motion_data/BEAT2/retargeted/gmr_baseline/*.pkl` | 1464 |
| baseline robot eval cache | `motion_data/BEAT2/eval_cache/gmr_baseline/*.npz` | 1464 |
| stage3 wrist retargeted pkl | `motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_30/*.pkl` | 1464 |
| stage3 wrist robot eval cache | `motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_30/*.npz` | 1464 |

两个目标 backend 均覆盖 manifest 中全部 1464 个 clip。

## 2. BEAT2 原始 / Source 侧

### 2.1 Manifest 总览

| 项 | 数值 |
| --- | ---: |
| clip 总数 | 1464 |
| speaker 数 | 25 |
| 有音频 clip | 1464 |
| frame min / mean / max | 534 / 1970.64 / 3504 |
| duration min / mean / max sec | 17.800 / 65.688 / 116.800 |
| total duration sec | 96167.705 |
| translation drift min / mean / max m | 0.015409 / 0.281546 / 1.731333 |
| problematic clips | `{}` |

| emotion | clips | total_sec | avg_sec | speakers |
| --- | ---: | ---: | ---: | ---: |
| neutral | 756 | 52278.398 | 69.151 | 25 |
| happiness | 104 | 6076.184 | 58.425 | 25 |
| anger | 102 | 5469.123 | 53.619 | 24 |
| sadness | 86 | 5817.482 | 67.645 | 25 |
| contempt | 104 | 7278.688 | 69.987 | 25 |
| surprise | 104 | 6045.490 | 58.130 | 25 |
| fear | 104 | 6691.354 | 64.340 | 25 |
| disgust | 104 | 6510.986 | 62.606 | 25 |

### 2.2 Source Laban Feature Means

文件：

- `motion_data/BEAT2/features/source/beat2_source_features.csv`
- `motion_data/BEAT2/features/source/beat2_source_feature_summary_by_emotion.csv`
- `motion_data/BEAT2/features/source/beat2_source_feature_errors.json`

Feature warning：3 个 clip 记录 `static_keypoints_excluded_in_space_windows`，累计 exclusion count 为 5；无 fatal error 摘录。

| emotion | W | Ti | S | F |
| --- | ---: | ---: | ---: | ---: |
| anger | 6.3443 | 70.336 | 0.223826 | 46.414 |
| contempt | 3.0913 | 41.860 | 0.235531 | 27.810 |
| disgust | 3.0249 | 42.609 | 0.232706 | 30.480 |
| fear | 3.8917 | 53.874 | 0.237566 | 31.577 |
| happiness | 4.5084 | 54.650 | 0.225594 | 37.905 |
| neutral | 2.7179 | 41.029 | 0.235123 | 27.295 |
| sadness | 2.0297 | 37.709 | 0.233411 | 22.286 |
| surprise | 3.8638 | 45.559 | 0.231705 | 32.718 |

### 2.3 Source ANOVA

文件：

- `motion_data/BEAT2/anova/source/anova_main_table.csv`
- `motion_data/BEAT2/anova/source/anova_shapiro_by_group.csv`
- `motion_data/BEAT2/anova/source/anova_tukey_hsd.csv`
- `motion_data/BEAT2/anova/source/anova_diagnostics.json`

| feature | p_oneway | p_welch | p_kruskal | eta2 | omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 6.94977e-16 | 4.83364e-11 | 1.96016e-10 | 0.057484 | 0.0529184 | 10 |
| Ti | 4.91783e-15 | 5.68509e-07 | 2.68829e-14 | 0.0547956 | 0.0502188 | 11 |
| S | 0.0695912 | 0.0975545 | 0.112463 | 0.00894503 | 0.0041775 | 0 |
| F | 1.18965e-41 | 6.36901e-20 | 2.354e-22 | 0.133724 | 0.129482 | 17 |

## 3. GMR Baseline 结果

### 3.1 Feature Summary

文件：

- `motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv`
- `motion_data/BEAT2/features/gmr_baseline/beat2_nao_feature_summary_by_emotion.csv`
- `motion_data/BEAT2/features/gmr_baseline/beat2_nao_feature_errors.json`

Feature warning：458 个 clip 记录 `static_keypoints_excluded_in_space_windows`，累计 exclusion count 为 6229；无 fatal error 摘录。

| emotion | W | Ti | S | F |
| --- | ---: | ---: | ---: | ---: |
| anger | 1.6435 | 37.399 | 0.257626 | 21.740 |
| contempt | 0.823003 | 23.668 | 0.272948 | 15.996 |
| disgust | 1.0897 | 28.435 | 0.268473 | 16.894 |
| fear | 0.863205 | 27.209 | 0.272608 | 15.419 |
| happiness | 1.0990 | 29.861 | 0.251826 | 20.706 |
| neutral | 0.816011 | 25.660 | 0.260648 | 16.078 |
| sadness | 0.694455 | 21.372 | 0.265232 | 12.409 |
| surprise | 0.971148 | 28.303 | 0.261589 | 17.530 |

### 3.2 ANOVA

文件：

- `motion_data/BEAT2/anova/gmr_baseline/anova_main_table.csv`
- `motion_data/BEAT2/anova/gmr_baseline/anova_shapiro_by_group.csv`
- `motion_data/BEAT2/anova/gmr_baseline/anova_tukey_hsd.csv`
- `motion_data/BEAT2/anova/gmr_baseline/anova_diagnostics.json`

| feature | p_oneway | p_welch | p_kruskal | eta2 | omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 3.8266e-06 | 8.03878e-07 | 6.11784e-06 | 0.0253613 | 0.0206617 | 5 |
| Ti | 2.28047e-09 | 6.61639e-10 | 3.18464e-08 | 0.0363923 | 0.0317385 | 7 |
| S | 0.00717447 | 0.0170325 | 0.0279067 | 0.0131756 | 0.00842557 | 2 |
| F | 8.22028e-17 | 1.10256e-13 | 2.73016e-11 | 0.0603995 | 0.0558462 | 14 |

### 3.3 EFPR

文件：

- `motion_data/BEAT2/efpr/gmr_baseline/efpr_summary.json`
- `motion_data/BEAT2/efpr/gmr_baseline/efpr_dimension_table.csv`
- `motion_data/BEAT2/efpr/gmr_baseline/efpr_bootstrap_summary.json`
- `motion_data/BEAT2/efpr/gmr_baseline/efpr_bootstrap_ci.csv`
- `motion_data/BEAT2/efpr/gmr_baseline/efpr_bootstrap_samples.csv`

EFPR aggregate：

| metric | point |
| --- | ---: |
| aggregate_efpr_eta_squared | 0.5096095603 |
| aggregate_efpr_omega_squared | 0.4739015217 |

Dimension EFPR：

| feature | EFPR_eta2 | EFPR_omega2 |
| --- | ---: | ---: |
| W | 0.441189 | 0.390444 |
| Ti | 0.664145 | 0.632005 |
| F | 0.451674 | 0.431305 |

Bootstrap CI：method 为 paired stratified bootstrap by emotion，`n_bootstrap=1000`，`n_pairs=1464`，`seed=20260502`。

| metric | point | ci_low | ci_high | mean | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| W_eta_squared | 0.441189 | 0.349845 | 0.904647 | 0.543979 | 1000 |
| W_omega_squared | 0.390444 | 0.302244 | 0.897646 | 0.506224 | 1000 |
| Ti_eta_squared | 0.664145 | 0.456583 | 1.0585 | 0.71068 | 1000 |
| Ti_omega_squared | 0.632005 | 0.417892 | 1.0664 | 0.685433 | 1000 |
| F_eta_squared | 0.451674 | 0.357236 | 0.578515 | 0.468597 | 1000 |
| F_omega_squared | 0.431305 | 0.333332 | 0.563367 | 0.448954 | 1000 |
| aggregate_eta_squared | 0.50961 | 0.417882 | 0.741177 | 0.56054 | 1000 |
| aggregate_omega_squared | 0.473902 | 0.379104 | 0.722887 | 0.531576 | 1000 |

### 3.4 Retarget Metrics

文件：

- `motion_data/BEAT2/retarget_metrics/gmr_baseline/nao_metric_config.json`
- `motion_data/BEAT2/retarget_metrics/gmr_baseline/nao_retarget_metrics_per_clip.csv`
- `motion_data/BEAT2/retarget_metrics/gmr_baseline/nao_retarget_metrics_summary_by_emotion.csv`
- `motion_data/BEAT2/retarget_metrics/gmr_baseline/nao_retarget_metrics_logs.json`

Metric config 摘要：

| item | value |
| --- | --- |
| jump_threshold_rad | 0.5 |
| scale | 0.31868066133536127 |
| scale_source | `auto_mean_arm_chain_length` |
| scale_sample_limit | 0 |
| SCR enabled | true |
| upper joints | `LShoulderPitch,LShoulderRoll,LElbowYaw,LElbowRoll,LWristYaw,RShoulderPitch,RShoulderRoll,RElbowYaw,RElbowRoll,RWristYaw` |

Overall：

| backend | MPJPE_mean | MPJPE_median | JJR_mean | max_jump_mean | SCR_mean | SCR_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gmr_baseline | 66.968 | 66.207 | 4.56485e-05 | 0.286056 | 0.230247 | 0.131907 |

By emotion：

| emotion | MPJPE | JJR | max_jump | SCR |
| --- | ---: | ---: | ---: | ---: |
| anger | 66.108 | 0.000159083 | 0.311003 | 0.173948 |
| contempt | 67.355 | 4.10123e-05 | 0.285716 | 0.176586 |
| disgust | 68.585 | 1.6959e-05 | 0.280437 | 0.215564 |
| fear | 67.356 | 4.72762e-05 | 0.27979 | 0.242767 |
| happiness | 68.146 | 0.000107845 | 0.335456 | 0.201567 |
| neutral | 66.890 | 3.24581e-05 | 0.280659 | 0.243519 |
| sadness | 64.539 | 0 | 0.246061 | 0.318847 |
| surprise | 66.816 | 3.7529e-05 | 0.296716 | 0.200234 |

Collision log 摘要：`nao_retarget_metrics_logs.json` 中 1388 个 clip 有 collision pair 记录。累计最多的 pair：

| pair | count |
| --- | ---: |
| r_wrist--torso | 471115 |
| LForeArm--torso | 331424 |
| RForeArm--torso | 298328 |
| l_wrist--torso | 292012 |
| l_wrist--r_wrist | 171455 |
| LForeArm--r_wrist | 37217 |
| RForeArm--l_wrist | 10289 |
| RBicep--l_wrist | 164 |

## 4. GMR Velocity Stage3 Wrist 结果

### 4.1 Feature Summary

文件：

- `motion_data/BEAT2/features/gmr_velocity_stage3_wrist_30/beat2_nao_features.csv`
- `motion_data/BEAT2/features/gmr_velocity_stage3_wrist_30/beat2_nao_feature_summary_by_emotion.csv`
- `motion_data/BEAT2/features/gmr_velocity_stage3_wrist_30/beat2_nao_feature_errors.json`

Feature warning：436 个 clip 记录 `static_keypoints_excluded_in_space_windows`，累计 exclusion count 为 5735；无 fatal error 摘录。

| emotion | W | Ti | S | F |
| --- | ---: | ---: | ---: | ---: |
| anger | 1.3360 | 35.125 | 0.245284 | 26.644 |
| contempt | 0.644875 | 24.784 | 0.258007 | 20.339 |
| disgust | 0.760485 | 27.311 | 0.254286 | 21.411 |
| fear | 0.702295 | 26.816 | 0.257192 | 20.377 |
| happiness | 0.814613 | 27.888 | 0.236273 | 25.585 |
| neutral | 0.607824 | 25.773 | 0.244292 | 21.310 |
| sadness | 0.501224 | 21.760 | 0.253894 | 16.514 |
| surprise | 0.698628 | 25.390 | 0.246407 | 21.474 |

### 4.2 ANOVA

文件：

- `motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_30/anova_main_table.csv`
- `motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_30/anova_shapiro_by_group.csv`
- `motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_30/anova_tukey_hsd.csv`
- `motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_30/anova_diagnostics.json`

| feature | p_oneway | p_welch | p_kruskal | eta2 | omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 1.78438e-05 | 1.18145e-05 | 1.17846e-05 | 0.0229872 | 0.0182777 | 6 |
| Ti | 1.91181e-06 | 0.000578952 | 0.000119668 | 0.0264191 | 0.0217239 | 7 |
| S | 0.000643831 | 0.00262826 | 0.00399531 | 0.0172657 | 0.0125326 | 2 |
| F | 1.04474e-12 | 2.7517e-12 | 1.60086e-10 | 0.0473477 | 0.0427397 | 15 |

### 4.3 EFPR

文件：

- `motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_30/efpr_summary.json`
- `motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_30/efpr_dimension_table.csv`
- `motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_30/efpr_bootstrap_summary.json`
- `motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_30/efpr_bootstrap_ci.csv`
- `motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_30/efpr_bootstrap_samples.csv`

EFPR aggregate：

| metric | point |
| --- | ---: |
| aggregate_efpr_eta_squared | 0.4086962098 |
| aggregate_efpr_omega_squared | 0.3667217766 |

Dimension EFPR：

| feature | EFPR_eta2 | EFPR_omega2 |
| --- | ---: | ---: |
| W | 0.399889 | 0.345395 |
| Ti | 0.482139 | 0.432585 |
| F | 0.354071 | 0.330082 |

Bootstrap CI：method 为 paired stratified bootstrap by emotion，`n_bootstrap=1000`，`n_pairs=1464`，`seed=20260502`。

| metric | point | ci_low | ci_high | mean | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| W_eta_squared | 0.399889 | 0.348665 | 0.802577 | 0.503734 | 1000 |
| W_omega_squared | 0.345395 | 0.295138 | 0.785533 | 0.46201 | 1000 |
| Ti_eta_squared | 0.482139 | 0.307073 | 0.767834 | 0.521717 | 1000 |
| Ti_omega_squared | 0.432585 | 0.241432 | 0.74121 | 0.476827 | 1000 |
| F_eta_squared | 0.354071 | 0.281774 | 0.481536 | 0.37408 | 1000 |
| F_omega_squared | 0.330082 | 0.251505 | 0.463248 | 0.350956 | 1000 |
| aggregate_eta_squared | 0.408696 | 0.339455 | 0.601814 | 0.456865 | 1000 |
| aggregate_omega_squared | 0.366722 | 0.293468 | 0.575694 | 0.419938 | 1000 |

### 4.4 Retarget Metrics

文件：

- `motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_30/nao_metric_config.json`
- `motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_30/nao_retarget_metrics_per_clip.csv`
- `motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_30/nao_retarget_metrics_summary_by_emotion.csv`
- `motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_30/nao_retarget_metrics_logs.json`

Metric config 摘要：

| item | value |
| --- | --- |
| jump_threshold_rad | 0.5 |
| scale | 0.31868066133245043 |
| scale_source | `auto_mean_arm_chain_length` |
| scale_sample_limit | 0 |
| SCR enabled | true |
| upper joints | `LShoulderPitch,LShoulderRoll,LElbowYaw,LElbowRoll,LWristYaw,RShoulderPitch,RShoulderRoll,RElbowYaw,RElbowRoll,RWristYaw` |

Overall：

| backend | MPJPE_mean | MPJPE_median | JJR_mean | max_jump_mean | SCR_mean | SCR_median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gmr_velocity_stage3_wrist_30 | 68.464 | 67.846 | 8.25426e-05 | 0.281543 | 0.214267 | 0.12417 |

By emotion：

| emotion | MPJPE | JJR | max_jump | SCR |
| --- | ---: | ---: | ---: | ---: |
| anger | 67.453 | 0.000198103 | 0.302595 | 0.166258 |
| contempt | 68.567 | 3.51525e-05 | 0.268086 | 0.159093 |
| disgust | 69.761 | 6.61009e-05 | 0.272869 | 0.196652 |
| fear | 68.561 | 2.73011e-05 | 0.26873 | 0.236075 |
| happiness | 69.975 | 0.000205277 | 0.335097 | 0.186903 |
| neutral | 68.531 | 6.91827e-05 | 0.280509 | 0.225714 |
| sadness | 65.613 | 2.68142e-05 | 0.242525 | 0.298784 |
| surprise | 68.312 | 0.000108744 | 0.282065 | 0.186601 |

Collision log 摘要：`nao_retarget_metrics_logs.json` 中 1392 个 clip 有 collision pair 记录。累计最多的 pair：

| pair | count |
| --- | ---: |
| r_wrist--torso | 461162 |
| RForeArm--torso | 277969 |
| LForeArm--torso | 270405 |
| l_wrist--torso | 232984 |
| l_wrist--r_wrist | 163787 |
| LForeArm--r_wrist | 32805 |
| RForeArm--l_wrist | 9057 |
| RBicep--l_wrist | 41 |

## 5. 结果文件索引

Source / raw analysis：

- `motion_data/BEAT2/manifests/beat2_emotion_manifest.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_group_stats.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_speaker_distribution.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_spot_check_samples.csv`
- `motion_data/BEAT2/manifests/beat2_emotion_problematic_clips.json`
- `motion_data/BEAT2/features/source/beat2_source_features.csv`
- `motion_data/BEAT2/features/source/beat2_source_feature_summary_by_emotion.csv`
- `motion_data/BEAT2/features/source/beat2_source_feature_errors.json`
- `motion_data/BEAT2/anova/source/anova_main_table.csv`
- `motion_data/BEAT2/anova/source/anova_shapiro_by_group.csv`
- `motion_data/BEAT2/anova/source/anova_tukey_hsd.csv`
- `motion_data/BEAT2/anova/source/anova_diagnostics.json`

Baseline：

- `motion_data/BEAT2/retargeted/gmr_baseline/`
- `motion_data/BEAT2/eval_cache/gmr_baseline/`
- `motion_data/BEAT2/features/gmr_baseline/`
- `motion_data/BEAT2/anova/gmr_baseline/`
- `motion_data/BEAT2/efpr/gmr_baseline/`
- `motion_data/BEAT2/retarget_metrics/gmr_baseline/`

Stage3 wrist：

- `motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_30/`
- `motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_30/`
- `motion_data/BEAT2/features/gmr_velocity_stage3_wrist_30/`
- `motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_30/`
- `motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_30/`
- `motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_30/`
