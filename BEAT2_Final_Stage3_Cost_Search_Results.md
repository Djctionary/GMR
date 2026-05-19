# BEAT2 Final Parameter Search Results

生成日期：2026-05-09

范围：当前 `motion_data/BEAT2` 中的最终实验结果。包含 `gmr_baseline` 与 Stage3 wrist cost search：`1, 5, 10, 30`。旧无后缀 `gmr_velocity_stage3_wrist` 结果目录已重命名为 `gmr_velocity_stage3_wrist_30`。

说明：代码中的算法 backend 仍是 `gmr_velocity_stage3_wrist`；带后缀的名字是输出 backend / 结果目录，用于区分不同 `velocity_stage3_cost`。

## 1. Artifact Coverage
| item | path | count |
| --- | --- | --- |
| manifest rows | motion_data/BEAT2/manifests/beat2_emotion_manifest.csv | 1464 |
| converted npz | motion_data/BEAT2/converted/*.npz | 1464 |
| source eval cache | motion_data/BEAT2/eval_cache/source/*.npz | 1464 |
| gmr_baseline retargeted pkl | motion_data/BEAT2/retargeted/gmr_baseline/*.pkl | 1464 |
| gmr_baseline robot eval cache | motion_data/BEAT2/eval_cache/gmr_baseline/*.npz | 1464 |
| gmr_velocity_stage3_wrist_1 retargeted pkl | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_1/*.pkl | 1464 |
| gmr_velocity_stage3_wrist_1 robot eval cache | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_1/*.npz | 1464 |
| gmr_velocity_stage3_wrist_5 retargeted pkl | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_5/*.pkl | 1464 |
| gmr_velocity_stage3_wrist_5 robot eval cache | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_5/*.npz | 1464 |
| gmr_velocity_stage3_wrist_10 retargeted pkl | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_10/*.pkl | 1464 |
| gmr_velocity_stage3_wrist_10 robot eval cache | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_10/*.npz | 1464 |
| gmr_velocity_stage3_wrist_30 retargeted pkl | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_30/*.pkl | 1464 |
| gmr_velocity_stage3_wrist_30 robot eval cache | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_30/*.npz | 1464 |

## 2. BEAT2 Source / Raw Summary
| item | value |
| --- | --- |
| clips | 1464 |
| speakers | 25 |
| clips with audio | 1464 |
| frames min / mean / max | 534 / 1970.64 / 3504 |
| duration sec min / mean / max | 17.800 / 65.688 / 116.800 |
| duration sec total | 96167.705 |
| translation drift m min / mean / max | 0.015409 / 0.281546 / 1.731333 |
| problematic clips | {} |

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

## 3. Source Laban And ANOVA

Source feature files:

- `motion_data/BEAT2/features/source/beat2_source_features.csv`
- `motion_data/BEAT2/features/source/beat2_source_feature_summary_by_emotion.csv`
- `motion_data/BEAT2/features/source/beat2_source_feature_errors.json`

| emotion | W_mean | Ti_mean | S_mean | F_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 6.3443 | 70.336 | 0.223826 | 46.414 |
| contempt | 3.0913 | 41.860 | 0.235531 | 27.810 |
| disgust | 3.0249 | 42.609 | 0.232706 | 30.480 |
| fear | 3.8917 | 53.874 | 0.237566 | 31.577 |
| happiness | 4.5084 | 54.650 | 0.225594 | 37.905 |
| neutral | 2.7179 | 41.029 | 0.235123 | 27.295 |
| sadness | 2.0297 | 37.709 | 0.233411 | 22.286 |
| surprise | 3.8638 | 45.559 | 0.231705 | 32.718 |

Source ANOVA files:

- `motion_data/BEAT2/anova/source/anova_main_table.csv`
- `motion_data/BEAT2/anova/source/anova_shapiro_by_group.csv`
- `motion_data/BEAT2/anova/source/anova_tukey_hsd.csv`
- `motion_data/BEAT2/anova/source/anova_diagnostics.json`

| feature | p_oneway | p_welch | p_kruskal | source_eta2 | source_omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 6.94977e-16 | 4.83364e-11 | 1.96016e-10 | 0.057484 | 0.0529184 | 10 |
| Ti | 4.91783e-15 | 5.68509e-07 | 2.68829e-14 | 0.0547956 | 0.0502188 | 11 |
| S | 0.0695912 | 0.0975545 | 0.112463 | 0.00894503 | 0.0041775 | 0 |
| F | 1.18965e-41 | 6.36901e-20 | 2.354e-22 | 0.133724 | 0.129482 | 17 |

## 4. Overall Retarget Metrics
| backend | stage3_cost | MPJPE_mean_mm | MPJPE_median_mm | JJR_mean | max_jump_mean_rad | SCR_mean | SCR_median | metric_scale |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gmr_baseline | baseline | 66.968 | 66.207 | 4.56485e-05 | 0.286056 | 0.230247 | 0.131907 | 0.318681 |
| gmr_velocity_stage3_wrist_1 | 1 | 66.978 | 66.231 | 4.13761e-05 | 0.275773 | 0.229884 | 0.131433 | 0.318681 |
| gmr_velocity_stage3_wrist_5 | 5 | 67.179 | 66.409 | 3.00992e-05 | 0.231692 | 0.224898 | 0.125772 | 0.318681 |
| gmr_velocity_stage3_wrist_10 | 10 | 67.610 | 66.888 | 4.83473e-05 | 0.236825 | 0.219085 | 0.121585 | 0.318681 |
| gmr_velocity_stage3_wrist_30 | 30 | 68.464 | 67.846 | 8.25426e-05 | 0.281543 | 0.214267 | 0.12417 | 0.318681 |

## 5. EFPR Summary

EFPR values are ratios of robot-side effect size to source-side effect size. Raw robot η²/ω² values are reported separately in Section 6.

| backend | stage3_cost | agg_EFPR_eta | agg_EFPR_omega | W_EFPR_eta | Ti_EFPR_eta | F_EFPR_eta | W_EFPR_omega | Ti_EFPR_omega | F_EFPR_omega |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gmr_baseline | baseline | 0.50961 | 0.473902 | 0.441189 | 0.664145 | 0.451674 | 0.390444 | 0.632005 | 0.431305 |
| gmr_velocity_stage3_wrist_1 | 1 | 0.5148 | 0.479537 | 0.445137 | 0.67624 | 0.453233 | 0.394751 | 0.645257 | 0.432922 |
| gmr_velocity_stage3_wrist_5 | 5 | 0.478453 | 0.440455 | 0.4298 | 0.583266 | 0.436903 | 0.378022 | 0.543387 | 0.415986 |
| gmr_velocity_stage3_wrist_10 | 10 | 0.450681 | 0.411524 | 0.416726 | 0.564652 | 0.389023 | 0.363761 | 0.522992 | 0.36633 |
| gmr_velocity_stage3_wrist_30 | 30 | 0.408696 | 0.366722 | 0.399889 | 0.482139 | 0.354071 | 0.345395 | 0.432585 | 0.330082 |

Baseline raw effect sizes and EFPR ratios:

| feature | source_eta2 | robot_eta2 | EFPR_eta2 | source_omega2 | robot_omega2 | EFPR_omega2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 0.057484 | 0.0253613 | 0.441189 | 0.0529184 | 0.0206617 | 0.390444 |
| Ti | 0.0547956 | 0.0363923 | 0.664145 | 0.0502188 | 0.0317385 | 0.632005 |
| F | 0.133724 | 0.0603995 | 0.451674 | 0.129482 | 0.0558462 | 0.431305 |

Bootstrap CI files are preserved in each `motion_data/BEAT2/efpr/<backend>/efpr_bootstrap_ci.csv`. The sweep uses the existing bootstrap settings from the backend pipeline.

### Bootstrap CI (95%, n=1000)

#### gmr_baseline
| metric | point | CI low (2.5%) | CI high (97.5%) | bootstrap_mean | bootstrap_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| W EFPR η² | 0.441189 | 0.349845 | 0.904647 | 0.543979 | 0.139022 |
| W EFPR ω² | 0.390444 | 0.302244 | 0.897646 | 0.506224 | 0.149269 |
| Ti EFPR η² | 0.664145 | 0.456583 | 1.058509 | 0.710680 | 0.150640 |
| Ti EFPR ω² | 0.632005 | 0.417892 | 1.066364 | 0.685433 | 0.165079 |
| F EFPR η² | 0.451674 | 0.357236 | 0.578515 | 0.468597 | 0.057926 |
| F EFPR ω² | 0.431305 | 0.333332 | 0.563367 | 0.448954 | 0.060802 |
| **agg EFPR η²** | **0.509610** | **0.417882** | **0.741177** | **0.560540** | **0.081632** |
| **agg EFPR ω²** | **0.473902** | **0.379104** | **0.722887** | **0.531576** | **0.087767** |

#### gmr_velocity_stage3_wrist_1 (cost=1)
| metric | point | CI low (2.5%) | CI high (97.5%) | bootstrap_mean | bootstrap_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| W EFPR η² | 0.445137 | 0.361454 | 0.946466 | 0.561891 | 0.148131 |
| W EFPR ω² | 0.394751 | 0.311513 | 0.942424 | 0.525497 | 0.159703 |
| Ti EFPR η² | 0.676240 | 0.474653 | 1.053758 | 0.723468 | 0.146462 |
| Ti EFPR ω² | 0.645257 | 0.433654 | 1.060854 | 0.699345 | 0.160556 |
| F EFPR η² | 0.453233 | 0.359371 | 0.580788 | 0.470089 | 0.057486 |
| F EFPR ω² | 0.432922 | 0.334879 | 0.565797 | 0.450498 | 0.060373 |
| **agg EFPR η²** | **0.514800** | **0.432265** | **0.750867** | **0.570456** | **0.081426** |
| **agg EFPR ω²** | **0.479537** | **0.392324** | **0.733870** | **0.542283** | **0.087697** |

#### gmr_velocity_stage3_wrist_5 (cost=5)
| metric | point | CI low (2.5%) | CI high (97.5%) | bootstrap_mean | bootstrap_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| W EFPR η² | 0.429800 | 0.366499 | 0.999820 | 0.577661 | 0.170221 |
| W EFPR ω² | 0.378022 | 0.316106 | 0.999804 | 0.542185 | 0.184650 |
| Ti EFPR η² | 0.583266 | 0.398784 | 0.882440 | 0.621058 | 0.120206 |
| Ti EFPR ω² | 0.543387 | 0.344839 | 0.869698 | 0.585895 | 0.130937 |
| F EFPR η² | 0.436903 | 0.354185 | 0.555780 | 0.453834 | 0.051926 |
| F EFPR ω² | 0.415986 | 0.327041 | 0.543099 | 0.433617 | 0.054957 |
| **agg EFPR η²** | **0.478453** | **0.417210** | **0.710233** | **0.539479** | **0.074229** |
| **agg EFPR ω²** | **0.440455** | **0.374258** | **0.693067** | **0.508183** | **0.080150** |

#### gmr_velocity_stage3_wrist_10 (cost=10)
| metric | point | CI low (2.5%) | CI high (97.5%) | bootstrap_mean | bootstrap_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| W EFPR η² | 0.416726 | 0.360157 | 0.899318 | 0.538933 | 0.139072 |
| W EFPR ω² | 0.363761 | 0.309339 | 0.891373 | 0.500159 | 0.150919 |
| Ti EFPR η² | 0.564652 | 0.365282 | 0.849458 | 0.596918 | 0.123196 |
| Ti EFPR ω² | 0.522992 | 0.307500 | 0.833862 | 0.559130 | 0.135034 |
| F EFPR η² | 0.389023 | 0.315572 | 0.509822 | 0.407360 | 0.050543 |
| F EFPR ω² | 0.366330 | 0.285992 | 0.492982 | 0.385444 | 0.053512 |
| **agg EFPR η²** | **0.450681** | **0.379120** | **0.669035** | **0.502632** | **0.069369** |
| **agg EFPR ω²** | **0.411524** | **0.336808** | **0.651509** | **0.469125** | **0.075380** |

#### gmr_velocity_stage3_wrist_30 (cost=30)
| metric | point | CI low (2.5%) | CI high (97.5%) | bootstrap_mean | bootstrap_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| W EFPR η² | 0.399889 | 0.348665 | 0.802577 | 0.503734 | 0.118021 |
| W EFPR ω² | 0.345395 | 0.295138 | 0.785533 | 0.462010 | 0.128228 |
| Ti EFPR η² | 0.482139 | 0.307073 | 0.767834 | 0.521717 | 0.120221 |
| Ti EFPR ω² | 0.432585 | 0.241432 | 0.741210 | 0.476827 | 0.131376 |
| F EFPR η² | 0.354071 | 0.281774 | 0.481536 | 0.374080 | 0.051277 |
| F EFPR ω² | 0.330082 | 0.251505 | 0.463248 | 0.350956 | 0.054138 |
| **agg EFPR η²** | **0.408696** | **0.339455** | **0.601814** | **0.456865** | **0.066157** |
| **agg EFPR ω²** | **0.366722** | **0.293468** | **0.575694** | **0.419938** | **0.072143** |

## 6. Robot ANOVA Main Table (Raw Robot Effect Sizes)

### gmr_baseline
| feature | p_oneway | p_welch | p_kruskal | robot_eta2 | robot_omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 3.8266e-06 | 8.03878e-07 | 6.11784e-06 | 0.0253613 | 0.0206617 | 5 |
| Ti | 2.28047e-09 | 6.61639e-10 | 3.18464e-08 | 0.0363923 | 0.0317385 | 7 |
| S | 0.00717447 | 0.0170325 | 0.0279067 | 0.0131756 | 0.00842557 | 2 |
| F | 8.22028e-17 | 1.10256e-13 | 2.73016e-11 | 0.0603995 | 0.0558462 | 14 |

### gmr_velocity_stage3_wrist_1
| feature | p_oneway | p_welch | p_kruskal | robot_eta2 | robot_omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 3.29855e-06 | 6.15996e-08 | 2.33089e-06 | 0.0255882 | 0.0208896 | 5 |
| Ti | 1.44329e-09 | 5.29594e-10 | 2.33867e-08 | 0.037055 | 0.032404 | 7 |
| S | 0.00678489 | 0.0163544 | 0.0266313 | 0.0132738 | 0.00852413 | 2 |
| F | 7.05291e-17 | 9.51504e-14 | 2.67854e-11 | 0.0606081 | 0.0560556 | 14 |

### gmr_velocity_stage3_wrist_5
| feature | p_oneway | p_welch | p_kruskal | robot_eta2 | robot_omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 5.86547e-06 | 9.7971e-10 | 7.19413e-08 | 0.0247066 | 0.0200043 | 6 |
| Ti | 4.71607e-08 | 5.21112e-06 | 4.96701e-06 | 0.0319604 | 0.0272882 | 7 |
| S | 0.00512008 | 0.0111111 | 0.016574 | 0.0137657 | 0.00901812 | 2 |
| F | 3.49605e-16 | 2.78633e-13 | 1.55524e-11 | 0.0584243 | 0.0538627 | 14 |

### gmr_velocity_stage3_wrist_10
| feature | p_oneway | p_welch | p_kruskal | robot_eta2 | robot_omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 9.5549e-06 | 3.66701e-07 | 8.00692e-07 | 0.0239551 | 0.0192497 | 6 |
| Ti | 9.39455e-08 | 0.000215869 | 4.48652e-05 | 0.0309405 | 0.026264 | 7 |
| S | 0.0013912 | 0.00409396 | 0.0063671 | 0.0159871 | 0.0112486 | 2 |
| F | 3.65903e-14 | 5.02834e-13 | 3.08498e-11 | 0.0520216 | 0.0474332 | 14 |

### gmr_velocity_stage3_wrist_30
| feature | p_oneway | p_welch | p_kruskal | robot_eta2 | robot_omega2 | tukey_pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W | 1.78438e-05 | 1.18145e-05 | 1.17846e-05 | 0.0229872 | 0.0182777 | 6 |
| Ti | 1.91181e-06 | 0.000578952 | 0.000119668 | 0.0264191 | 0.0217239 | 7 |
| S | 0.000643831 | 0.00262826 | 0.00399531 | 0.0172657 | 0.0125326 | 2 |
| F | 1.04474e-12 | 2.7517e-12 | 1.60086e-10 | 0.0473477 | 0.0427397 | 15 |

## 7. Robot Laban Feature Means By Emotion

### gmr_baseline
| emotion | W_mean | Ti_mean | S_mean | F_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 1.6435 | 37.399 | 0.257626 | 21.740 |
| contempt | 0.8230 | 23.668 | 0.272948 | 15.996 |
| disgust | 1.0897 | 28.435 | 0.268473 | 16.894 |
| fear | 0.8632 | 27.209 | 0.272608 | 15.419 |
| happiness | 1.0990 | 29.861 | 0.251826 | 20.706 |
| neutral | 0.8160 | 25.660 | 0.260648 | 16.078 |
| sadness | 0.6945 | 21.372 | 0.265232 | 12.409 |
| surprise | 0.9711 | 28.303 | 0.261589 | 17.530 |

### gmr_velocity_stage3_wrist_1
| emotion | W_mean | Ti_mean | S_mean | F_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 1.5586 | 35.961 | 0.258486 | 21.374 |
| contempt | 0.7635 | 22.846 | 0.273846 | 15.687 |
| disgust | 1.0106 | 27.143 | 0.269347 | 16.558 |
| fear | 0.8098 | 26.012 | 0.273489 | 15.162 |
| happiness | 1.0404 | 28.756 | 0.252552 | 20.287 |
| neutral | 0.7540 | 24.407 | 0.261483 | 15.792 |
| sadness | 0.6272 | 20.127 | 0.266137 | 12.173 |
| surprise | 0.8962 | 26.792 | 0.262497 | 17.139 |

### gmr_velocity_stage3_wrist_5
| emotion | W_mean | Ti_mean | S_mean | F_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 1.2672 | 32.117 | 0.259311 | 22.165 |
| contempt | 0.5723 | 20.813 | 0.275701 | 16.116 |
| disgust | 0.7058 | 23.334 | 0.270521 | 17.059 |
| fear | 0.6172 | 23.294 | 0.274584 | 16.332 |
| happiness | 0.7835 | 24.716 | 0.254025 | 20.619 |
| neutral | 0.5204 | 21.695 | 0.262136 | 16.710 |
| sadness | 0.3912 | 17.953 | 0.266941 | 13.052 |
| surprise | 0.6262 | 22.541 | 0.263799 | 17.400 |

### gmr_velocity_stage3_wrist_10
| emotion | W_mean | Ti_mean | S_mean | F_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 1.2830 | 33.790 | 0.252796 | 24.252 |
| contempt | 0.5833 | 22.460 | 0.268927 | 17.927 |
| disgust | 0.7023 | 25.024 | 0.263524 | 19.084 |
| fear | 0.6437 | 24.574 | 0.267410 | 18.231 |
| happiness | 0.7601 | 25.824 | 0.246114 | 22.771 |
| neutral | 0.5404 | 23.367 | 0.254342 | 18.840 |
| sadness | 0.4267 | 19.776 | 0.261519 | 14.638 |
| surprise | 0.6405 | 23.693 | 0.256801 | 19.154 |

### gmr_velocity_stage3_wrist_30
| emotion | W_mean | Ti_mean | S_mean | F_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 1.3360 | 35.125 | 0.245284 | 26.644 |
| contempt | 0.6449 | 24.784 | 0.258007 | 20.339 |
| disgust | 0.7605 | 27.311 | 0.254286 | 21.411 |
| fear | 0.7023 | 26.816 | 0.257192 | 20.377 |
| happiness | 0.8146 | 27.888 | 0.236273 | 25.585 |
| neutral | 0.6078 | 25.773 | 0.244292 | 21.310 |
| sadness | 0.5012 | 21.760 | 0.253894 | 16.514 |
| surprise | 0.6986 | 25.390 | 0.246407 | 21.474 |

## 8. Retarget Metrics By Emotion

### gmr_baseline
| emotion | MPJPE_mean_mm | JJR_mean | max_jump_mean_rad | SCR_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 66.108 | 0.000159083 | 0.311003 | 0.173948 |
| contempt | 67.355 | 4.10123e-05 | 0.285716 | 0.176586 |
| disgust | 68.585 | 1.6959e-05 | 0.280437 | 0.215564 |
| fear | 67.356 | 4.72762e-05 | 0.27979 | 0.242767 |
| happiness | 68.146 | 0.000107845 | 0.335456 | 0.201567 |
| neutral | 66.890 | 3.24581e-05 | 0.280659 | 0.243519 |
| sadness | 64.539 | 0 | 0.246061 | 0.318847 |
| surprise | 66.816 | 3.7529e-05 | 0.296716 | 0.200234 |

### gmr_velocity_stage3_wrist_1
| emotion | MPJPE_mean_mm | JJR_mean | max_jump_mean_rad | SCR_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 66.132 | 0.000156694 | 0.302137 | 0.173286 |
| contempt | 67.364 | 2.7249e-05 | 0.274735 | 0.17632 |
| disgust | 68.596 | 2.15158e-05 | 0.271272 | 0.214931 |
| fear | 67.365 | 2.72003e-05 | 0.269969 | 0.242415 |
| happiness | 68.154 | 9.56923e-05 | 0.326105 | 0.201299 |
| neutral | 66.899 | 3.05063e-05 | 0.270857 | 0.243182 |
| sadness | 64.547 | 5.04246e-06 | 0.236027 | 0.31847 |
| surprise | 66.824 | 3.11825e-05 | 0.279527 | 0.200049 |

### gmr_velocity_stage3_wrist_5
| emotion | MPJPE_mean_mm | JJR_mean | max_jump_mean_rad | SCR_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 66.333 | 7.7602e-05 | 0.260231 | 0.168703 |
| contempt | 67.511 | 3.43652e-06 | 0.224698 | 0.173162 |
| disgust | 68.719 | 2.22013e-05 | 0.225568 | 0.210852 |
| fear | 67.538 | 1.18449e-05 | 0.225151 | 0.237748 |
| happiness | 68.342 | 6.19091e-05 | 0.280778 | 0.196363 |
| neutral | 67.135 | 2.95015e-05 | 0.228554 | 0.237767 |
| sadness | 64.691 | 1.51274e-05 | 0.193885 | 0.311029 |
| surprise | 66.993 | 2.12393e-05 | 0.228344 | 0.196706 |

### gmr_velocity_stage3_wrist_10
| emotion | MPJPE_mean_mm | JJR_mean | max_jump_mean_rad | SCR_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 66.680 | 9.74777e-05 | 0.273436 | 0.166204 |
| contempt | 67.795 | 1.92945e-05 | 0.218966 | 0.169454 |
| disgust | 69.028 | 3.01898e-05 | 0.229585 | 0.205035 |
| fear | 67.856 | 6.19548e-06 | 0.231171 | 0.233195 |
| happiness | 69.026 | 9.32369e-05 | 0.285619 | 0.18975 |
| neutral | 67.609 | 5.25055e-05 | 0.234916 | 0.231503 |
| sadness | 65.045 | 1.44011e-05 | 0.201781 | 0.302662 |
| surprise | 67.389 | 4.24786e-05 | 0.225741 | 0.190473 |

### gmr_velocity_stage3_wrist_30
| emotion | MPJPE_mean_mm | JJR_mean | max_jump_mean_rad | SCR_mean |
| --- | ---: | ---: | ---: | ---: |
| anger | 67.453 | 0.000198103 | 0.302595 | 0.166258 |
| contempt | 68.567 | 3.51525e-05 | 0.268086 | 0.159093 |
| disgust | 69.761 | 6.61009e-05 | 0.272869 | 0.196652 |
| fear | 68.561 | 2.73011e-05 | 0.26873 | 0.236075 |
| happiness | 69.975 | 0.000205277 | 0.335097 | 0.186903 |
| neutral | 68.531 | 6.91827e-05 | 0.280509 | 0.225714 |
| sadness | 65.613 | 2.68142e-05 | 0.242525 | 0.298784 |
| surprise | 68.312 | 0.000108744 | 0.282065 | 0.186601 |

## 9. Feature Warning Summary
| dataset | file | entries | summary |
| --- | --- | --- | --- |
| source | features/source/beat2_source_feature_errors.json | 3 | static exclusion total=5 |
| gmr_baseline | features/gmr_baseline/beat2_nao_feature_errors.json | 458 | static exclusion total=6229 |
| gmr_velocity_stage3_wrist_1 | features/gmr_velocity_stage3_wrist_1/beat2_nao_feature_errors.json | 455 | static exclusion total=6237 |
| gmr_velocity_stage3_wrist_5 | features/gmr_velocity_stage3_wrist_5/beat2_nao_feature_errors.json | 459 | static exclusion total=6179 |
| gmr_velocity_stage3_wrist_10 | features/gmr_velocity_stage3_wrist_10/beat2_nao_feature_errors.json | 451 | static exclusion total=6015 |
| gmr_velocity_stage3_wrist_30 | features/gmr_velocity_stage3_wrist_30/beat2_nao_feature_errors.json | 436 | static exclusion total=5735 |

## 10. Result Directory Index
| backend | retargeted | eval_cache | features | anova | efpr | retarget_metrics |
| --- | --- | --- | --- | --- | --- | --- |
| gmr_baseline | motion_data/BEAT2/retargeted/gmr_baseline/ | motion_data/BEAT2/eval_cache/gmr_baseline/ | motion_data/BEAT2/features/gmr_baseline/ | motion_data/BEAT2/anova/gmr_baseline/ | motion_data/BEAT2/efpr/gmr_baseline/ | motion_data/BEAT2/retarget_metrics/gmr_baseline/ |
| gmr_velocity_stage3_wrist_1 | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_1/ | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_1/ | motion_data/BEAT2/features/gmr_velocity_stage3_wrist_1/ | motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_1/ | motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_1/ | motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_1/ |
| gmr_velocity_stage3_wrist_5 | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_5/ | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_5/ | motion_data/BEAT2/features/gmr_velocity_stage3_wrist_5/ | motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_5/ | motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_5/ | motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_5/ |
| gmr_velocity_stage3_wrist_10 | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_10/ | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_10/ | motion_data/BEAT2/features/gmr_velocity_stage3_wrist_10/ | motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_10/ | motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_10/ | motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_10/ |
| gmr_velocity_stage3_wrist_30 | motion_data/BEAT2/retargeted/gmr_velocity_stage3_wrist_30/ | motion_data/BEAT2/eval_cache/gmr_velocity_stage3_wrist_30/ | motion_data/BEAT2/features/gmr_velocity_stage3_wrist_30/ | motion_data/BEAT2/anova/gmr_velocity_stage3_wrist_30/ | motion_data/BEAT2/efpr/gmr_velocity_stage3_wrist_30/ | motion_data/BEAT2/retarget_metrics/gmr_velocity_stage3_wrist_30/ |
