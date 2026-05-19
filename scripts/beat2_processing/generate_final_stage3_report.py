import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


BACKENDS = (
    ("gmr_baseline", "baseline"),
    ("gmr_velocity_stage3_wrist_1", "1"),
    ("gmr_velocity_stage3_wrist_5", "5"),
    ("gmr_velocity_stage3_wrist_10", "10"),
    ("gmr_velocity_stage3_wrist_30", "30"),
)
EMOTION_ORDER = (
    "neutral",
    "happiness",
    "anger",
    "sadness",
    "contempt",
    "surprise",
    "fear",
    "disgust",
)
FEATURES = ("W", "Ti", "S", "F")


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_sig(value, digits=6):
    number = float(value)
    if number == 0:
        return "0"
    return f"{number:.{digits}g}"


def fmt_fixed(value, digits):
    return f"{float(value):.{digits}f}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---"] * (len(headers) - 1)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def numeric_table(headers, rows, text_cols=1):
    aligns = ["---"] * text_cols + ["---:"] * (len(headers) - text_cols)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def find_row(rows, key, value):
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError(f"Missing row where {key}={value}")


def warning_summary(path):
    data = read_json(path)
    if not data:
        return 0, "{}"

    warning_counter = Counter()
    warning_total = Counter()
    for item in data.values():
        for warning in item.get("warnings", []):
            name, _, value = warning.partition(":")
            warning_counter[name] += 1
            if value:
                warning_total[name] += int(value)

    if len(warning_counter) == 1:
        name = next(iter(warning_counter))
        total = warning_total.get(name, 0)
        if total:
            return len(data), f"static exclusion total={total}"
        return len(data), name

    parts = []
    for name in sorted(warning_counter):
        total = warning_total.get(name, 0)
        if total:
            parts.append(f"{name} total={total}")
        else:
            parts.append(name)
    return len(data), "; ".join(parts)


def validate_efpr_identity(root, backend, tolerance=5e-9):
    rows = read_csv(root / "efpr" / backend / "efpr_dimension_table.csv")
    for row in rows:
        for effect in ("eta_squared", "omega_squared"):
            human = float(row[f"human_{effect}"])
            robot = float(row[f"robot_{effect}"])
            efpr = float(row[f"efpr_{effect}"])
            expected = robot / human
            if not math.isclose(efpr, expected, rel_tol=tolerance, abs_tol=tolerance):
                feature = row["feature"]
                raise ValueError(
                    f"EFPR mismatch for {backend} {feature} {effect}: "
                    f"{efpr} != {robot} / {human}"
                )


def generate_report(root, generated_date):
    for backend, _stage3_cost in BACKENDS:
        validate_efpr_identity(root, backend)

    manifest = read_csv(root / "manifests" / "beat2_emotion_manifest.csv")
    group_stats = read_csv(root / "manifests" / "beat2_emotion_group_stats.csv")
    source_feature_summary = read_csv(
        root / "features" / "source" / "beat2_source_feature_summary_by_emotion.csv"
    )
    source_anova = read_csv(root / "anova" / "source" / "anova_main_table.csv")
    problematic = read_json(root / "manifests" / "beat2_emotion_problematic_clips.json")

    lines = [
        "# BEAT2 Final Parameter Search Results",
        "",
        f"生成日期：{generated_date}",
        "",
        "范围：当前 `motion_data/BEAT2` 中的最终实验结果。包含 `gmr_baseline` 与 Stage3 wrist cost search：`1, 5, 10, 30`。旧无后缀 `gmr_velocity_stage3_wrist` 结果目录已重命名为 `gmr_velocity_stage3_wrist_30`。",
        "",
        "说明：代码中的算法 backend 仍是 `gmr_velocity_stage3_wrist`；带后缀的名字是输出 backend / 结果目录，用于区分不同 `velocity_stage3_cost`。",
        "",
        "## 1. Artifact Coverage",
    ]

    coverage_rows = [
        [
            "manifest rows",
            "motion_data/BEAT2/manifests/beat2_emotion_manifest.csv",
            str(len(manifest)),
        ],
        [
            "converted npz",
            "motion_data/BEAT2/converted/*.npz",
            str(len(list((root / "converted").glob("*.npz")))),
        ],
        [
            "source eval cache",
            "motion_data/BEAT2/eval_cache/source/*.npz",
            str(len(list((root / "eval_cache" / "source").glob("*.npz")))),
        ],
    ]
    for backend, _stage3_cost in BACKENDS:
        coverage_rows.extend(
            [
                [
                    f"{backend} retargeted pkl",
                    f"motion_data/BEAT2/retargeted/{backend}/*.pkl",
                    str(len(list((root / "retargeted" / backend).glob("*.pkl")))),
                ],
                [
                    f"{backend} robot eval cache",
                    f"motion_data/BEAT2/eval_cache/{backend}/*.npz",
                    str(len(list((root / "eval_cache" / backend).glob("*.npz")))),
                ],
            ]
        )
    lines.extend(table(["item", "path", "count"], coverage_rows))

    frames = [int(row["num_frames"]) for row in manifest]
    durations = [float(row["duration_sec"]) for row in manifest]
    drift = [float(row["trans_drift_m"]) for row in manifest]
    speakers = {row["speaker_id"] for row in manifest}
    clips_with_audio = sum(row["has_audio"] == "True" for row in manifest)
    lines.extend(
        [
            "",
            "## 2. BEAT2 Source / Raw Summary",
        ]
    )
    lines.extend(
        table(
            ["item", "value"],
            [
                ["clips", str(len(manifest))],
                ["speakers", str(len(speakers))],
                ["clips with audio", str(clips_with_audio)],
                [
                    "frames min / mean / max",
                    f"{min(frames)} / {sum(frames) / len(frames):.2f} / {max(frames)}",
                ],
                [
                    "duration sec min / mean / max",
                    f"{min(durations):.3f} / {sum(durations) / len(durations):.3f} / {max(durations):.3f}",
                ],
                ["duration sec total", f"{sum(durations):.3f}"],
                [
                    "translation drift m min / mean / max",
                    f"{min(drift):.6f} / {sum(drift) / len(drift):.6f} / {max(drift):.6f}",
                ],
                ["problematic clips", json.dumps(problematic, sort_keys=True)],
            ],
        )
    )
    lines.append("")
    group_rows = []
    for emotion in EMOTION_ORDER:
        row = find_row(group_stats, "emotion", emotion)
        group_rows.append(
            [
                emotion,
                row["clip_count"],
                fmt_fixed(row["total_duration_sec"], 3),
                fmt_fixed(row["avg_duration_sec"], 3),
                row["speaker_count"],
            ]
        )
    lines.extend(numeric_table(["emotion", "clips", "total_sec", "avg_sec", "speakers"], group_rows))

    lines.extend(
        [
            "",
            "## 3. Source Laban And ANOVA",
            "",
            "Source feature files:",
            "",
            "- `motion_data/BEAT2/features/source/beat2_source_features.csv`",
            "- `motion_data/BEAT2/features/source/beat2_source_feature_summary_by_emotion.csv`",
            "- `motion_data/BEAT2/features/source/beat2_source_feature_errors.json`",
            "",
        ]
    )
    source_feature_rows = []
    for emotion in ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"):
        row = find_row(source_feature_summary, "emotion", emotion)
        source_feature_rows.append(
            [
                emotion,
                fmt_fixed(row["W_mean"], 4),
                fmt_fixed(row["Ti_mean"], 3),
                fmt_fixed(row["S_mean"], 6),
                fmt_fixed(row["F_mean"], 3),
            ]
        )
    lines.extend(numeric_table(["emotion", "W_mean", "Ti_mean", "S_mean", "F_mean"], source_feature_rows))
    lines.extend(
        [
            "",
            "Source ANOVA files:",
            "",
            "- `motion_data/BEAT2/anova/source/anova_main_table.csv`",
            "- `motion_data/BEAT2/anova/source/anova_shapiro_by_group.csv`",
            "- `motion_data/BEAT2/anova/source/anova_tukey_hsd.csv`",
            "- `motion_data/BEAT2/anova/source/anova_diagnostics.json`",
            "",
        ]
    )
    source_anova_rows = []
    for feature in FEATURES:
        row = find_row(source_anova, "feature", feature)
        source_anova_rows.append(
            [
                feature,
                fmt_sig(row["p_oneway"]),
                fmt_sig(row["p_welch"]),
                fmt_sig(row["p_kruskal"]),
                fmt_sig(row["eta_squared"]),
                fmt_sig(row["omega_squared"]),
                row["n_significant_pairs_tukey"],
            ]
        )
    lines.extend(
        numeric_table(
            ["feature", "p_oneway", "p_welch", "p_kruskal", "source_eta2", "source_omega2", "tukey_pairs"],
            source_anova_rows,
        )
    )

    lines.extend(["", "## 4. Overall Retarget Metrics"])
    overall_rows = []
    for backend, stage3_cost in BACKENDS:
        summary = read_csv(root / "retarget_metrics" / backend / "nao_retarget_metrics_summary_by_emotion.csv")
        config = read_json(root / "retarget_metrics" / backend / "nao_metric_config.json")
        row = find_row(summary, "emotion", "ALL")
        overall_rows.append(
            [
                backend,
                stage3_cost,
                fmt_fixed(row["mpjpe_mm_mean"], 3),
                fmt_fixed(row["mpjpe_mm_median"], 3),
                fmt_sig(row["joint_jump_rate_mean"]),
                fmt_sig(row["max_joint_jump_rad_mean"]),
                fmt_sig(row["self_collision_rate_mean"]),
                fmt_sig(row["self_collision_rate_median"]),
                fmt_sig(config["scale"]),
            ]
        )
    lines.extend(
        numeric_table(
            [
                "backend",
                "stage3_cost",
                "MPJPE_mean_mm",
                "MPJPE_median_mm",
                "JJR_mean",
                "max_jump_mean_rad",
                "SCR_mean",
                "SCR_median",
                "metric_scale",
            ],
            overall_rows,
            text_cols=2,
        )
    )

    lines.extend(
        [
            "",
            "## 5. EFPR Summary",
            "",
            "EFPR values are ratios of robot-side effect size to source-side effect size. Raw robot η²/ω² values are reported separately in Section 6.",
            "",
        ]
    )
    efpr_rows = []
    for backend, stage3_cost in BACKENDS:
        summary = read_json(root / "efpr" / backend / "efpr_summary.json")
        eta = summary["dimension_efpr_eta_squared"]
        omega = summary["dimension_efpr_omega_squared"]
        efpr_rows.append(
            [
                backend,
                stage3_cost,
                fmt_sig(summary["aggregate_efpr_eta_squared"]),
                fmt_sig(summary["aggregate_efpr_omega_squared"]),
                fmt_sig(eta["W"]),
                fmt_sig(eta["Ti"]),
                fmt_sig(eta["F"]),
                fmt_sig(omega["W"]),
                fmt_sig(omega["Ti"]),
                fmt_sig(omega["F"]),
            ]
        )
    lines.extend(
        numeric_table(
            [
                "backend",
                "stage3_cost",
                "agg_EFPR_eta",
                "agg_EFPR_omega",
                "W_EFPR_eta",
                "Ti_EFPR_eta",
                "F_EFPR_eta",
                "W_EFPR_omega",
                "Ti_EFPR_omega",
                "F_EFPR_omega",
            ],
            efpr_rows,
            text_cols=2,
        )
    )
    lines.extend(
        [
            "",
            "Baseline raw effect sizes and EFPR ratios:",
            "",
        ]
    )
    baseline_efpr_rows = []
    for row in read_csv(root / "efpr" / "gmr_baseline" / "efpr_dimension_table.csv"):
        baseline_efpr_rows.append(
            [
                row["feature"],
                fmt_sig(row["human_eta_squared"]),
                fmt_sig(row["robot_eta_squared"]),
                fmt_sig(row["efpr_eta_squared"]),
                fmt_sig(row["human_omega_squared"]),
                fmt_sig(row["robot_omega_squared"]),
                fmt_sig(row["efpr_omega_squared"]),
            ]
        )
    lines.extend(
        numeric_table(
            [
                "feature",
                "source_eta2",
                "robot_eta2",
                "EFPR_eta2",
                "source_omega2",
                "robot_omega2",
                "EFPR_omega2",
            ],
            baseline_efpr_rows,
        )
    )
    lines.extend(
        [
            "",
            "Bootstrap CI files are preserved in each `motion_data/BEAT2/efpr/<backend>/efpr_bootstrap_ci.csv`. The sweep uses the existing bootstrap settings from the backend pipeline.",
            "",
            "### Bootstrap CI (95%, n=1000)",
        ]
    )
    metric_labels = {
        "W_eta_squared": "W EFPR η²",
        "W_omega_squared": "W EFPR ω²",
        "Ti_eta_squared": "Ti EFPR η²",
        "Ti_omega_squared": "Ti EFPR ω²",
        "F_eta_squared": "F EFPR η²",
        "F_omega_squared": "F EFPR ω²",
        "aggregate_eta_squared": "**agg EFPR η²**",
        "aggregate_omega_squared": "**agg EFPR ω²**",
    }
    for backend, stage3_cost in BACKENDS:
        heading = f"#### {backend}" if stage3_cost == "baseline" else f"#### {backend} (cost={stage3_cost})"
        lines.extend(["", heading])
        ci_rows = []
        for row in read_csv(root / "efpr" / backend / "efpr_bootstrap_ci.csv"):
            label = metric_labels[row["metric"]]
            is_aggregate = row["metric"].startswith("aggregate")
            values = [
                fmt_fixed(row["point"], 6),
                fmt_fixed(row["ci_low_2_5"], 6),
                fmt_fixed(row["ci_high_97_5"], 6),
                fmt_fixed(row["bootstrap_mean"], 6),
                fmt_fixed(row["bootstrap_std"], 6),
            ]
            if is_aggregate:
                values = [f"**{value}**" for value in values]
            ci_rows.append([label, *values])
        lines.extend(
            numeric_table(
                ["metric", "point", "CI low (2.5%)", "CI high (97.5%)", "bootstrap_mean", "bootstrap_std"],
                ci_rows,
            )
        )

    lines.extend(["", "## 6. Robot ANOVA Main Table (Raw Robot Effect Sizes)"])
    for backend, _stage3_cost in BACKENDS:
        lines.extend(["", f"### {backend}"])
        anova_rows = []
        for feature in FEATURES:
            row = find_row(read_csv(root / "anova" / backend / "anova_main_table.csv"), "feature", feature)
            anova_rows.append(
                [
                    feature,
                    fmt_sig(row["p_oneway"]),
                    fmt_sig(row["p_welch"]),
                    fmt_sig(row["p_kruskal"]),
                    fmt_sig(row["eta_squared"]),
                    fmt_sig(row["omega_squared"]),
                    row["n_significant_pairs_tukey"],
                ]
            )
        lines.extend(
            numeric_table(
                ["feature", "p_oneway", "p_welch", "p_kruskal", "robot_eta2", "robot_omega2", "tukey_pairs"],
                anova_rows,
            )
        )

    lines.extend(["", "## 7. Robot Laban Feature Means By Emotion"])
    for backend, _stage3_cost in BACKENDS:
        lines.extend(["", f"### {backend}"])
        rows = []
        summary = read_csv(root / "features" / backend / "beat2_nao_feature_summary_by_emotion.csv")
        for emotion in ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"):
            row = find_row(summary, "emotion", emotion)
            rows.append(
                [
                    emotion,
                    fmt_fixed(row["W_mean"], 4),
                    fmt_fixed(row["Ti_mean"], 3),
                    fmt_fixed(row["S_mean"], 6),
                    fmt_fixed(row["F_mean"], 3),
                ]
            )
        lines.extend(numeric_table(["emotion", "W_mean", "Ti_mean", "S_mean", "F_mean"], rows))

    lines.extend(["", "## 8. Retarget Metrics By Emotion"])
    for backend, _stage3_cost in BACKENDS:
        lines.extend(["", f"### {backend}"])
        rows = []
        summary = read_csv(root / "retarget_metrics" / backend / "nao_retarget_metrics_summary_by_emotion.csv")
        for emotion in ("anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"):
            row = find_row(summary, "emotion", emotion)
            rows.append(
                [
                    emotion,
                    fmt_fixed(row["mpjpe_mm_mean"], 3),
                    fmt_sig(row["joint_jump_rate_mean"]),
                    fmt_sig(row["max_joint_jump_rad_mean"]),
                    fmt_sig(row["self_collision_rate_mean"]),
                ]
            )
        lines.extend(numeric_table(["emotion", "MPJPE_mean_mm", "JJR_mean", "max_jump_mean_rad", "SCR_mean"], rows))

    lines.extend(["", "## 9. Feature Warning Summary"])
    warning_rows = []
    source_entries, source_summary = warning_summary(root / "features" / "source" / "beat2_source_feature_errors.json")
    warning_rows.append(
        [
            "source",
            "features/source/beat2_source_feature_errors.json",
            str(source_entries),
            source_summary,
        ]
    )
    for backend, _stage3_cost in BACKENDS:
        entries, summary = warning_summary(root / "features" / backend / "beat2_nao_feature_errors.json")
        warning_rows.append(
            [
                backend,
                f"features/{backend}/beat2_nao_feature_errors.json",
                str(entries),
                summary,
            ]
        )
    lines.extend(table(["dataset", "file", "entries", "summary"], warning_rows))

    lines.extend(["", "## 10. Result Directory Index"])
    index_rows = []
    for backend, _stage3_cost in BACKENDS:
        index_rows.append(
            [
                backend,
                f"motion_data/BEAT2/retargeted/{backend}/",
                f"motion_data/BEAT2/eval_cache/{backend}/",
                f"motion_data/BEAT2/features/{backend}/",
                f"motion_data/BEAT2/anova/{backend}/",
                f"motion_data/BEAT2/efpr/{backend}/",
                f"motion_data/BEAT2/retarget_metrics/{backend}/",
            ]
        )
    lines.extend(table(["backend", "retargeted", "eval_cache", "features", "anova", "efpr", "retarget_metrics"], index_rows))
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate or validate the final BEAT2 Stage3 cost-search report from motion_data artifacts."
    )
    parser.add_argument("--motion-data-root", default="motion_data/BEAT2")
    parser.add_argument("--output", default="BEAT2_Final_Stage3_Cost_Search_Results.md")
    parser.add_argument("--generated-date", default="2026-05-09")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    motion_data_root = (repo_root / args.motion_data_root).resolve()
    output_path = (repo_root / args.output).resolve()
    generated = generate_report(motion_data_root, args.generated_date)

    if args.check:
        current = output_path.read_text(encoding="utf-8")
        if current != generated:
            raise SystemExit(f"{output_path} is out of date with motion_data artifacts")
        print(f"[OK] {output_path} matches motion_data artifacts")
        return

    output_path.write_text(generated, encoding="utf-8")
    print(f"[DONE] wrote {output_path}")


if __name__ == "__main__":
    main()
