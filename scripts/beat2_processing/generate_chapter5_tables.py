import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


EMOTION_RANGES = (
    ("0-64", "neutral"),
    ("65-72", "happiness"),
    ("73-80", "anger"),
    ("81-86", "sadness"),
    ("87-94", "contempt"),
    ("95-102", "surprise"),
    ("103-110", "fear"),
    ("111-118", "disgust"),
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

FEATURES_CORE = ("W", "Ti", "F")
FEATURES_ANOVA = ("W", "Ti", "S", "F")

BACKENDS = (
    ("0 (baseline)", "gmr_baseline"),
    ("1", "gmr_velocity_stage3_wrist_1"),
    ("5", "gmr_velocity_stage3_wrist_5"),
    ("10", "gmr_velocity_stage3_wrist_10"),
    ("30", "gmr_velocity_stage3_wrist_30"),
)


def repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(rows: List[Dict[str, str]], key: str, value: str) -> Dict[str, str]:
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError(f"Missing row where {key}={value}")


def fmt3(value) -> str:
    return f"{float(value):.3f}"


def fmt_p(value) -> str:
    number = float(value)
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def markdown_table(headers: List[str], rows: List[List[str]], numeric_from: int = 1) -> str:
    aligns = ["---"] * len(headers)
    for index in range(numeric_from, len(headers)):
        aligns[index] = "---:"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate_inputs(root: Path) -> None:
    required = [
        root / "manifests" / "beat2_emotion_group_stats.csv",
        root / "features" / "source" / "beat2_source_feature_summary_by_emotion.csv",
        root / "features" / "gmr_baseline" / "beat2_nao_feature_summary_by_emotion.csv",
        root / "anova" / "source" / "anova_main_table.csv",
    ]
    for _label, backend in BACKENDS:
        required.extend(
            [
                root / "features" / backend / "beat2_nao_feature_summary_by_emotion.csv",
                root / "efpr" / backend / "efpr_summary.json",
                root / "efpr" / backend / "efpr_bootstrap_ci.csv",
                root / "efpr" / backend / "efpr_bootstrap_samples.csv",
                root / "retarget_metrics" / backend / "nao_retarget_metrics_summary_by_emotion.csv",
            ]
        )
    for path in required:
        require_file(path)

    for _label, backend in BACKENDS:
        sample_path = root / "efpr" / backend / "efpr_bootstrap_samples.csv"
        sample_count = sum(1 for _ in sample_path.open()) - 1
        if sample_count != 1000:
            raise ValueError(f"Expected 1000 bootstrap samples in {sample_path}, got {sample_count}")

        ci_rows = read_csv(root / "efpr" / backend / "efpr_bootstrap_ci.csv")
        for metric in ("W_eta_squared", "Ti_eta_squared", "F_eta_squared", "aggregate_eta_squared"):
            row = find_row(ci_rows, "metric", metric)
            if int(row["finite_bootstrap_samples"]) != 1000:
                raise ValueError(f"Expected 1000 finite samples for {backend} {metric}")


def table_5_1() -> str:
    rows = [[interval, emotion] for interval, emotion in EMOTION_RANGES]
    return markdown_table(["编号区间", "情感类别"], rows, numeric_from=2)


def table_5_2(root: Path) -> str:
    stats = read_csv(root / "manifests" / "beat2_emotion_group_stats.csv")
    rows = []
    for emotion in EMOTION_ORDER:
        row = find_row(stats, "emotion", emotion)
        rows.append(
            [
                emotion,
                row["clip_count"],
                fmt3(row["total_duration_sec"]),
                fmt3(row["avg_duration_sec"]),
                row["speaker_count"],
            ]
        )
    return markdown_table(["情感", "片段数", "总时长(s)", "平均时长(s)", "说话者数"], rows)


def table_5_3(root: Path) -> str:
    source_rows = read_csv(root / "features" / "source" / "beat2_source_feature_summary_by_emotion.csv")
    target_rows = read_csv(
        root / "features" / "gmr_baseline" / "beat2_nao_feature_summary_by_emotion.csv"
    )
    source_by_emotion = {row["emotion"]: row for row in source_rows}
    target_by_emotion = {row["emotion"]: row for row in target_rows}

    emotions = sorted(EMOTION_ORDER, key=lambda emotion: float(source_by_emotion[emotion]["W_mean"]), reverse=True)
    rows = []
    for emotion in emotions:
        source = source_by_emotion[emotion]
        target = target_by_emotion[emotion]
        rows.append(
            [
                emotion,
                fmt3(source["W_mean"]),
                fmt3(target["W_mean"]),
                fmt3(source["Ti_mean"]),
                fmt3(target["Ti_mean"]),
                fmt3(source["S_mean"]),
                fmt3(target["S_mean"]),
                fmt3(source["F_mean"]),
                fmt3(target["F_mean"]),
            ]
        )
    return markdown_table(["情感", "W源", "W目", "Ti源", "Ti目", "S源", "S目", "F源", "F目"], rows)


def table_5_4(root: Path) -> str:
    anova = read_csv(root / "anova" / "source" / "anova_main_table.csv")
    rows = []
    for feature in FEATURES_ANOVA:
        row = find_row(anova, "feature", feature)
        rows.append(
            [
                feature,
                fmt_p(row["p_oneway"]),
                fmt_p(row["p_welch"]),
                fmt_p(row["p_kruskal"]),
                fmt3(row["eta_squared"]),
                fmt3(row["omega_squared"]),
                row["n_significant_pairs_tukey"],
            ]
        )
    return markdown_table(["维度", "单因素ANOVA p", "Welch p", "Kruskal-Wallis p", "η²", "ω²", "Tukey显著对数"], rows)


def table_5_5(root: Path, effect: str) -> str:
    if effect not in ("eta_squared", "omega_squared"):
        raise ValueError(effect)

    summary_key = f"aggregate_efpr_{effect}"
    dimension_key = f"dimension_efpr_{effect}"
    rows = []
    for label, backend in BACKENDS:
        summary = read_json(root / "efpr" / backend / "efpr_summary.json")
        dim = summary[dimension_key]
        rows.append(
            [
                label,
                fmt3(dim["W"]),
                fmt3(dim["Ti"]),
                fmt3(dim["F"]),
                fmt3(summary[summary_key]),
            ]
        )
    return markdown_table(["w_s3", "W", "Ti", "F", "聚合"], rows)


def table_5_6(root: Path) -> str:
    headers = ["w_s3"]
    for metric in ("W", "Ti", "F", "聚合"):
        headers.extend([f"{metric}点估计", f"{metric}置信区间"])

    metric_names = {
        "W": "W_eta_squared",
        "Ti": "Ti_eta_squared",
        "F": "F_eta_squared",
        "聚合": "aggregate_eta_squared",
    }
    rows = []
    for label, backend in BACKENDS:
        ci_rows = read_csv(root / "efpr" / backend / "efpr_bootstrap_ci.csv")
        row_values = [label]
        for metric in ("W", "Ti", "F", "聚合"):
            row = find_row(ci_rows, "metric", metric_names[metric])
            row_values.extend(
                [
                    fmt3(row["point"]),
                    f"[{fmt3(row['ci_low_2_5'])}, {fmt3(row['ci_high_97_5'])}]",
                ]
            )
        rows.append(row_values)
    return markdown_table(headers, rows)


def table_5_7(root: Path) -> str:
    rows = []
    for label, backend in BACKENDS:
        metrics = read_csv(root / "retarget_metrics" / backend / "nao_retarget_metrics_summary_by_emotion.csv")
        row = find_row(metrics, "emotion", "ALL")
        rows.append(
            [
                label,
                fmt3(row["mpjpe_mm_mean"]),
                fmt3(float(row["joint_jump_rate_mean"]) * 1e5),
                fmt3(row["max_joint_jump_rad_mean"]),
                fmt3(row["self_collision_rate_mean"]),
            ]
        )
    return markdown_table(["w_s3", "MPJPE(mm)", "JJR(×10^-5)", "最大跳变(rad)", "SCR"], rows)


def write_outputs(output_dir: Path, sections: List[Tuple[str, str, str]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_lines = [
        "# 第5章表格 Markdown 汇总",
        "",
        "说明：本文件仅包含表格内容；字体、行距和 Word 底纹在最终 doc/docx 排版阶段处理。",
        "数据范围：仅包含论文最终五个后端，完全排除中间实现目录。",
        "",
    ]
    for filename, title, content in sections:
        path = output_dir / filename
        path.write_text(f"## {title}\n\n{content}\n", encoding="utf-8")
        combined_lines.extend([f"## {title}", "", content, ""])

    combined_path = output_dir / "chapter5_tables.md"
    combined_path.write_text("\n".join(combined_lines).rstrip() + "\n", encoding="utf-8")
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 5 Markdown tables from BEAT2 artifacts.")
    parser.add_argument("--root", default="motion_data/BEAT2", help="BEAT2 artifact root.")
    parser.add_argument("--output_dir", default="outputs/chapter5_tables", help="Markdown output directory.")
    args = parser.parse_args()

    root = repo_path(args.root).resolve()
    output_dir = repo_path(args.output_dir).resolve()
    validate_inputs(root)

    sections = [
        ("table_5_1_emotion_id_ranges.md", "表 5.1 情感编号区间映射规则", table_5_1()),
        ("table_5_2_emotion_sample_distribution.md", "表 5.2 情感类别样本分布", table_5_2(root)),
        ("table_5_3_source_target_feature_means.md", "表 5.3 源端与目标端力效特征各情感类别均值", table_5_3(root)),
        ("table_5_4_source_anova_main.md", "表 5.4 源端方差分析主表", table_5_4(root)),
        ("table_5_5a_efpr_eta_squared.md", "表 5.5a EFPR 权重扫描（η²）", table_5_5(root, "eta_squared")),
        ("table_5_5b_efpr_omega_squared.md", "表 5.5b EFPR 权重扫描（ω²）", table_5_5(root, "omega_squared")),
        ("table_5_6_efpr_bootstrap_ci_eta_squared.md", "表 5.6 聚合 + 单维度 EFPR 自助 95% CI（η²）", table_5_6(root)),
        ("table_5_7_retarget_quality_scan.md", "表 5.7 几何质量指标权重扫描", table_5_7(root)),
    ]
    combined_path = write_outputs(output_dir, sections)

    print(f"[DONE] combined tables: {combined_path}")
    for filename, _title, _content in sections:
        print(f"[DONE] table: {output_dir / filename}")


if __name__ == "__main__":
    main()
