import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from scipy.stats import gaussian_kde, spearmanr


BACKENDS = (
    ("0", "gmr_baseline"),
    ("1", "gmr_velocity_stage3_wrist_1"),
    ("5", "gmr_velocity_stage3_wrist_5"),
    ("10", "gmr_velocity_stage3_wrist_10"),
    ("30", "gmr_velocity_stage3_wrist_30"),
)

FEATURES = ("W", "Ti", "F")
FEATURE_NAMES = {
    "W": "W",
    "Ti": "Ti",
    "F": "F",
}
FEATURE_CN = {
    "W": "重量感 W",
    "Ti": "时间性 Ti",
    "F": "流畅性 F",
}

EMOTION_CN = {
    "neutral": "中性",
    "happiness": "高兴",
    "anger": "愤怒",
    "sadness": "悲伤",
    "contempt": "轻蔑",
    "surprise": "惊讶",
    "fear": "恐惧",
    "disgust": "厌恶",
}

LINE_COLORS = {
    "W": "#2563eb",
    "Ti": "#059669",
    "F": "#dc2626",
    "aggregate": "#111827",
}


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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def validate_inputs(root: Path) -> None:
    required = [
        root / "features" / "source" / "beat2_source_feature_summary_by_emotion.csv",
        root / "features" / "gmr_baseline" / "beat2_nao_feature_summary_by_emotion.csv",
    ]
    for _label, backend in BACKENDS:
        required.extend(
            [
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


def load_fonts() -> Tuple[FontProperties, FontProperties, FontProperties]:
    fonts_dir = repo_path("assets/fonts")
    serif_regular = fonts_dir / "SourceHanSerifSC" / "OTF" / "SimplifiedChinese" / "SourceHanSerifSC-Regular.otf"
    serif_bold = fonts_dir / "SourceHanSerifSC" / "OTF" / "SimplifiedChinese" / "SourceHanSerifSC-Bold.otf"
    times = fonts_dir / "Times New Roman.ttf"
    for path in (serif_regular, serif_bold, times):
        require_file(path)

    return (
        FontProperties(fname=str(serif_regular)),
        FontProperties(fname=str(serif_bold)),
        FontProperties(fname=str(times)),
    )


def configure_matplotlib() -> Tuple[FontProperties, FontProperties, FontProperties]:
    zh, zh_bold, latin = load_fonts()
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )
    return zh, zh_bold, latin


def apply_font(ax, zh: FontProperties, size: int = 9) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(zh)
        label.set_fontsize(size)
    ax.xaxis.label.set_fontproperties(zh)
    ax.yaxis.label.set_fontproperties(zh)
    ax.xaxis.label.set_fontsize(size + 1)
    ax.yaxis.label.set_fontsize(size + 1)


def style_axis(ax, zh: FontProperties) -> None:
    ax.grid(axis="y", color="#d4d4d8", linewidth=0.55, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_font(ax, zh)


def set_title(ax, text: str, zh_bold: FontProperties, size: int = 11) -> None:
    ax.set_title(text, loc="left", fontproperties=zh_bold, fontsize=size, pad=8)


def set_suptitle(fig, text: str, zh_bold: FontProperties) -> None:
    return None


def save_figure(fig, output_dir: Path, stem: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(str(svg_path), bbox_inches="tight")
    paths = [svg_path]
    plt.close(fig)
    for path in paths:
        print(f"[DONE] {path}")
    return paths


def metric_ci(root: Path, backend: str, metric: str) -> Tuple[float, float, float]:
    rows = read_csv(root / "efpr" / backend / "efpr_bootstrap_ci.csv")
    row = find_row(rows, "metric", metric)
    return float(row["point"]), float(row["ci_low_2_5"]), float(row["ci_high_97_5"])


def efpr_summary_rows(root: Path, effect: str) -> Dict[str, Dict[str, float]]:
    values = {}
    for label, backend in BACKENDS:
        summary = read_json(root / "efpr" / backend / "efpr_summary.json")
        dim = summary[f"dimension_efpr_{effect}"]
        values[label] = {
            "W": float(dim["W"]),
            "Ti": float(dim["Ti"]),
            "F": float(dim["F"]),
            "aggregate": float(summary[f"aggregate_efpr_{effect}"]),
        }
    return values


def load_bootstrap_samples(root: Path, backend: str, metric: str) -> np.ndarray:
    rows = read_csv(root / "efpr" / backend / "efpr_bootstrap_samples.csv")
    return np.asarray([float(row[metric]) for row in rows], dtype=float)


def plot_5_1(root: Path, output_dir: Path, zh: FontProperties, zh_bold: FontProperties) -> List[Path]:
    source = read_csv(root / "features" / "source" / "beat2_source_feature_summary_by_emotion.csv")
    target = read_csv(root / "features" / "gmr_baseline" / "beat2_nao_feature_summary_by_emotion.csv")
    source_by_emotion = {row["emotion"]: row for row in source}
    target_by_emotion = {row["emotion"]: row for row in target}
    emotions = sorted(source_by_emotion, key=lambda e: float(source_by_emotion[e]["W_mean"]), reverse=True)
    labels = [EMOTION_CN[e] for e in emotions]
    x = np.arange(len(emotions))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), sharex=True)
    for ax, feature in zip(axes, FEATURES):
        ax2 = ax.twinx()
        source_values = np.asarray([float(source_by_emotion[e][f"{feature}_mean"]) for e in emotions], dtype=float)
        target_values = np.asarray([float(target_by_emotion[e][f"{feature}_mean"]) for e in emotions], dtype=float)
        rho, _p = spearmanr(source_values, target_values)

        ax.bar(x - width / 2, source_values, width=width, color="#2563eb", alpha=0.88, label="源端")
        ax2.bar(x + width / 2, target_values, width=width, color="#f97316", alpha=0.88, label="目标端")
        set_title(ax, f"{FEATURE_CN[feature]}  Spearman ρ={rho:.3f}", zh_bold)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("源端均值", color="#2563eb", fontproperties=zh)
        ax2.set_ylabel("目标端均值", color="#f97316", fontproperties=zh)
        ax.tick_params(axis="y", colors="#2563eb")
        ax2.tick_params(axis="y", colors="#f97316")
        style_axis(ax, zh)
        apply_font(ax2, zh)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)

        if ax is axes[0]:
            handles = [
                plt.Rectangle((0, 0), 1, 1, color="#2563eb", alpha=0.88),
                plt.Rectangle((0, 0), 1, 1, color="#f97316", alpha=0.88),
            ]
            ax.legend(handles, ["源端", "目标端"], loc="upper right", frameon=False, prop=zh)

    set_suptitle(fig, "图 5.1 源端与目标端力效均值排序对比", zh_bold)
    return save_figure(fig, output_dir, "figure_5_1_source_target_feature_means")


def plot_5_2(root: Path, output_dir: Path, zh: FontProperties, zh_bold: FontProperties) -> List[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4), sharey=False)
    cmap_values = np.linspace(0.86, 0.22, len(BACKENDS))
    colors = [plt.cm.viridis(value) for value in cmap_values]

    xlims = {
        "W": (0.20, 1.20),
        "Ti": (0.20, 1.20),
        "F": (0.24, 0.64),
    }
    for ax, feature in zip(axes, FEATURES):
        xmin, xmax = xlims[feature]
        grid = np.linspace(xmin, xmax, 360)
        bins = np.linspace(xmin, xmax, 28)
        metric = f"{feature}_eta_squared"

        for color, (label, backend) in zip(colors, BACKENDS):
            samples = load_bootstrap_samples(root, backend, metric)
            point, low, high = metric_ci(root, backend, metric)
            finite = samples[np.isfinite(samples)]
            ax.hist(finite, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.08)
            ax.hist(finite, bins=bins, density=True, histtype="step", color=color, alpha=0.35, linewidth=0.75)
            kde = gaussian_kde(finite)
            ax.plot(grid, kde(grid), color=color, linewidth=1.8, label=f"w_s3={label}")
            ax.axvline(point, color=color, linewidth=1.3, alpha=0.92)
            ax.axvline(low, color=color, linewidth=0.95, linestyle="--", alpha=0.62)
            ax.axvline(high, color=color, linewidth=0.95, linestyle="--", alpha=0.62)

        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("EFPR η²", fontproperties=zh)
        ax.set_ylabel("密度", fontproperties=zh)
        set_title(ax, FEATURE_CN[feature], zh_bold)
        style_axis(ax, zh)

    axes[0].legend(loc="upper right", frameon=False, prop=zh, fontsize=8)
    set_suptitle(fig, "图 5.2 单维度 EFPR 自助分布", zh_bold)
    return save_figure(fig, output_dir, "figure_5_2_efpr_bootstrap_distribution")


def plot_5_3(root: Path, output_dir: Path, zh: FontProperties, zh_bold: FontProperties) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True)
    x = np.arange(len(BACKENDS))
    labels = [label for label, _backend in BACKENDS]
    metric_labels = {
        "W": "W",
        "Ti": "Ti",
        "F": "F",
        "aggregate": "聚合",
    }

    for ax, effect, effect_label in zip(axes, ("eta_squared", "omega_squared"), ("η²", "ω²")):
        values = efpr_summary_rows(root, effect)
        for metric in ("W", "Ti", "F", "aggregate"):
            y = np.asarray([values[label][metric] for label in labels], dtype=float)
            lows = []
            highs = []
            for _label, backend in BACKENDS:
                ci_metric = f"{metric}_{effect}" if metric != "aggregate" else f"aggregate_{effect}"
                point, low, high = metric_ci(root, backend, ci_metric)
                lows.append(point - low)
                highs.append(high - point)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([lows, highs]),
                color=LINE_COLORS[metric],
                marker="o",
                markersize=4.2,
                linewidth=1.5,
                capsize=3,
                label=metric_labels[metric],
            )
        ax.axhline(1.0, color="#52525b", linestyle=":", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("w_s3")
        ax.set_ylabel(f"EFPR {effect_label}")
        ax.set_ylim(0.2, 1.15)
        set_title(ax, f"EFPR 权重扫描（{effect_label}）", zh_bold)
        style_axis(ax, zh)
    axes[0].legend(loc="upper right", frameon=False, prop=zh, ncol=2)
    set_suptitle(fig, "图 5.3 EFPR 各维度随 w_s3 变化", zh_bold)
    return save_figure(fig, output_dir, "figure_5_3_efpr_weight_scan")


def load_overall_quality(root: Path) -> Dict[str, Dict[str, float]]:
    values = {}
    for label, backend in BACKENDS:
        rows = read_csv(root / "retarget_metrics" / backend / "nao_retarget_metrics_summary_by_emotion.csv")
        row = find_row(rows, "emotion", "ALL")
        values[label] = {
            "MPJPE": float(row["mpjpe_mm_mean"]),
            "JJR": float(row["joint_jump_rate_mean"]) * 1e5,
            "MaxJump": float(row["max_joint_jump_rad_mean"]),
            "SCR": float(row["self_collision_rate_mean"]),
        }
    return values


def plot_5_4(root: Path, output_dir: Path, zh: FontProperties, zh_bold: FontProperties) -> List[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), sharex=True)
    x = np.arange(len(BACKENDS))
    labels = [label for label, _backend in BACKENDS]
    values = load_overall_quality(root)
    panels = (
        ("MPJPE", "MPJPE (mm)", "#2563eb"),
        ("JJR", "JJR (×10^-5)", "#059669"),
        ("SCR", "SCR", "#dc2626"),
    )

    for ax, (metric, ylabel, color) in zip(axes, panels):
        y = np.asarray([values[label][metric] for label in labels], dtype=float)
        ax.plot(x, y, color=color, marker="o", markersize=4.5, linewidth=1.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("w_s3")
        ax.set_ylabel(ylabel)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        margin = max((ymax - ymin) * 0.22, 0.001)
        ax.set_ylim(ymin - margin, ymax + margin)
        set_title(ax, ylabel, zh_bold)
        style_axis(ax, zh)

    set_suptitle(fig, "图 5.4 几何质量指标随 w_s3 变化", zh_bold)
    return save_figure(fig, output_dir, "figure_5_4_retarget_quality_scan")


def plot_5_5(root: Path, output_dir: Path, zh: FontProperties, zh_bold: FontProperties) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    labels = [label for label, _backend in BACKENDS]
    efpr = efpr_summary_rows(root, "eta_squared")
    quality = load_overall_quality(root)
    colors = [plt.cm.viridis(value) for value in np.linspace(0.86, 0.22, len(BACKENDS))]
    y = np.asarray([efpr[label]["aggregate"] for label in labels], dtype=float)
    x_mpjpe = np.asarray([quality[label]["MPJPE"] for label in labels], dtype=float)
    x_scr = np.asarray([quality[label]["SCR"] for label in labels], dtype=float)

    for ax, x_values, xlabel in (
        (axes[0], x_mpjpe, "MPJPE (mm)"),
        (axes[1], x_scr, "SCR"),
    ):
        for color, label, x_value, y_value in zip(colors, labels, x_values, y):
            ax.scatter(x_value, y_value, s=58, color=color, edgecolor="#111827", linewidth=0.45, zorder=3)
            ax.annotate(
                f"w={label}",
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontproperties=zh,
                fontsize=8,
            )
        ax.plot(x_values, y, color="#71717a", linewidth=0.9, alpha=0.75, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("聚合 EFPR η²")
        set_title(ax, f"聚合 EFPR 与 {xlabel}", zh_bold)
        style_axis(ax, zh)

    set_suptitle(fig, "图 5.5 EFPR 与几何质量权衡关系", zh_bold)
    return save_figure(fig, output_dir, "figure_5_5_efpr_quality_tradeoff")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 5 publication figures.")
    parser.add_argument("--root", default="motion_data/BEAT2", help="BEAT2 artifact root.")
    parser.add_argument("--output_dir", default="outputs/chapter5_figures", help="Figure output directory.")
    parser.add_argument(
        "--skip_optional_5_5",
        action="store_true",
        help="Only generate the four figures from the confirmed design table.",
    )
    args = parser.parse_args()

    root = repo_path(args.root).resolve()
    output_dir = repo_path(args.output_dir).resolve()
    validate_inputs(root)
    zh, zh_bold, _latin = configure_matplotlib()

    generated = []
    generated.extend(plot_5_1(root, output_dir, zh, zh_bold))
    generated.extend(plot_5_2(root, output_dir, zh, zh_bold))
    generated.extend(plot_5_3(root, output_dir, zh, zh_bold))
    generated.extend(plot_5_4(root, output_dir, zh, zh_bold))
    if not args.skip_optional_5_5:
        generated.extend(plot_5_5(root, output_dir, zh, zh_bold))

    print(f"[DONE] generated {len(generated)} files in {output_dir}")


if __name__ == "__main__":
    main()
