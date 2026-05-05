import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES_CORE = ["W", "Ti", "F"]
FEATURE_LABELS = {
    "W": "Weight / Energy",
    "Ti": "Time / Suddenness",
    "F": "Flow / Jerk",
}
EMOTION_ORDER = [
    "neutral",
    "happiness",
    "anger",
    "sadness",
    "contempt",
    "surprise",
    "fear",
    "disgust",
]
EMOTION_COLORS = {
    "neutral": "#6b7280",
    "happiness": "#f59e0b",
    "anger": "#dc2626",
    "sadness": "#2563eb",
    "contempt": "#7c3aed",
    "surprise": "#0891b2",
    "fear": "#16a34a",
    "disgust": "#9333ea",
}
SIDE_COLORS = {"Source": "#1f77b4", "Robot": "#ff7f0e"}


def repo_path(path: str) -> Path:
    return Path(__file__).resolve().parents[2] / path


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] missing: {path}")
        return None
    return pd.read_csv(path)


def style_axes(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] {path}")


def ordered_emotion_frame(df: pd.DataFrame) -> pd.DataFrame:
    present = [emotion for emotion in EMOTION_ORDER if emotion in set(df["emotion"])]
    return df.set_index("emotion").loc[present].reset_index()


def feature_profile(summary_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    df = ordered_emotion_frame(summary_df)
    for feature in features:
        values = df[f"{feature}_mean"].astype(float).to_numpy()
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if std == 0 or not math.isfinite(std):
            z_values = np.zeros_like(values)
        else:
            z_values = (values - mean) / std
        for emotion, value, z_value in zip(df["emotion"], values, z_values):
            rows.append(
                {
                    "emotion": emotion,
                    "feature": feature,
                    "mean": value,
                    "z": z_value,
                }
            )
    return pd.DataFrame(rows)


def plot_emotion_profiles(source_summary: pd.DataFrame, robot_summary: pd.DataFrame, output_dir: Path, backend: str) -> Path:
    source_profile = feature_profile(source_summary, FEATURES_CORE)
    source_profile["side"] = "Source"
    robot_profile = feature_profile(robot_summary, FEATURES_CORE)
    robot_profile["side"] = "Robot"
    combined = pd.concat([source_profile, robot_profile], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    x = np.arange(len([emotion for emotion in EMOTION_ORDER if emotion in set(combined["emotion"])]))
    width = 0.36

    for ax, feature in zip(axes, FEATURES_CORE):
        feature_df = combined[combined["feature"] == feature]
        source_values = feature_df[feature_df["side"] == "Source"]["z"].to_numpy()
        robot_values = feature_df[feature_df["side"] == "Robot"]["z"].to_numpy()
        labels = feature_df[feature_df["side"] == "Source"]["emotion"].tolist()
        ax.bar(x - width / 2, source_values, width, label="Source", color=SIDE_COLORS["Source"])
        ax.bar(x + width / 2, robot_values, width, label="Robot", color=SIDE_COLORS["Robot"])
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        style_axes(
            ax,
            f"Section 3-4: {FEATURE_LABELS[feature]} emotion profile",
            "z-scored mean" if ax is axes[0] else None,
        )

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        f"Core Laban Emotion Profiles: Source vs NAO ({backend})",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    output_path = output_dir / f"01_laban_emotion_profiles_{backend}.png"
    save_figure(fig, output_path)
    return output_path


def plot_anova_effects(source_anova: pd.DataFrame, robot_anova: pd.DataFrame, output_dir: Path, backend: str) -> Path:
    source = source_anova[source_anova["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]
    robot = robot_anova[robot_anova["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]

    x = np.arange(len(FEATURES_CORE))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar(x - width / 2, source["eta_squared"], width, label="Source", color=SIDE_COLORS["Source"])
    ax.bar(x + width / 2, robot["eta_squared"], width, label="Robot", color=SIDE_COLORS["Robot"])
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES_CORE], rotation=15, ha="right")
    style_axes(ax, "Section 5: Emotion separability effect size", "ANOVA eta squared")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(source["eta_squared"].max(), robot["eta_squared"].max()) * 1.25)
    output_path = output_dir / f"02_anova_effect_sizes_{backend}.png"
    save_figure(fig, output_path)
    return output_path


def plot_efpr(efpr_df: pd.DataFrame, ci_df: pd.DataFrame | None, output_dir: Path, backend: str) -> Path:
    df = efpr_df[efpr_df["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]
    values = df["efpr_eta_squared"].astype(float).to_numpy()
    low = None
    high = None
    if ci_df is not None:
        ci_rows = []
        for feature in FEATURES_CORE:
            metric = f"{feature}_eta_squared"
            match = ci_df[ci_df["metric"] == metric]
            if match.empty:
                ci_rows = []
                break
            ci_rows.append(match.iloc[0])
        if ci_rows:
            low = np.array([row["ci_low_2_5"] for row in ci_rows], dtype=float)
            high = np.array([row["ci_high_97_5"] for row in ci_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = np.arange(len(FEATURES_CORE))
    errors = None
    if low is not None and high is not None:
        errors = np.vstack([values - low, high - values])
    ax.bar(x, values, color=["#4f46e5", "#059669", "#db2777"], yerr=errors, capsize=5)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, label="Robot = Source")
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURES_CORE], rotation=15, ha="right")
    style_axes(ax, "Section 6: Emotion Feature Preservation Rate", "EFPR eta squared ratio")
    ax.set_ylim(0, max(1.05, np.nanmax(values) * 1.35))
    ax.legend(frameon=False)
    output_path = output_dir / f"03_efpr_preservation_{backend}.png"
    save_figure(fig, output_path)
    return output_path


def plot_retarget_quality(metrics_df: pd.DataFrame, output_dir: Path, backend: str) -> Path:
    df = ordered_emotion_frame(metrics_df[metrics_df["emotion"] != "ALL"].copy())
    colors = [EMOTION_COLORS.get(emotion, "#64748b") for emotion in df["emotion"]]
    labels = df["emotion"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2))
    axes = axes.ravel()

    axes[0].bar(x, df["mpjpe_mm_mean"], color=colors)
    axes[0].errorbar(x, df["mpjpe_mm_mean"], yerr=df["mpjpe_mm_std"], fmt="none", ecolor="#111827", capsize=3, linewidth=0.8)
    style_axes(axes[0], "Section 7: Geometry fidelity", "MPJPE mean (mm)")

    axes[1].bar(x, df["joint_jump_rate_mean"] * 1e4, color=colors)
    style_axes(axes[1], "Section 7: Temporal continuity", "JJR mean (x1e-4)")

    axes[2].bar(x, df["max_joint_jump_rad_mean"], color=colors)
    axes[2].errorbar(x, df["max_joint_jump_rad_mean"], yerr=df["max_joint_jump_rad_std"], fmt="none", ecolor="#111827", capsize=3, linewidth=0.8)
    style_axes(axes[2], "Section 7: Worst local joint change", "Max joint jump mean (rad)")

    axes[3].bar(x, df["self_collision_rate_mean"] * 100.0, color=colors)
    axes[3].errorbar(x, df["self_collision_rate_mean"] * 100.0, yerr=df["self_collision_rate_std"] * 100.0, fmt="none", ecolor="#111827", capsize=3, linewidth=0.8)
    style_axes(axes[3], "Section 7: Mesh self-collision estimate", "SCR mean (%)")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")

    fig.suptitle(
        f"Retargeting Quality Metrics by Emotion ({backend})",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    output_path = output_dir / f"04_retarget_quality_{backend}.png"
    save_figure(fig, output_path)
    return output_path


def plot_core_dashboard(
    source_anova: pd.DataFrame,
    robot_anova: pd.DataFrame,
    efpr_df: pd.DataFrame | None,
    retarget_metrics: pd.DataFrame | None,
    output_dir: Path,
    backend: str,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
    axes = axes.ravel()

    source = source_anova[source_anova["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]
    robot = robot_anova[robot_anova["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]
    x = np.arange(len(FEATURES_CORE))
    width = 0.36
    axes[0].bar(x - width / 2, source["eta_squared"], width, label="Source", color=SIDE_COLORS["Source"])
    axes[0].bar(x + width / 2, robot["eta_squared"], width, label="Robot", color=SIDE_COLORS["Robot"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(FEATURES_CORE)
    axes[0].legend(frameon=False)
    style_axes(axes[0], "A. Source signal and robot signal", "eta squared")

    if efpr_df is not None:
        efpr = efpr_df[efpr_df["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]
        axes[1].bar(x, efpr["efpr_eta_squared"], color=["#4f46e5", "#059669", "#db2777"])
        axes[1].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(FEATURES_CORE)
        style_axes(axes[1], "B. Preservation after retargeting", "EFPR eta ratio")
    else:
        axes[1].axis("off")

    if retarget_metrics is not None:
        all_row = retarget_metrics[retarget_metrics["emotion"] == "ALL"].iloc[0]
        values = [
            float(all_row["mpjpe_mm_mean"]),
            float(all_row["joint_jump_rate_mean"]) * 1e4,
            float(all_row["max_joint_jump_rad_mean"]),
            float(all_row["self_collision_rate_mean"]) * 100.0,
        ]
        labels = ["MPJPE\nmm", "JJR\nx1e-4", "Max jump\nrad", "SCR\n%"]
        axes[2].bar(np.arange(len(labels)), values, color=["#2563eb", "#16a34a", "#f59e0b", "#dc2626"])
        axes[2].set_xticks(np.arange(len(labels)))
        axes[2].set_xticklabels(labels)
        style_axes(axes[2], "C. Overall robot quality", "mean value")

        emotion_df = ordered_emotion_frame(retarget_metrics[retarget_metrics["emotion"] != "ALL"].copy())
        axes[3].scatter(
            emotion_df["mpjpe_mm_mean"],
            emotion_df["self_collision_rate_mean"] * 100.0,
            s=80,
            color=[EMOTION_COLORS.get(e, "#64748b") for e in emotion_df["emotion"]],
        )
        for _, row in emotion_df.iterrows():
            axes[3].annotate(row["emotion"], (row["mpjpe_mm_mean"], row["self_collision_rate_mean"] * 100.0), fontsize=8, xytext=(4, 3), textcoords="offset points")
        style_axes(axes[3], "D. Geometry vs collision trade-off", "SCR mean (%)")
        axes[3].set_xlabel("MPJPE mean (mm)")
    else:
        axes[2].axis("off")
        axes[3].axis("off")

    fig.suptitle(
        f"BEAT2 -> NAO Core Pipeline Dashboard ({backend})",
        fontsize=16,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    output_path = output_dir / f"00_core_dashboard_{backend}.png"
    save_figure(fig, output_path)
    return output_path


def write_summary_markdown(
    output_dir: Path,
    backend: str,
    source_anova: pd.DataFrame | None,
    robot_anova: pd.DataFrame | None,
    efpr_df: pd.DataFrame | None,
    retarget_metrics: pd.DataFrame | None,
    generated: list[Path],
) -> Path:
    lines = [
        f"# BEAT2 -> NAO Core Visualization Summary ({backend})",
        "",
        "Core metrics for explaining the pipeline to a new audience:",
        "",
        "1. Source ANOVA effect sizes: shows that emotion labels correspond to measurable motion differences.",
        "2. Robot ANOVA effect sizes and EFPR: shows how much of the emotion-related motion structure survives retargeting.",
        "3. Retargeting quality metrics: MPJPE, JJR, max joint jump, and SCR check geometry, continuity, and physical plausibility.",
        "",
    ]

    if source_anova is not None and robot_anova is not None:
        lines.append("## Section 5 Effect Sizes")
        table = []
        source = source_anova[source_anova["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]
        robot = robot_anova[robot_anova["feature"].isin(FEATURES_CORE)].set_index("feature").loc[FEATURES_CORE]
        for feature in FEATURES_CORE:
            table.append(
                {
                    "feature": feature,
                    "source_eta": source.loc[feature, "eta_squared"],
                    "robot_eta": robot.loc[feature, "eta_squared"],
                }
            )
        lines.append(pd.DataFrame(table).to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    if efpr_df is not None:
        lines.append("## Section 6 EFPR")
        lines.append(
            efpr_df[efpr_df["feature"].isin(FEATURES_CORE)][
                ["feature", "efpr_eta_squared", "efpr_omega_squared"]
            ].to_markdown(index=False, floatfmt=".4f")
        )
        lines.append("")

    if retarget_metrics is not None:
        lines.append("## Section 7 Overall Retargeting Quality")
        all_row = retarget_metrics[retarget_metrics["emotion"] == "ALL"].iloc[0]
        quality = pd.DataFrame(
            [
                {"metric": "MPJPE mean (mm)", "value": all_row["mpjpe_mm_mean"]},
                {"metric": "JJR mean", "value": all_row["joint_jump_rate_mean"]},
                {"metric": "Max joint jump mean (rad)", "value": all_row["max_joint_jump_rad_mean"]},
                {"metric": "SCR mean", "value": all_row["self_collision_rate_mean"]},
            ]
        )
        lines.append(quality.to_markdown(index=False, floatfmt=".6f"))
        lines.append("")

    lines.append("## Generated Figures")
    for path in generated:
        lines.append(f"- `{path.name}`")
    lines.append("")

    output_path = output_dir / f"core_visualization_summary_{backend}.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate professional core BEAT2 pipeline visualizations."
    )
    parser.add_argument("--backend", default="gmr_baseline")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--source_feature_summary", default="motion_data/BEAT2/features/source/beat2_source_feature_summary_by_emotion.csv")
    parser.add_argument("--robot_feature_summary", default=None)
    parser.add_argument("--source_anova", default="motion_data/BEAT2/anova/source/anova_main_table.csv")
    parser.add_argument("--robot_anova", default=None)
    parser.add_argument("--efpr_table", default=None)
    parser.add_argument("--efpr_ci", default=None)
    parser.add_argument("--retarget_metrics", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = args.backend
    output_dir = repo_path(args.output_dir) if args.output_dir else repo_path(f"motion_data/BEAT2/visualizations/{backend}")

    robot_feature_summary = args.robot_feature_summary or f"motion_data/BEAT2/features/{backend}/beat2_nao_feature_summary_by_emotion.csv"
    robot_anova = args.robot_anova or f"motion_data/BEAT2/anova/{backend}/anova_main_table.csv"
    efpr_table = args.efpr_table or f"motion_data/BEAT2/efpr/{backend}/efpr_dimension_table.csv"
    efpr_ci = args.efpr_ci or f"motion_data/BEAT2/efpr/{backend}/efpr_bootstrap_ci.csv"
    retarget_metrics = args.retarget_metrics or f"motion_data/BEAT2/retarget_metrics/{backend}/nao_retarget_metrics_summary_by_emotion.csv"

    source_summary = read_csv_if_exists(repo_path(args.source_feature_summary))
    robot_summary = read_csv_if_exists(repo_path(robot_feature_summary))
    source_anova = read_csv_if_exists(repo_path(args.source_anova))
    robot_anova_df = read_csv_if_exists(repo_path(robot_anova))
    efpr_df = read_csv_if_exists(repo_path(efpr_table))
    efpr_ci_df = read_csv_if_exists(repo_path(efpr_ci))
    retarget_metrics_df = read_csv_if_exists(repo_path(retarget_metrics))

    generated = []
    if source_summary is not None and robot_summary is not None:
        generated.append(plot_emotion_profiles(source_summary, robot_summary, output_dir, backend))
    if source_anova is not None and robot_anova_df is not None:
        generated.append(plot_anova_effects(source_anova, robot_anova_df, output_dir, backend))
    if efpr_df is not None:
        generated.append(plot_efpr(efpr_df, efpr_ci_df, output_dir, backend))
    if retarget_metrics_df is not None:
        generated.append(plot_retarget_quality(retarget_metrics_df, output_dir, backend))
    if source_anova is not None and robot_anova_df is not None:
        generated.append(
            plot_core_dashboard(
                source_anova,
                robot_anova_df,
                efpr_df,
                retarget_metrics_df,
                output_dir,
                backend,
            )
        )

    write_summary_markdown(
        output_dir,
        backend,
        source_anova,
        robot_anova_df,
        efpr_df,
        retarget_metrics_df,
        generated,
    )


if __name__ == "__main__":
    main()
