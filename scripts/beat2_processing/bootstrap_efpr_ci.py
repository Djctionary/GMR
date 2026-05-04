import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEFAULT_DIMENSIONS = ("W", "Ti", "F")
EFFECT_NAMES = ("eta_squared", "omega_squared")


def resolve_repo_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def read_feature_rows(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows[row["clip_id"]] = row
    return rows


def load_paired_features(
    source_path: Path, robot_path: Path, dimensions: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    source_rows = read_feature_rows(source_path)
    robot_rows = read_feature_rows(robot_path)
    clip_ids = sorted(set(source_rows) & set(robot_rows))
    if not clip_ids:
        raise ValueError("No paired clip IDs found between source and robot feature tables")

    source_values = []
    robot_values = []
    emotions = []
    for clip_id in clip_ids:
        source_row = source_rows[clip_id]
        robot_row = robot_rows[clip_id]
        if source_row["emotion"] != robot_row["emotion"]:
            raise ValueError(
                f"Emotion mismatch for {clip_id}: "
                f"{source_row['emotion']} vs {robot_row['emotion']}"
            )
        source_values.append([float(source_row[dimension]) for dimension in dimensions])
        robot_values.append([float(robot_row[dimension]) for dimension in dimensions])
        emotions.append(source_row["emotion"])

    return (
        np.asarray(source_values, dtype=np.float64),
        np.asarray(robot_values, dtype=np.float64),
        np.asarray(emotions),
        clip_ids,
    )


def effect_sizes(values: np.ndarray, emotions: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(values)
    values = values[finite]
    emotions = emotions[finite]
    if values.size == 0:
        return {"eta_squared": math.nan, "omega_squared": math.nan}

    groups = []
    for emotion in sorted(set(emotions.tolist())):
        group_values = values[emotions == emotion]
        if group_values.size:
            groups.append(group_values)
    if len(groups) < 2:
        return {"eta_squared": math.nan, "omega_squared": math.nan}

    grand_mean = float(np.mean(values))
    ss_between = 0.0
    ss_within = 0.0
    for group_values in groups:
        group_mean = float(np.mean(group_values))
        ss_between += group_values.size * (group_mean - grand_mean) ** 2
        ss_within += float(np.sum((group_values - group_mean) ** 2))

    df_between = len(groups) - 1
    df_within = values.size - len(groups)
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total > 0 else math.nan

    if df_within <= 0:
        omega_squared = math.nan
    else:
        ms_within = ss_within / df_within
        denominator = ss_total + ms_within
        omega_squared = (
            (ss_between - df_between * ms_within) / denominator
            if denominator > 0
            else math.nan
        )

    return {"eta_squared": eta_squared, "omega_squared": omega_squared}


def geometric_mean(values: list[float]) -> float:
    if any((not math.isfinite(value)) or value <= 0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compute_sample_efpr(
    source_values: np.ndarray,
    robot_values: np.ndarray,
    emotions: np.ndarray,
    dimensions: tuple[str, ...],
) -> dict[str, float]:
    result = {}
    by_effect = {effect_name: [] for effect_name in EFFECT_NAMES}

    for dim_idx, dimension in enumerate(dimensions):
        source_effects = effect_sizes(source_values[:, dim_idx], emotions)
        robot_effects = effect_sizes(robot_values[:, dim_idx], emotions)
        for effect_name in EFFECT_NAMES:
            source_effect = source_effects[effect_name]
            robot_effect = robot_effects[effect_name]
            efpr = (
                robot_effect / source_effect
                if math.isfinite(source_effect) and source_effect > 0
                else math.nan
            )
            result[f"{dimension}_{effect_name}"] = efpr
            by_effect[effect_name].append(efpr)

    for effect_name, values in by_effect.items():
        result[f"aggregate_{effect_name}"] = geometric_mean(values)

    return result


def stratified_bootstrap_indices(emotions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled = []
    for emotion in sorted(set(emotions.tolist())):
        indices = np.flatnonzero(emotions == emotion)
        sampled.append(rng.choice(indices, size=indices.size, replace=True))
    return np.concatenate(sampled)


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired stratified bootstrap 95% CIs for EFPR."
    )
    parser.add_argument(
        "--source_features",
        default="motion_data/BEAT2/features/source/beat2_source_features.csv",
        help="Source-side per-clip feature CSV.",
    )
    parser.add_argument(
        "--robot_features",
        default="motion_data/BEAT2/features/gmr_baseline/beat2_nao_features.csv",
        help="Robot-side per-clip feature CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="motion_data/BEAT2/efpr/gmr_baseline",
        help="Output directory for bootstrap EFPR CI files.",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help="Feature dimensions included in aggregate EFPR. Default: W Ti F.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    source_features = resolve_repo_path(repo_root, args.source_features).resolve()
    robot_features = resolve_repo_path(repo_root, args.robot_features).resolve()
    output_dir = resolve_repo_path(repo_root, args.output_dir).resolve()
    dimensions = tuple(args.dimensions)

    source_values, robot_values, emotions, clip_ids = load_paired_features(
        source_features, robot_features, dimensions
    )
    rng = np.random.default_rng(args.seed)

    point = compute_sample_efpr(source_values, robot_values, emotions, dimensions)
    bootstrap_rows = []
    metrics = list(point.keys())

    for iteration in tqdm(range(args.n_bootstrap), desc="Bootstrap EFPR CI"):
        indices = stratified_bootstrap_indices(emotions, rng)
        sample_result = compute_sample_efpr(
            source_values[indices],
            robot_values[indices],
            emotions[indices],
            dimensions,
        )
        row = {"iteration": str(iteration)}
        row.update({metric: fmt(sample_result[metric]) for metric in metrics})
        bootstrap_rows.append(row)

    ci_rows = []
    summary = {
        "n_pairs": len(clip_ids),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "dimensions": list(dimensions),
        "method": "paired stratified bootstrap by emotion; percentile 95% CI",
    }
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in bootstrap_rows], dtype=np.float64)
        low, high = percentile_ci(values)
        ci_rows.append(
            {
                "metric": metric,
                "point": fmt(point[metric]),
                "ci_low_2_5": fmt(low),
                "ci_high_97_5": fmt(high),
                "bootstrap_mean": fmt(float(np.nanmean(values))),
                "bootstrap_std": fmt(float(np.nanstd(values, ddof=1))),
                "finite_bootstrap_samples": str(int(np.count_nonzero(np.isfinite(values)))),
            }
        )
        summary[metric] = {
            "point": point[metric],
            "ci_low_2_5": low,
            "ci_high_97_5": high,
            "bootstrap_mean": float(np.nanmean(values)),
            "bootstrap_std": float(np.nanstd(values, ddof=1)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    ci_path = output_dir / "efpr_bootstrap_ci.csv"
    sample_path = output_dir / "efpr_bootstrap_samples.csv"
    summary_path = output_dir / "efpr_bootstrap_summary.json"
    write_csv(
        ci_path,
        ci_rows,
        [
            "metric",
            "point",
            "ci_low_2_5",
            "ci_high_97_5",
            "bootstrap_mean",
            "bootstrap_std",
            "finite_bootstrap_samples",
        ],
    )
    write_csv(sample_path, bootstrap_rows, ["iteration", *metrics])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[DONE] CI table: {ci_path}")
    print(f"[DONE] bootstrap samples: {sample_path}")
    print(f"[DONE] summary: {summary_path}")


if __name__ == "__main__":
    main()
