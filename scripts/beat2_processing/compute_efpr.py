import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_DIMENSIONS = ("W", "Ti", "F")
EFFECT_COLUMNS = ("eta_squared", "omega_squared")


def resolve_repo_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def read_anova_main_table(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            feature = row["feature"]
            rows[feature] = {
                effect_name: float(row[effect_name]) for effect_name in EFFECT_COLUMNS
            }
    return rows


def geometric_mean(values: list[float]) -> float:
    if not values:
        return math.nan
    if any(value < 0 for value in values):
        raise ValueError(f"Geometric mean is undefined for negative values: {values}")
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.10g}"


def compute_efpr(
    human_rows: dict[str, dict[str, float]],
    robot_rows: dict[str, dict[str, float]],
    dimensions: tuple[str, ...],
) -> tuple[list[dict], dict]:
    dimension_rows = []
    summary = {"dimensions": list(dimensions)}
    per_effect_values = {effect_name: [] for effect_name in EFFECT_COLUMNS}

    for dimension in dimensions:
        if dimension not in human_rows:
            raise ValueError(f"Missing dimension in human ANOVA table: {dimension}")
        if dimension not in robot_rows:
            raise ValueError(f"Missing dimension in robot ANOVA table: {dimension}")

        row = {"feature": dimension}
        for effect_name in EFFECT_COLUMNS:
            human_effect = human_rows[dimension][effect_name]
            robot_effect = robot_rows[dimension][effect_name]
            if human_effect <= 0:
                efpr = math.nan
            else:
                efpr = robot_effect / human_effect
                per_effect_values[effect_name].append(efpr)

            row[f"human_{effect_name}"] = fmt(human_effect)
            row[f"robot_{effect_name}"] = fmt(robot_effect)
            row[f"efpr_{effect_name}"] = fmt(efpr)

        dimension_rows.append(row)

    for effect_name, values in per_effect_values.items():
        summary[f"aggregate_efpr_{effect_name}"] = geometric_mean(values)
        summary[f"dimension_efpr_{effect_name}"] = {
            row["feature"]: float(row[f"efpr_{effect_name}"]) for row in dimension_rows
        }

    return dimension_rows, summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Section 6 EFPR from source and robot ANOVA main tables."
    )
    parser.add_argument(
        "--human_anova",
        default="motion_data/BEAT2/anova/source/anova_main_table.csv",
        help="Source/human ANOVA main table.",
    )
    parser.add_argument(
        "--robot_anova",
        default="motion_data/BEAT2/anova/gmr_baseline/anova_main_table.csv",
        help="Robot/NAO ANOVA main table.",
    )
    parser.add_argument(
        "--output_dir",
        default="motion_data/BEAT2/efpr/gmr_baseline",
        help="Output directory for EFPR results.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help="Feature dimensions used for aggregate EFPR. Default: W Ti F.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    human_anova = resolve_repo_path(repo_root, args.human_anova).resolve()
    robot_anova = resolve_repo_path(repo_root, args.robot_anova).resolve()
    output_dir = resolve_repo_path(repo_root, args.output_dir).resolve()

    dimensions = tuple(args.dimensions)
    human_rows = read_anova_main_table(human_anova)
    robot_rows = read_anova_main_table(robot_anova)
    dimension_rows, summary = compute_efpr(human_rows, robot_rows, dimensions)

    fieldnames = ["feature"]
    for effect_name in EFFECT_COLUMNS:
        fieldnames.extend(
            [
                f"human_{effect_name}",
                f"robot_{effect_name}",
                f"efpr_{effect_name}",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dimension_path = output_dir / "efpr_dimension_table.csv"
    summary_path = output_dir / "efpr_summary.json"

    write_csv(dimension_path, dimension_rows, fieldnames)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[DONE] dimension table: {dimension_path}")
    print(f"[DONE] summary: {summary_path}")
    print(
        "[DONE] aggregate EFPR: "
        f"eta_squared={summary['aggregate_efpr_eta_squared']:.10g}, "
        f"omega_squared={summary['aggregate_efpr_omega_squared']:.10g}"
    )


if __name__ == "__main__":
    main()
