import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd


FEATURES = ("W", "Ti", "S", "F")
MAIN_COLUMNS = (
    "feature",
    "F_oneway",
    "p_oneway",
    "F_welch",
    "p_welch",
    "H_kruskal",
    "p_kruskal",
    "levene_p",
    "eta_squared",
    "omega_squared",
    "n_significant_pairs_tukey",
)


def write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value) -> float:
    if value is None:
        return math.nan
    return float(value)


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.10g}"


def series_first(series: pd.Series, names: tuple[str, ...]) -> float:
    for name in names:
        if name in series:
            return to_float(series[name])
    raise KeyError(f"None of these columns were found: {names}. Available: {list(series.index)}")


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"clip_id", "emotion", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    return df


def grouped_values(df: pd.DataFrame, feature: str) -> list[np.ndarray]:
    groups = []
    for _, group in df.groupby("emotion", sort=True):
        values = group[feature].dropna().to_numpy(dtype=float)
        if values.size:
            groups.append(values)
    if len(groups) < 2:
        raise ValueError(f"{feature} has fewer than two non-empty emotion groups")
    return groups


def effect_sizes(df: pd.DataFrame, feature: str) -> tuple[float, float]:
    model = smf.ols(f"{feature} ~ C(emotion)", data=df[["emotion", feature]].dropna()).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    ss_between = float(anova_table.loc["C(emotion)", "sum_sq"])
    ss_within = float(anova_table.loc["Residual", "sum_sq"])
    df_between = float(anova_table.loc["C(emotion)", "df"])
    df_within = float(anova_table.loc["Residual", "df"])
    ss_total = ss_between + ss_within
    ms_within = ss_within / df_within

    eta_squared = ss_between / ss_total if ss_total > 0 else math.nan
    omega_denominator = ss_total + ms_within
    omega_squared = (
        (ss_between - df_between * ms_within) / omega_denominator
        if omega_denominator > 0
        else math.nan
    )
    return eta_squared, omega_squared


def tukey_results(df: pd.DataFrame, feature: str) -> tuple[int, list[dict]]:
    clean = df[["emotion", feature]].dropna()
    result = pairwise_tukeyhsd(clean[feature], clean["emotion"], alpha=0.05)
    rows = []
    significant = 0
    for row in result.summary().data[1:]:
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        if bool(reject):
            significant += 1
        rows.append(
            {
                "feature": feature,
                "group1": group1,
                "group2": group2,
                "meandiff": fmt(float(meandiff)),
                "p_adj": fmt(float(p_adj)),
                "lower": fmt(float(lower)),
                "upper": fmt(float(upper)),
                "reject": str(bool(reject)),
            }
        )
    return significant, rows


def shapiro_results(df: pd.DataFrame, feature: str) -> list[dict]:
    rows = []
    for emotion, group in df.groupby("emotion", sort=True):
        values = group[feature].dropna().to_numpy(dtype=float)
        if values.size < 3:
            statistic = math.nan
            p_value = math.nan
        else:
            statistic, p_value = stats.shapiro(values)
        rows.append(
            {
                "feature": feature,
                "emotion": emotion,
                "n": str(values.size),
                "W_shapiro": fmt(float(statistic)),
                "p_shapiro": fmt(float(p_value)),
            }
        )
    return rows


def run_anova(df: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict], dict]:
    main_rows = []
    shapiro_rows = []
    tukey_rows = []
    diagnostics = {}

    for feature in FEATURES:
        clean = df[["emotion", feature]].dropna()
        groups = grouped_values(clean, feature)

        f_oneway, p_oneway = stats.f_oneway(*groups)
        welch = pg.welch_anova(data=clean, dv=feature, between="emotion").iloc[0]
        h_kruskal, p_kruskal = stats.kruskal(*groups)
        _, levene_p = stats.levene(*groups)
        eta_squared, omega_squared = effect_sizes(clean, feature)
        n_significant_pairs, feature_tukey_rows = tukey_results(clean, feature)

        main_rows.append(
            {
                "feature": feature,
                "F_oneway": fmt(float(f_oneway)),
                "p_oneway": fmt(float(p_oneway)),
                "F_welch": fmt(series_first(welch, ("F",))),
                "p_welch": fmt(series_first(welch, ("p-unc", "p_unc", "pval", "p-val"))),
                "H_kruskal": fmt(float(h_kruskal)),
                "p_kruskal": fmt(float(p_kruskal)),
                "levene_p": fmt(float(levene_p)),
                "eta_squared": fmt(eta_squared),
                "omega_squared": fmt(omega_squared),
                "n_significant_pairs_tukey": str(n_significant_pairs),
            }
        )
        shapiro_rows.extend(shapiro_results(clean, feature))
        tukey_rows.extend(feature_tukey_rows)
        diagnostics[feature] = {
            "n_total": int(clean.shape[0]),
            "group_sizes": {
                emotion: int(group[feature].dropna().shape[0])
                for emotion, group in clean.groupby("emotion", sort=True)
            },
        }

    return main_rows, shapiro_rows, tukey_rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Section 3 ANOVA tests on BEAT2 source Laban features."
    )
    parser.add_argument(
        "--features_csv",
        default="motion_data/BEAT2/features/source/beat2_source_features.csv",
        help="Input feature CSV, relative to repository root unless absolute.",
    )
    parser.add_argument(
        "--output_dir",
        default="motion_data/BEAT2/anova/source",
        help="Output directory, relative to repository root unless absolute.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    features_csv = Path(args.features_csv)
    if not features_csv.is_absolute():
        features_csv = repo_root / features_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    df = load_feature_table(features_csv)
    main_rows, shapiro_rows, tukey_rows, diagnostics = run_anova(df)

    write_csv(output_dir / "anova_main_table.csv", main_rows, MAIN_COLUMNS)
    write_csv(
        output_dir / "anova_shapiro_by_group.csv",
        shapiro_rows,
        ["feature", "emotion", "n", "W_shapiro", "p_shapiro"],
    )
    write_csv(
        output_dir / "anova_tukey_hsd.csv",
        tukey_rows,
        ["feature", "group1", "group2", "meandiff", "p_adj", "lower", "upper", "reject"],
    )
    (output_dir / "anova_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"[DONE] main table: {output_dir / 'anova_main_table.csv'}")
    print(f"[DONE] shapiro: {output_dir / 'anova_shapiro_by_group.csv'}")
    print(f"[DONE] tukey: {output_dir / 'anova_tukey_hsd.csv'}")
    print(f"[DONE] diagnostics: {output_dir / 'anova_diagnostics.json'}")


if __name__ == "__main__":
    main()
