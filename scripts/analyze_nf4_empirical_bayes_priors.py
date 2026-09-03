#!/usr/bin/env python3
"""Select a shared fit4 prior width using interpolation-fit log evidence.

This is a post-processing diagnostic. It reads completed fit4 width runs and
does not rerun or modify any analysis, notebook, or existing result file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIT_KEYS = ["operator", "flow_time"]
SUMMARY_NAME = "interpolation_fit_quality_summary.csv"


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("widths must be positive")
    return value


def width_tag(width: float) -> str:
    return f"{width:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empirical-Bayes comparison of completed NF4 fit4 prior-width runs."
    )
    parser.add_argument(
        "--output-base", type=Path, required=True,
        help="Directory containing fit4_width*/order_* results.",
    )
    parser.add_argument("--orders", type=int, nargs="+", default=(3, 4))
    parser.add_argument(
        "--widths", type=positive_float, nargs="+", default=(1, 3, 10, 30, 1000)
    )
    parser.add_argument(
        "--result-dir", type=Path,
        help="New output directory (default: OUTPUT_BASE/empirical_bayes_prior_selection).",
    )
    return parser.parse_args()


def load_summary(output_base: Path, order: int, width: float) -> pd.DataFrame:
    path = (
        output_base / f"fit4_width{width_tag(width)}" / f"order_{order}" / SUMMARY_NAME
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing completed width run: {path}")
    frame = pd.read_csv(path)
    required = {*FIT_KEYS, "logGBF", "chi2_per_dof", "p_value", "xerrors"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    xerrors = frame["xerrors"].map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() == "true"
    )
    if not xerrors.all():
        raise ValueError(f"proper-prior evidence comparison requires xerrors=True: {path}")

    # The same interpolation fit occurs in multiple overlapping continuum
    # windows. Count it once, and verify that its saved diagnostics agree.
    grouped = frame.groupby(FIT_KEYS, dropna=False)
    if (grouped["logGBF"].nunique(dropna=False) > 1).any():
        raise ValueError(f"inconsistent duplicated fit diagnostics in {path}")
    unique = frame.drop_duplicates(FIT_KEYS).copy()
    if not np.isfinite(unique["logGBF"]).all():
        raise ValueError(f"non-finite logGBF values in {path}")
    unique["order"] = order
    unique["width"] = float(width)
    return unique


def common_fit_frames(frames: dict[float, pd.DataFrame]) -> dict[float, pd.DataFrame]:
    key_sets = {
        width: set(map(tuple, frame[FIT_KEYS].itertuples(index=False, name=None)))
        for width, frame in frames.items()
    }
    common = set.intersection(*key_sets.values())
    if not common:
        raise ValueError("the width runs have no common interpolation fits")
    if any(keys != common for keys in key_sets.values()):
        counts = {width: len(keys) for width, keys in key_sets.items()}
        raise ValueError(f"width runs do not contain identical fit keys: {counts}")
    return {
        width: frame.set_index(FIT_KEYS).sort_index().loc[sorted(common)].reset_index()
        for width, frame in frames.items()
    }


def local_log_width_optimum(grid: pd.DataFrame) -> float | None:
    ordered = grid.sort_values("width").reset_index(drop=True)
    best = int(ordered["composite_logGBF"].idxmax())
    if best == 0 or best == len(ordered) - 1:
        return None
    local = ordered.iloc[best - 1 : best + 2]
    coefficients = np.polyfit(
        np.log(local["width"].to_numpy()),
        local["composite_logGBF"].to_numpy(),
        deg=2,
    )
    if coefficients[0] >= 0:
        return None
    optimum = float(np.exp(-coefficients[1] / (2 * coefficients[0])))
    lower, upper = local["width"].iloc[[0, -1]]
    return optimum if lower <= optimum <= upper else None


def analyze_order(order: int, widths: list[float], output_base: Path):
    frames = common_fit_frames(
        {width: load_summary(output_base, order, width) for width in widths}
    )
    grid_rows = []
    evidence_columns = {}
    operator_rows = []
    for width, frame in frames.items():
        evidence_columns[width] = frame.set_index(FIT_KEYS)["logGBF"]
        grid_rows.append({
            "order": order,
            "width": width,
            "n_unique_fits": len(frame),
            "composite_logGBF": frame["logGBF"].sum(),
            "mean_logGBF": frame["logGBF"].mean(),
            "median_logGBF": frame["logGBF"].median(),
            "median_chi2_per_dof": frame["chi2_per_dof"].median(),
            "fraction_p_above_0p05": (frame["p_value"] > 0.05).mean(),
        })
        for operator, subset in frame.groupby("operator"):
            operator_rows.append({
                "order": order, "operator": operator, "width": width,
                "n_unique_fits": len(subset),
                "composite_logGBF": subset["logGBF"].sum(),
            })

    grid = pd.DataFrame(grid_rows).sort_values("width").reset_index(drop=True)
    best_width = float(grid.loc[grid["composite_logGBF"].idxmax(), "width"])
    grid["delta_composite_logGBF"] = grid["composite_logGBF"] - grid["composite_logGBF"].max()

    evidence = pd.concat(evidence_columns, axis=1)
    evidence.columns.name = "width"
    winners = evidence.idxmax(axis=1).rename("preferred_width").reset_index()
    winner_counts = (
        winners.groupby("preferred_width").size().rename("n_fits").reset_index()
    )
    winner_counts["order"] = order
    winner_counts["fraction"] = winner_counts["n_fits"] / len(winners)
    return grid, pd.DataFrame(operator_rows), winner_counts, best_width, local_log_width_optimum(grid)


def main() -> int:
    args = parse_args()
    output_base = args.output_base.expanduser().resolve()
    result_dir = (
        args.result_dir.expanduser().resolve()
        if args.result_dir else output_base / "empirical_bayes_prior_selection"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    widths = sorted(set(map(float, args.widths)))

    grids, operators, winners, recommendations = [], [], [], []
    for order in args.orders:
        grid, operator, winner, best, continuous = analyze_order(order, widths, output_base)
        grids.append(grid)
        operators.append(operator)
        winners.append(winner)
        recommendations.append({
            "order": order,
            "recommended_tested_width": best,
            "local_quadratic_width_diagnostic": continuous,
        })

    grid = pd.concat(grids, ignore_index=True)
    operator = pd.concat(operators, ignore_index=True)
    winner = pd.concat(winners, ignore_index=True)
    recommendation = pd.DataFrame(recommendations)
    grid.to_csv(result_dir / "empirical_bayes_evidence_grid.csv", index=False)
    operator.to_csv(result_dir / "empirical_bayes_evidence_by_operator.csv", index=False)
    winner.to_csv(result_dir / "empirical_bayes_per_fit_winners.csv", index=False)
    recommendation.to_csv(result_dir / "empirical_bayes_recommendation.csv", index=False)

    report = [
        "NF4 fit4 empirical-Bayes prior-width diagnostic",
        "================================================",
        "",
        "Criterion: maximize interpolation-fit logGBF over a shared zero-centered",
        "coefficient-prior width. Duplicate fits from overlapping continuum windows",
        "are counted once. The summed logGBF is a composite diagnostic because flow",
        "times and operators are correlated; per-fit and per-operator stability tables",
        "are therefore reported alongside it.",
        "",
        "The no-prior fit is excluded: an improper/absent parameter prior does not",
        "define evidence on the same normalization and cannot be ranked by logGBF",
        "against the proper Gaussian priors.",
        "",
        "Recommendation",
        recommendation.to_string(index=False),
        "",
        "Evidence grid",
        grid.to_string(index=False),
        "",
        "Per-fit winning-width counts",
        winner.to_string(index=False),
        "",
        "Interpretation: empirical Bayes uses the data twice (to tune and then fit).",
        "Treat the selected width as a sensitivity-analysis recommendation, and retain",
        "neighboring widths when quoting systematic dependence.",
    ]
    (result_dir / "empirical_bayes_report.txt").write_text("\n".join(report) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for order, subset in grid.groupby("order"):
        ax.plot(
            subset["width"], subset["delta_composite_logGBF"], "o-", label=f"order {order}"
        )
    ax.set_xscale("log")
    ax.set_xlabel("shared prior width")
    ax.set_ylabel(r"$\Delta$ composite logGBF from best tested width")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(result_dir / "empirical_bayes_evidence.png", dpi=250)
    fig.savefig(result_dir / "empirical_bayes_evidence.pdf")
    plt.close(fig)

    print(recommendation.to_string(index=False))
    print(f"wrote empirical-Bayes diagnostics to {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
