#!/usr/bin/env python3
"""Regenerate the computational panels and source tables underlying Figure 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr


DOSE_COLUMNS = ["C1D1_mg", "C1D8_mg", "C1D15_mg", "C2D1_mg", "C3D1_mg", "C4D1_mg"]
DOSE_LABELS = ["C1D1", "C1D8", "C1D15", "C2D1", "C3D1", "C4D1"]
HOSSEINI_DOSES = np.array([1.6, 10.0, 10.0, 20.0, 20.0, 20.0])
RP2D_DOSES = np.array([1.0, 2.0, 60.0, 60.0, 30.0, 30.0])

ENDPOINTS = [
    ("peak_il6", "Peak IL-6", "reference_il6_peak", "optimized_il6_peak"),
    ("il6_auc", "IL-6 AUC", "reference_il6_auc", "optimized_il6_auc"),
    ("day84_tumor", "Day-84 tumor", "reference_tumor_ratio", "optimized_tumor_ratio"),
    ("tumor_auc", "Tumor AUC", "reference_tumor_auc", "optimized_tumor_auc"),
    ("total_dose", "Total dose", "reference_total_dose_mg", "optimized_total_dose_mg"),
    ("drug_exposure", "Drug exposure", "reference_drug_exposure", "optimized_drug_exposure"),
]

CORRELATION_PARAMETERS = [
    ("BT_ratio_tumor_init", "Initial B-cell : T-cell ratio"),
    ("nkill", "B-cell killing steepness"),
    ("Vtumor", "Tumor volume scale"),
    ("tumor_burden_factor", "Initial tumor burden"),
    ("kBkill", "B-cell killing rate"),
    ("kBtumorprolif", "Tumor B-cell proliferation"),
]


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in optimization output: {', '.join(missing)}")


def bounded_improvement(reference: pd.Series, optimized: pd.Series) -> pd.Series:
    denominator = reference + optimized
    return 100.0 * (reference - optimized) / denominator.where(denominator != 0)


def endpoint_patterns(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient = pd.DataFrame({"vpop_id": results["vpop_id"]})
    for key, _, reference, optimized in ENDPOINTS:
        patient[key] = results[optimized].to_numpy(float) < results[reference].to_numpy(float)
    endpoint_keys = [item[0] for item in ENDPOINTS]
    patient["pattern"] = patient[endpoint_keys].astype(int).astype(str).agg("".join, axis=1)
    counts = (
        patient.groupby("pattern", sort=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["n", "pattern"], ascending=[False, False], kind="mergesort")
        .reset_index(drop=True)
    )
    counts["wins"] = counts["pattern"].str.count("1")
    counts["pct"] = 100.0 * counts["n"] / len(patient)
    counts["rank"] = np.arange(1, len(counts) + 1)
    return patient, counts


def local_density(x: pd.Series, y: pd.Series) -> np.ndarray:
    coordinates = np.vstack([x.to_numpy(float), y.to_numpy(float)])
    finite = np.isfinite(coordinates).all(axis=0)
    density = np.full(coordinates.shape[1], np.nan)
    if finite.sum() < 3:
        density[finite] = 1.0
        return density
    try:
        density[finite] = gaussian_kde(coordinates[:, finite])(coordinates[:, finite])
    except np.linalg.LinAlgError:
        density[finite] = 1.0
    return density


def plot_upset(bar_axis, matrix_axis, counts: pd.DataFrame) -> None:
    top = counts.head(8).reset_index(drop=True)
    positions = np.arange(len(top))
    bar_axis.bar(positions, top["n"], color="#639b7b", edgecolor="#356b55")
    for x, row in top.iterrows():
        bar_axis.text(
            x,
            row.n + max(top.n) * 0.025,
            f"{row.n}\n{row.pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    bar_axis.set_ylabel("Virtual patients")
    bar_axis.set_xticks([])
    bar_axis.spines[["top", "right"]].set_visible(False)

    row_positions = np.arange(len(ENDPOINTS))[::-1]
    for x, pattern in enumerate(top["pattern"]):
        selected = [row_positions[index] for index, value in enumerate(pattern) if value == "1"]
        if len(selected) > 1:
            matrix_axis.plot([x, x], [min(selected), max(selected)], color="0.2", lw=1.2, zorder=1)
        for index, value in enumerate(pattern):
            color = "0.1" if value == "1" else "0.85"
            matrix_axis.scatter(x, row_positions[index], s=28, color=color, zorder=2)
    matrix_axis.set_yticks(row_positions, [item[1] for item in ENDPOINTS])
    matrix_axis.set_xticks([])
    matrix_axis.set_xlim(-0.6, max(len(top) - 0.4, 0.6))
    matrix_axis.set_ylim(-0.6, len(ENDPOINTS) - 0.4)
    matrix_axis.spines[["top", "right", "bottom"]].set_visible(False)
    matrix_axis.set_xlabel("Top endpoint-win combinations")


def write_source_tables(
    source_dir: Path,
    patient_patterns: pd.DataFrame,
    pattern_counts: pd.DataFrame,
    improvements: pd.DataFrame,
    results: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    patient_patterns.to_csv(source_dir / "endpoint_win_patterns_by_patient.csv", index=False)
    pattern_counts.to_csv(source_dir / "endpoint_win_patterns_complete.csv", index=False)
    pattern_counts.head(8).to_csv(source_dir / "endpoint_win_patterns_top8.csv", index=False)
    improvements.to_csv(source_dir / "bounded_primary_endpoint_improvements.csv", index=False)

    dose_summary = pd.DataFrame(
        {
            "administration": DOSE_LABELS,
            "median_mg": results[DOSE_COLUMNS].median().to_numpy(float),
            "q1_mg": results[DOSE_COLUMNS].quantile(0.25).to_numpy(float),
            "q3_mg": results[DOSE_COLUMNS].quantile(0.75).to_numpy(float),
            "hosseini_mg": HOSSEINI_DOSES,
            "rp2d_mg": RP2D_DOSES,
        }
    )
    dose_summary.to_csv(source_dir / "optimized_dose_summary.csv", index=False)
    correlations.to_csv(source_dir / "final_loss_parameter_associations.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("output/Mosun_optimization.csv"))
    parser.add_argument("--vpop", type=Path, default=Path("data/MosunVPop250.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/Figure4_computational.svg"))
    parser.add_argument("--source-dir", type=Path, default=Path("output/figures/Figure4_source"))
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    required = ["vpop_id", "final_loss", *DOSE_COLUMNS]
    required.extend(column for _, _, reference, optimized in ENDPOINTS for column in (reference, optimized))
    require_columns(results, required)
    vpop = pd.read_csv(args.vpop)
    merged = results.merge(vpop, on="vpop_id", how="left", validate="one_to_one")

    patient_patterns, pattern_counts = endpoint_patterns(results)
    tumor = bounded_improvement(results["reference_tumor_ratio"], results["optimized_tumor_ratio"])
    il6 = bounded_improvement(results["reference_il6_peak"], results["optimized_il6_peak"])
    improvements = pd.DataFrame(
        {
            "vpop_id": results["vpop_id"],
            "day84_residual_tumor_improvement_pct": tumor,
            "global_il6_peak_improvement_pct": il6,
        }
    )

    correlation_rows = []
    for parameter, label in CORRELATION_PARAMETERS:
        if parameter not in merged:
            continue
        rho, p_value = spearmanr(merged[parameter], merged["final_loss"], nan_policy="omit")
        correlation_rows.append(
            {"parameter": parameter, "label": label, "spearman_rho": rho, "p_value": p_value}
        )
    correlations = pd.DataFrame(correlation_rows)

    figure = plt.figure(figsize=(12, 9))
    outer = figure.add_gridspec(2, 2, hspace=0.34, wspace=0.3)
    upset = outer[0, 0].subgridspec(2, 1, height_ratios=(1.8, 1.2), hspace=0.04)
    axis_a_bar = figure.add_subplot(upset[0])
    axis_a_matrix = figure.add_subplot(upset[1], sharex=axis_a_bar)
    axis_b = figure.add_subplot(outer[0, 1])
    axis_c = figure.add_subplot(outer[1, 0])
    axis_d = figure.add_subplot(outer[1, 1])

    plot_upset(axis_a_bar, axis_a_matrix, pattern_counts)
    axis_a_bar.set_title("Individualized optimization vs Hosseini et al.", loc="left", fontweight="bold")
    axis_a_bar.text(-0.12, 1.06, "a", transform=axis_a_bar.transAxes, fontsize=17, fontweight="bold")

    density = local_density(tumor, il6)
    axis_b.scatter(tumor, il6, c=density, cmap="viridis_r", s=22, alpha=0.8, edgecolors="none")
    axis_b.axhline(0, color="0.5", lw=0.8)
    axis_b.axvline(0, color="0.5", lw=0.8)
    axis_b.set_xscale("symlog", linthresh=0.1)
    axis_b.set_yscale("symlog", linthresh=0.1)
    axis_b.set_xlim(-100, 100)
    axis_b.set_ylim(-100, 100)
    axis_b.set_xlabel("Day-84 residual tumor improvement (%)")
    axis_b.set_ylabel("Global IL-6 peak improvement (%)")
    axis_b.set_title("Individualized optimization vs Hosseini et al.", loc="left", fontweight="bold")
    axis_b.text(-0.12, 1.02, "b", transform=axis_b.transAxes, fontsize=17, fontweight="bold")

    dose_values = [results[column].dropna().to_numpy(float) for column in DOSE_COLUMNS]
    violins = axis_c.violinplot(dose_values, showextrema=False)
    for body in violins["bodies"]:
        body.set_facecolor("0.82")
        body.set_edgecolor("0.55")
        body.set_alpha(0.65)
    positions = np.arange(1, 7)
    medians = results[DOSE_COLUMNS].median().to_numpy(float)
    q1 = results[DOSE_COLUMNS].quantile(0.25).to_numpy(float)
    q3 = results[DOSE_COLUMNS].quantile(0.75).to_numpy(float)
    axis_c.errorbar(
        positions,
        medians,
        yerr=[medians - q1, q3 - medians],
        fmt="o",
        color="0.1",
        capsize=2,
        label="Median / IQR",
    )
    axis_c.plot(positions, HOSSEINI_DOSES, "o-", color="#0072b2", label="Hosseini et al.")
    axis_c.plot(positions, RP2D_DOSES, "s-", color="#009e73", label="RP2D")
    axis_c.set_xticks(positions, DOSE_LABELS, rotation=25)
    axis_c.set_ylabel("Dose (mg)")
    axis_c.set_title("Dose schedule selected by individualized optimization", loc="left", fontweight="bold")
    axis_c.legend(frameon=False, fontsize=8)
    axis_c.text(-0.12, 1.02, "c", transform=axis_c.transAxes, fontsize=17, fontweight="bold")

    if not correlations.empty:
        plotted = correlations.iloc[::-1]
        colors = ["#cf5046" if value > 0 else "#4f91bd" for value in plotted["spearman_rho"]]
        axis_d.barh(plotted["label"], plotted["spearman_rho"], color=colors, edgecolor="0.25")
    axis_d.axvline(0, color="0.3", lw=0.8)
    axis_d.set_xlabel(r"Spearman $\rho$ with final loss")
    axis_d.set_title("Final-loss associations", loc="left", fontweight="bold")
    axis_d.text(-0.12, 1.02, "d", transform=axis_d.transAxes, fontsize=17, fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    write_source_tables(args.source_dir, patient_patterns, pattern_counts, improvements, results, correlations)


if __name__ == "__main__":
    main()
