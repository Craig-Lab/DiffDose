#!/usr/bin/env python3
"""Compare generated VPop trajectories with digitized Hosseini Figure 5 targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing campaign output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, default=Path("output/vpop/candidate_trajectory_summary.csv"))
    parser.add_argument("--waterfall", type=Path, default=Path("output/vpop/candidate_waterfall.csv"))
    parser.add_argument("--digitized-trajectories", type=Path, default=Path("data/HosseiniFigure5IL6Tcell.csv"))
    parser.add_argument("--digitized-waterfall", type=Path, default=Path("data/HosseiniFigure5Tumor.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/SupplementaryFigure7_computational.svg"))
    args = parser.parse_args()
    for path in (args.trajectories, args.waterfall, args.digitized_trajectories, args.digitized_waterfall):
        require(path)
    trajectories = pd.read_csv(args.trajectories)
    waterfall = pd.read_csv(args.waterfall)
    targets = pd.read_csv(args.digitized_trajectories)
    tumor_targets = pd.read_csv(args.digitized_waterfall)
    regimen = trajectories["regimen"].drop_duplicates().iloc[0]
    subset = trajectories.query("regimen == @regimen and status == 'success'")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, column, biomarker in zip(axes[:2], ("il6combo", "tafraction_pb_pct"), ("IL6", "Tcell")):
        pivot = subset.pivot(index="candidate_id", columns="time_day", values=column)
        axis.fill_between(pivot.columns, np.nanpercentile(pivot, 5, axis=0), np.nanpercentile(pivot, 95, axis=0), color="0.85")
        axis.plot(pivot.columns, np.nanmedian(pivot, axis=0), color="#639b7b", label="Generated VPop")
        target = targets.query("biomarker == @biomarker and curve == 'median_green'")
        for _, group in target.groupby("regimen"):
            axis.plot(group.time_day, group.value, "--", color="#d97706", alpha=0.5)
        axis.set(xlabel="Time (days)", ylabel=biomarker)
    simulated = waterfall.query("regimen == @regimen and status == 'success'")["tumor_size_change_pct"].sort_values()
    axes[2].plot(np.linspace(0, 1, len(simulated)), simulated, color="#639b7b")
    for _, group in tumor_targets.groupby("regimen"):
        axes[2].plot(group.patient_rank_fraction, group.tumor_change_percent_day84, "--", color="#d97706", alpha=0.5)
    axes[2].set(xlabel="Patient rank", ylabel="Day-84 tumor change (%)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
