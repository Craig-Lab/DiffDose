#!/usr/bin/env python3
"""Plot cohort endpoint distributions before and after individualized optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/Mosun_optimization.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/SupplementaryFigure8_computational.svg"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].violinplot([data.reference_il6_peak, data.optimized_il6_peak], showmedians=True)
    axes[0].set_xticks([1, 2], ["Hosseini", "Individualized"])
    axes[0].set_ylabel("Global IL-6 peak")
    for index, column in enumerate(("reference_tumor_ratio", "optimized_tumor_ratio")):
        values = 100.0 * (data[column].sort_values().to_numpy() - 1.0)
        axes[1].plot(np.linspace(0, 1, len(values)), values, label=("Hosseini", "Individualized")[index])
    axes[1].axhline(-50, color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set(xlabel="Virtual-patient rank", ylabel="Day-84 tumor change (%)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
