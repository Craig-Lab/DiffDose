#!/usr/bin/env python3
"""Plot TGI convergence by wall time and optimizer iteration."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/TGI_convergence.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/SupplementaryFigure3_computational.svg"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for method, subset in data.groupby("method", sort=False):
        values = subset["best_objective"] if method in {"Random Walk", "Nelder-Mead"} else subset["objective"]
        axes[0].plot(subset["elapsed_seconds"], values, label=method)
        axes[1].plot(subset["iteration"], values)
    axes[0].set_xlabel("Wall-clock time (s)")
    axes[1].set_xlabel("Iteration / evaluation")
    axes[0].set_ylabel("TGI loss")
    for axis in axes:
        axis.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
