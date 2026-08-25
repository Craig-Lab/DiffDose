#!/usr/bin/env python3
"""Regenerate the computational convergence panels underlying Main Figure 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "OptiDose": "#1f4e79",
    "Adjoint Sensitivity": "#9ecae1",
    "Reverse AD": "#4e79a7",
    "Forward AD": "#b15928",
    "Forward Sensitivity": "#f28e2b",
    "Finite Difference": "#ffbe7d",
    "Random Walk": "#8c8c8c",
    "Nelder-Mead": "#c7c7c7",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/IDR_convergence.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/Figure2_computational.svg"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    methods = list(data["method"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for index, method in enumerate(methods):
        subset = data[data["method"] == method]
        derivative_free = method in {"Random Walk", "Nelder-Mead"}
        values = subset["best_objective"] if derivative_free else subset["objective"]
        style = "--" if derivative_free or method == "Finite Difference" else "-"
        marker = "s" if method in {"Forward AD", "Reverse AD"} else "o"
        axes[0].plot(subset["elapsed_seconds"], values, style, marker=marker, ms=3, color=COLORS.get(method), label=method)
        jitter = 0.08 * (index - (len(methods) - 1) / 2)
        axes[1].plot(subset["iteration"] + jitter, values, style, marker=marker, ms=3, color=COLORS.get(method))
    axes[0].set_xlabel("Wall-clock time (s)")
    axes[1].set_xlabel("Iteration / evaluation")
    axes[0].set_ylabel(r"Loss value $L(\mathbf{u})$")
    for axis in axes:
        axis.set_yscale("log")
        axis.tick_params(direction="in")
    fig.legend(loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
