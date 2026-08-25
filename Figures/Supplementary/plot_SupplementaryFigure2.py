#!/usr/bin/env python3
"""Regenerate the IDR epsilon-window versus exact-jump comparison.

The script writes a computational base beneath ``output/`` and never replaces
the manually finished manuscript SVG.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import IDR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("quick", "paper"), default="quick")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/SupplementaryFigure2_computational.svg"),
    )
    parser.add_argument("--results", type=Path, default=Path("output"))
    args = parser.parse_args()

    results = IDR.run_dosing_representation_comparison(
        preset=args.preset,
        output=args.results,
        timing="warm",
    )
    colors = {
        "Analytic adjoint (OptiDose)": "#25557f",
        "Forward-mode AD (jump)": "#b45f2a",
        "Forward-mode AD (smooth)": "#f28e2b",
        "Reverse-mode AD (jump)": "#4e79a7",
        "Reverse-mode AD (smooth)": "#9ecae1",
    }
    offsets = np.linspace(-0.06, 0.06, len(results))
    figure, (iteration_axis, time_axis) = plt.subplots(
        1, 2, figsize=(10, 4.2), sharey=True
    )
    for result, offset in zip(results, offsets):
        history = result.history
        color = colors[result.method]
        iteration_axis.plot(
            history["iteration"] + offset,
            history["objective"],
            marker="o",
            color=color,
            label=result.method,
        )
        time_axis.plot(
            history["elapsed_seconds"],
            history["objective"],
            marker="o",
            color=color,
            label=result.method,
        )

    iteration_axis.set_xlabel("Iteration / evaluation")
    iteration_axis.set_ylabel(r"Objective $J(\mathbf{u})$")
    time_axis.set_xlabel("Wall-clock time (s)")
    time_axis.legend(frameon=True, fontsize=9)
    iteration_axis.text(
        0.0,
        1.03,
        "a",
        transform=iteration_axis.transAxes,
        fontsize=16,
        fontweight="bold",
    )
    time_axis.text(
        0.0,
        1.03,
        "b",
        transform=time_axis.transAxes,
        fontsize=16,
        fontweight="bold",
    )
    for axis in (iteration_axis, time_axis):
        axis.tick_params(which="both", direction="in")

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
