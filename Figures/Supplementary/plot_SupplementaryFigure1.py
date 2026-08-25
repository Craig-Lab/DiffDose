#!/usr/bin/env python3
"""Plot IDR state trajectories at fixed and optimized dose amplitudes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import IDR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=Path("output/IDR_controls.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/SupplementaryFigure1_computational.svg"))
    args = parser.parse_args()
    params = IDR.make_params()
    controls = np.ones(6)
    if args.controls.exists():
        table = pd.read_csv(args.controls)
        row = table[table.method == "Forward AD"].iloc[0]
        controls = row[[f"u_{i}" for i in range(6)]].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
    for label, dose, color in (("Initial", np.ones(6), "0.6"), ("Optimized", controls, "#b15928")):
        time, states = IDR.forward_solve(dose, params)
        axes[0].plot(time, states[:, 0], label=label, color=color)
        axes[1].plot(time, states[:, 1], label=label, color=color)
    axes[1].plot(time, IDR.B_ref(time, params), "k--", label="Target")
    axes[0].set_ylabel("Drug concentration")
    axes[1].set_ylabel("Biomarker")
    for axis in axes:
        axis.set_xlabel("Time (days)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
