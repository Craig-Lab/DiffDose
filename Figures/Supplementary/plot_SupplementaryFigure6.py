#!/usr/bin/env python3
"""Plot BiTE complex trajectories for initial and optimized amplitudes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import BiTE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=Path("output/BiTE_controls.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/SupplementaryFigure6_computational.svg"))
    args = parser.parse_args()
    params = BiTE.make_params()
    optimized = 800.0
    if args.controls.exists():
        optimized = float(pd.read_csv(args.controls).query("method == 'Forward AD'").iloc[0]["u_0"])
    fig, axis = plt.subplots(figsize=(6.5, 3.7))
    for label, dose, color in (("Initial", 800.0, "0.6"), ("Optimized", optimized, "#b15928")):
        time, states = BiTE.forward_solve(dose, params)
        axis.plot(time, states[:, 5], label=label, color=color)
    axis.axhline(params.R_ref, color="k", linestyle="--", label="Target")
    axis.set(xlabel="Time (days)", ylabel="Ternary complex")
    axis.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
