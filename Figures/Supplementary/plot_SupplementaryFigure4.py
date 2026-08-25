#!/usr/bin/env python3
"""Plot tumor trajectories for initial and optimized TGI controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import TGI  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=Path("output/TGI_controls.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/SupplementaryFigure4_computational.svg"))
    args = parser.parse_args()
    params = TGI.make_params()
    optimized = np.zeros(len(params.t_doses))
    if args.controls.exists():
        row = pd.read_csv(args.controls).query("method == 'Forward AD'").iloc[0]
        optimized = row[[f"u_{i}" for i in range(len(params.t_doses))]].to_numpy(float)
    fig, axis = plt.subplots(figsize=(6.5, 3.7))
    for label, doses, color in (("Initial", np.zeros_like(optimized), "0.6"), ("Optimized", optimized, "#b15928")):
        time, states = TGI.forward_solve(doses, params)
        tumor = states[:, 2:6].sum(axis=1)
        axis.plot(time, tumor, label=label, color=color)
    axis.plot(time, TGI.W_ref(time, params), "k--", label="Target")
    axis.set(xlabel="Time (days)", ylabel="Total tumor burden")
    axis.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
