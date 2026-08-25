#!/usr/bin/env python3
"""Plot representative patient-level mosunetuzumab optimization histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/Mosun_optimization_history.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/figures/SupplementaryFigure9_computational.svg"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    patient_ids = list(data["vpop_id"].drop_duplicates()[:4])
    if not patient_ids:
        raise ValueError("No patient optimization histories were found.")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), squeeze=False)
    for axis, patient_id in zip(axes.flat, patient_ids):
        subset = data[data.vpop_id == patient_id]
        axis.plot(subset.evaluation, subset.total_loss, color="#5b2a86", label="Total")
        axis.plot(subset.evaluation, subset.tumor_loss, color="#4f91bd", alpha=0.8, label="Tumor")
        axis.plot(subset.evaluation, subset.il6_loss, color="#cf5046", alpha=0.8, label="IL-6")
        axis.set(title=f"Virtual patient {patient_id}", xlabel="Objective evaluation", ylabel="Loss")
        axis.set_yscale("log")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
