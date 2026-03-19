#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot empirical roofline from roofline_points.csv + empirical_roofline.json"
    )
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--ridge-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--title",
        default="Empirical Roofline (HBM + Synthetic Calibration)",
    )
    args = parser.parse_args()

    points = pd.read_csv(args.points_csv)

    with open(args.ridge_json, "r", encoding="utf-8") as f:
        ridge = json.load(f)

    required_cols = {"series", "ai_flops_per_byte", "perf_gflops"}
    missing = required_cols - set(points.columns)
    if missing:
        raise SystemExit(
            f"[plot_roofline] Missing columns in points CSV: {sorted(missing)}"
        )

    hbm_ceiling_gbs = float(ridge["hbm_ceiling_gbs"])
    compute_ceiling_gflops = float(ridge["compute_ceiling_gflops"])
    ridge_ai = float(ridge["ridge_point_flops_per_byte"])

    x_min = max(points["ai_flops_per_byte"].min() * 0.5, 1e-3)
    x_max = max(points["ai_flops_per_byte"].max() * 2.0, ridge_ai * 4.0)

    xs = np.logspace(np.log10(x_min), np.log10(x_max), 512)
    ys = np.minimum(compute_ceiling_gflops, xs * hbm_ceiling_gbs)

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, linewidth=2, label="Empirical roofline")
    plt.axvline(
        ridge_ai,
        linestyle="--",
        linewidth=1.5,
        label=f"Ridge AI = {ridge_ai:.3f} FLOP/B",
    )

    for series, group in points.groupby("series"):
        plt.scatter(
            group["ai_flops_per_byte"],
            group["perf_gflops"],
            s=42,
            alpha=0.85,
            label=series,
        )

        if "label" in group.columns:
            for _, row in group.iterrows():
                plt.annotate(
                    str(row["label"]),
                    (row["ai_flops_per_byte"], row["perf_gflops"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic intensity [FLOP/byte]")
    plt.ylabel("Performance [GFLOP/s]")
    plt.title(args.title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[plot_roofline] Wrote plot: {out_path}")


if __name__ == "__main__":
    main()