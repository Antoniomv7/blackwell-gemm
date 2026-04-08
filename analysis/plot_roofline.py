#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roofline-points", required=True)
    parser.add_argument("--roofline-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    points = pd.read_csv(args.roofline_points)
    with open(args.roofline_summary) as f:
        summary = json.load(f)

    hbm_ceiling_gbs = float(summary["hbm_ceiling_gbs"])
    compute_ceiling_tflops = float(summary["compute_ceiling_tflops"])
    ridge = float(summary["ridge_point_flop_per_byte"])

    x_min = max(points["ai_flop_per_byte"].min() * 0.5, 1e-2)
    x_max = points["ai_flop_per_byte"].max() * 2.0

    xs = np.logspace(np.log10(x_min), np.log10(x_max), 512)
    ys = np.minimum(compute_ceiling_tflops, (xs * hbm_ceiling_gbs) / 1000.0)

    plt.figure(figsize=(11, 7))
    plt.loglog(xs, ys, linewidth=2, label="Empirical roofline")
    plt.axvline(
        ridge,
        linestyle="--",
        linewidth=1.5,
        label=f"Ridge point = {ridge:.2f} FLOP/B",
    )

    synthetic_markers = {
        "ai_sweep": "o",
        "mem_compute_mix": "s",
    }

    for series_name, group in points.groupby("series"):
        if series_name in synthetic_markers:
            plt.scatter(
                group["ai_flop_per_byte"],
                group["tflops"],
                s=42,
                alpha=0.85,
                marker=synthetic_markers[series_name],
                label=series_name,
            )

    gemm = points[points["benchmark"] == "gemm"].copy()
    if not gemm.empty:
        gemm_markers = ["^", "v", "D", "P", "X", "*", "<", ">"]
        for idx, (family, group) in enumerate(gemm.groupby("family")):
            marker = gemm_markers[idx % len(gemm_markers)]
            plt.scatter(
                group["ai_flop_per_byte"],
                group["tflops"],
                s=46,
                alpha=0.82,
                marker=marker,
                label=f"gemm::{family}",
            )

    plt.xlabel("Arithmetic intensity [FLOP/byte]")
    plt.ylabel("Performance [TFLOP/s]")
    plt.title("Empirical roofline with GEMM integration")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=220)
    print(f"[INFO] Saved roofline plot -> {output}")


if __name__ == "__main__":
    main()