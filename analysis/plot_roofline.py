#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from roofline import logspace, make_roofline_curve, flops_per_second_to_gflops


def read_points_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "benchmark": r["benchmark"],
            "ai_flops_per_byte": float(r["ai_flops_per_byte"]),
            "perf_gflops": float(r["perf_gflops"]),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot roofline from points CSV + ridge JSON.")
    parser.add_argument("--points-csv", required=True)
    parser.add_argument("--ridge-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Empirical Roofline")
    args = parser.parse_args()

    points = read_points_csv(Path(args.points_csv))
    with Path(args.ridge_json).open("r", encoding="utf-8") as f:
        ridge = json.load(f)

    peak_compute_gflops = float(ridge["peak_compute_gflops"])
    peak_bw_gbs = float(ridge["peak_bw_gbs"])
    ridge_ai = float(ridge["ridge_ai_flops_per_byte"])

    peak_compute_flops_s = peak_compute_gflops * 1e9
    peak_bw_bytes_s = peak_bw_gbs * 1e9

    ai_vals = logspace(-3, 4, 300)
    roof_perf = [
        flops_per_second_to_gflops(x)
        for x in make_roofline_curve(ai_vals, peak_compute_flops_s, peak_bw_bytes_s)
    ]

    plt.figure(figsize=(9, 6))
    plt.plot(ai_vals, roof_perf, label="Roofline")
    plt.axvline(ridge_ai, linestyle="--", label=f"Ridge AI = {ridge_ai:.3f}")

    for p in points:
        plt.scatter(p["ai_flops_per_byte"], p["perf_gflops"])
        plt.annotate(
            p["benchmark"],
            (p["ai_flops_per_byte"], p["perf_gflops"]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Arithmetic Intensity [FLOPs/byte]")
    plt.ylabel("Performance [GFLOP/s]")
    plt.title(args.title)
    plt.grid(True, which="both", linestyle=":")
    plt.legend()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[plot_roofline] Wrote plot: {out_path}")


if __name__ == "__main__":
    main()