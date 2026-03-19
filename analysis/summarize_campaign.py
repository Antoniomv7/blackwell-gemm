#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a short text summary for a processed roofline campaign."
    )
    parser.add_argument("--processed-campaign-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    processed = Path(args.processed_campaign_dir)
    summary_json = processed / "roofline" / "empirical_roofline.json"
    points_csv = processed / "roofline" / "roofline_points.csv"

    if not summary_json.exists():
        raise SystemExit(
            f"[summarize_campaign] Missing summary JSON: {summary_json}"
        )
    if not points_csv.exists():
        raise SystemExit(
            f"[summarize_campaign] Missing points CSV: {points_csv}"
        )

    with open(summary_json, "r", encoding="utf-8") as f:
        roof = json.load(f)

    points = pd.read_csv(points_csv)

    lines = []
    lines.append(f"Campaign summary: {processed.name}")
    lines.append("")
    lines.append(f"HBM ceiling [GB/s]: {roof['hbm_ceiling_gbs']:.3f}")
    lines.append(f"Compute ceiling [GFLOP/s]: {roof['compute_ceiling_gflops']:.3f}")
    lines.append(f"Ridge point [FLOP/B]: {roof['ridge_point_flops_per_byte']:.6f}")
    lines.append(f"Compute ceiling source: {roof['compute_ceiling_source']}")
    lines.append(f"Total roofline points: {roof['num_points']}")
    lines.append("")

    for series, group in points.groupby("series"):
        best = group.sort_values("perf_gflops", ascending=False).iloc[0]
        lines.append(f"Series: {series}")
        lines.append(f"  points: {len(group)}")
        lines.append(f"  best point: {best['label']}")
        lines.append(f"  best perf [GFLOP/s]: {best['perf_gflops']:.3f}")
        lines.append(f"  AI [FLOP/B]: {best['ai_flops_per_byte']:.6f}")
        if "efficiency_vs_roofline" in best.index:
            lines.append(
                f"  efficiency vs roofline: {float(best['efficiency_vs_roofline']):.6f}"
            )
        lines.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize_campaign] Wrote summary: {out}")


if __name__ == "__main__":
    main()