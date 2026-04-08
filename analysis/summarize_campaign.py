#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-campaign-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    processed = Path(args.processed_campaign_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    roofline_summary_path = processed / "roofline" / "empirical_roofline.json"
    roofline_points_path = processed / "roofline" / "roofline_points.csv"
    gemm_summary_path = processed / "tables" / "gemm_summary.csv"

    with open(roofline_summary_path) as f:
        roof = json.load(f)

    points = pd.read_csv(roofline_points_path)

    hbm_ceiling_gbs = roof.get("hbm_ceiling_gbs")
    compute_ceiling_tflops = roof.get("compute_ceiling_tflops")
    ridge_point = roof.get("ridge_point_flop_per_byte")
    num_points = roof.get("num_points")
    num_gemm_points = roof.get("num_gemm_points")

    lines = []
    lines.append(f"Campaign summary for: {processed.name}")
    lines.append("")
    if hbm_ceiling_gbs is not None:
        lines.append(f"HBM ceiling [GB/s]: {hbm_ceiling_gbs:.3f}")
    if compute_ceiling_tflops is not None:
        lines.append(f"Compute ceiling [TFLOP/s]: {compute_ceiling_tflops:.3f}")
        lines.append(f"Compute ceiling [GFLOP/s]: {compute_ceiling_tflops * 1000.0:.3f}")
    if ridge_point is not None:
        lines.append(f"Ridge point [FLOP/B]: {ridge_point:.3f}")
    if num_points is not None:
        lines.append(f"Total roofline points: {num_points}")
    if num_gemm_points is not None:
        lines.append(f"GEMM roofline points: {num_gemm_points}")

    lines.append("")

    if "benchmark" in points.columns and (points["benchmark"] == "gemm").any():
        gemm_points = points[points["benchmark"] == "gemm"].copy()
        best_gemm = gemm_points.sort_values("tflops", ascending=False).head(10)

        lines.append("Top GEMM points by TFLOP/s:")
        for _, row in best_gemm.iterrows():
            lines.append(
                f"  - {row['label']}: "
                f"{row['tflops']:.3f} TFLOP/s, "
                f"AI={row['ai_flop_per_byte']:.3f}, "
                f"eff_vs_roofline={row['efficiency_vs_roofline']:.3f}"
            )
        lines.append("")
    else:
        lines.append("No GEMM points were found in roofline_points.csv.")
        lines.append("")

    if gemm_summary_path.exists():
        lines.append(f"GEMM summary table: {gemm_summary_path}")

    output.write_text("\n".join(lines))
    print(f"[INFO] Saved campaign summary -> {output}")


if __name__ == "__main__":
    main()