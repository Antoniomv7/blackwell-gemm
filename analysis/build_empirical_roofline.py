#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[build_empirical_roofline] Missing {label} CSV: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"[build_empirical_roofline] Empty {label} CSV: {path}")

    return df


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"[build_empirical_roofline] {label} CSV missing columns: {missing}"
        )


def normalize_points(
    df: pd.DataFrame,
    *,
    series: str,
    x_col: str,
    y_col: str,
    label_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    out["series"] = series
    out["ai_flops_per_byte"] = out[x_col].astype(float)
    out["perf_gflops"] = out[y_col].astype(float)
    out["label"] = out[label_cols].astype(str).agg(" | ".join, axis=1)

    keep_cols = [
        "series",
        "benchmark",
        "label",
        "ai_flops_per_byte",
        "perf_gflops",
        "best_ms",
        "mean_ms",
        "median_ms",
        "bw_best_gbs",
        "bw_mean_gbs",
        "bw_median_gbs",
    ]
    existing = [c for c in keep_cols if c in out.columns]
    return out[existing].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build empirical roofline artifacts from HBM + calibration CSVs."
    )
    parser.add_argument("--hbm-csv", required=True)
    parser.add_argument("--ai-csv", required=True)
    parser.add_argument("--mix-csv", required=True)
    parser.add_argument("--processed-campaign-dir", required=True)
    args = parser.parse_args()

    hbm_csv = Path(args.hbm_csv)
    ai_csv = Path(args.ai_csv)
    mix_csv = Path(args.mix_csv)
    processed_dir = Path(args.processed_campaign_dir)

    merged_dir = processed_dir / "merged"
    roofline_dir = processed_dir / "roofline"
    merged_dir.mkdir(parents=True, exist_ok=True)
    roofline_dir.mkdir(parents=True, exist_ok=True)

    hbm_df = read_required_csv(hbm_csv, "HBM")
    ai_df = read_required_csv(ai_csv, "AI sweep")
    mix_df = read_required_csv(mix_csv, "mem/compute mix")

    require_columns(
        hbm_df,
        ["benchmark", "bw_best_gbs", "best_ms", "block", "n"],
        "HBM",
    )
    require_columns(
        ai_df,
        ["benchmark", "ai_nominal", "best_gflops", "best_ms", "fmas_per_load"],
        "AI sweep",
    )
    require_columns(
        mix_df,
        ["benchmark", "ai_nominal", "best_gflops", "best_ms", "compute_iters"],
        "mem/compute mix",
    )

    hbm_df.to_csv(merged_dir / "hbm_stream_triad.csv", index=False)
    ai_df.to_csv(merged_dir / "ai_sweep.csv", index=False)
    mix_df.to_csv(merged_dir / "mem_compute_mix.csv", index=False)

    hbm_ceiling_gbs = float(hbm_df["bw_best_gbs"].max())

    ai_compute_gflops = float(ai_df["best_gflops"].max())
    mix_compute_gflops = float(mix_df["best_gflops"].max())

    compute_ceiling_gflops = max(ai_compute_gflops, mix_compute_gflops)
    compute_ceiling_source = (
        "ai_sweep" if ai_compute_gflops >= mix_compute_gflops else "mem_compute_mix"
    )

    ridge_ai = compute_ceiling_gflops / hbm_ceiling_gbs

    ai_points = normalize_points(
        ai_df,
        series="ai_sweep",
        x_col="ai_nominal",
        y_col="best_gflops",
        label_cols=["benchmark", "fmas_per_load"],
    )

    mix_points = normalize_points(
        mix_df,
        series="mem_compute_mix",
        x_col="ai_nominal",
        y_col="best_gflops",
        label_cols=["benchmark", "compute_iters"],
    )

    points = pd.concat([ai_points, mix_points], ignore_index=True)
    points["roofline_pred_gflops"] = points["ai_flops_per_byte"].apply(
        lambda x: min(compute_ceiling_gflops, x * hbm_ceiling_gbs)
    )
    points["efficiency_vs_roofline"] = (
        points["perf_gflops"] / points["roofline_pred_gflops"]
    )
    points["regime"] = points["ai_flops_per_byte"].apply(
        lambda x: "memory-bound" if x < ridge_ai else "compute-bound"
    )
    points = points.sort_values(
        ["series", "ai_flops_per_byte", "perf_gflops"]
    ).reset_index(drop=True)

    points.to_csv(roofline_dir / "roofline_points.csv", index=False)

    summary = {
        "hbm_ceiling_gbs": hbm_ceiling_gbs,
        "compute_ceiling_gflops": compute_ceiling_gflops,
        "ridge_point_flops_per_byte": ridge_ai,
        "hbm_source_csv": str(hbm_csv),
        "ai_source_csv": str(ai_csv),
        "mix_source_csv": str(mix_csv),
        "compute_ceiling_source": compute_ceiling_source,
        "num_points": int(len(points)),
        "num_ai_points": int(len(ai_points)),
        "num_mix_points": int(len(mix_points)),
    }

    with open(roofline_dir / "empirical_roofline.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()