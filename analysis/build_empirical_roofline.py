#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def infer_layout(df: pd.DataFrame) -> pd.Series:
    if "layout" in df.columns:
        return df["layout"].astype(str)
    transa = df["transa"].astype(str) if "transa" in df.columns else "N"
    transb = df["transb"].astype(str) if "transb" in df.columns else "N"
    return transa + transb


def normalize_gemm_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    required = {"m", "n", "k", "dtype", "best_tflops"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GEMM CSV missing required columns: {sorted(missing)}")

    bytes_per_elem = {
        "fp16": 2,
        "half": 2,
        "bf16": 2,
        "fp32": 4,
        "tf32": 4,
        "fp64": 8,
    }

    out = df.copy()

    if "bytes_model" not in out.columns:
        def model_bytes(row):
            bpe = bytes_per_elem.get(str(row["dtype"]).lower(), 2)
            m, n, k = int(row["m"]), int(row["n"]), int(row["k"])
            return bpe * (m * k + k * n + 2 * m * n)
        out["bytes_model"] = out.apply(model_bytes, axis=1)

    if "flops" not in out.columns:
        out["flops"] = 2.0 * out["m"] * out["n"] * out["k"]

    if "ai_model" not in out.columns:
        out["ai_model"] = out["flops"] / out["bytes_model"]

    if "family" not in out.columns:
        out["family"] = "gemm"

    if "case_id" not in out.columns:
        out["case_id"] = (
            "m" + out["m"].astype(str)
            + "_n" + out["n"].astype(str)
            + "_k" + out["k"].astype(str)
        )

    out["layout"] = infer_layout(out)

    if "workspace_bytes" not in out.columns:
        out["workspace_bytes"] = 0

    return out


def row_best_tflops(row: pd.Series) -> float:
    if "best_tflops" in row.index and pd.notna(row["best_tflops"]):
        return float(row["best_tflops"])
    if "best_gflops" in row.index and pd.notna(row["best_gflops"]):
        return float(row["best_gflops"]) / 1000.0
    raise KeyError("Neither best_tflops nor best_gflops found in row.")


def row_bw_gbs(row: pd.Series) -> float:
    for col in (
        "bw_effective_gbs",
        "bw_gbs",
        "effective_bw_gbs",
        "effective_bandwidth_gbs",
    ):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])

    if "ai_nominal" in row.index and pd.notna(row["ai_nominal"]):
        ai = float(row["ai_nominal"])
        tflops = row_best_tflops(row)
        return (tflops * 1000.0 / ai) if ai > 0 else 0.0

    raise KeyError(
        "Could not infer bandwidth: no bw_* column and no ai_nominal available."
    )


def max_compute_tflops(df: pd.DataFrame):
    if df.empty:
        return None
    if "best_tflops" in df.columns and df["best_tflops"].notna().any():
        return float(df["best_tflops"].max())
    if "best_gflops" in df.columns and df["best_gflops"].notna().any():
        return float(df["best_gflops"].max()) / 1000.0
    return None


def build_points(
    hbm_df: pd.DataFrame,
    ai_df: pd.DataFrame,
    mix_df: pd.DataFrame,
    gemm_df: pd.DataFrame,
    hbm_ceiling_gbs: float,
    compute_ceiling_tflops: float,
) -> pd.DataFrame:
    rows = []

    if not ai_df.empty:
        for _, row in ai_df.iterrows():
            rows.append(
                {
                    "series": "ai_sweep",
                    "benchmark": "ai_sweep",
                    "family": "synthetic",
                    "dtype": "na",
                    "layout": "na",
                    "workspace_bytes": 0,
                    "label": f"fmas={row.get('fmas_per_load', 'na')}",
                    "ai_flop_per_byte": float(row["ai_nominal"]),
                    "tflops": row_best_tflops(row),
                    "bw_gbs": row_bw_gbs(row),
                }
            )

    if not mix_df.empty:
        for _, row in mix_df.iterrows():
            rows.append(
                {
                    "series": "mem_compute_mix",
                    "benchmark": "mem_compute_mix",
                    "family": "synthetic",
                    "dtype": "na",
                    "layout": "na",
                    "workspace_bytes": 0,
                    "label": f"compute_iters={row.get('compute_iters', 'na')}",
                    "ai_flop_per_byte": float(row["ai_nominal"]),
                    "tflops": row_best_tflops(row),
                    "bw_gbs": row_bw_gbs(row),
                }
            )

    if not gemm_df.empty:
        for _, row in gemm_df.iterrows():
            ai = float(row["ai_model"])
            tflops = float(row["best_tflops"])
            bw_gbs = (tflops * 1000.0 / ai) if ai > 0 else 0.0

            rows.append(
                {
                    "series": f"gemm::{row['family']}",
                    "benchmark": "gemm",
                    "family": row["family"],
                    "dtype": row["dtype"],
                    "layout": row["layout"],
                    "workspace_bytes": int(row["workspace_bytes"]),
                    "label": (
                        f"{row['case_id']}::{row['dtype']}::{row['layout']}"
                        f"::ws{row['workspace_bytes']}"
                    ),
                    "ai_flop_per_byte": ai,
                    "tflops": tflops,
                    "bw_gbs": bw_gbs,
                }
            )

    points = pd.DataFrame(rows)
    if points.empty:
        return points

    ridge_point = compute_ceiling_tflops * 1000.0 / hbm_ceiling_gbs

    def roofline_pred(ai: float) -> float:
        return min(compute_ceiling_tflops, (ai * hbm_ceiling_gbs) / 1000.0)

    points["roofline_pred_tflops"] = points["ai_flop_per_byte"].apply(roofline_pred)
    points["efficiency_vs_roofline"] = points["tflops"] / points["roofline_pred_tflops"]
    points["gap_to_roofline_tflops"] = points["roofline_pred_tflops"] - points["tflops"]
    points["regime"] = points["ai_flop_per_byte"].apply(
        lambda x: "memory-bound" if x < ridge_point else "compute-bound"
    )

    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-campaign-dir", required=True)
    parser.add_argument("--processed-campaign-dir", required=True)
    parser.add_argument("--hbm-ceiling-gbs", type=float, default=None)
    parser.add_argument("--compute-ceiling-tflops", type=float, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_campaign_dir)
    processed_dir = Path(args.processed_campaign_dir)

    roofline_dir = processed_dir / "roofline"
    merged_dir = processed_dir / "merged"
    roofline_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    hbm_df = read_csv_if_exists(raw_dir / "hbm" / "hbm_stream_triad.csv")
    ai_df = read_csv_if_exists(raw_dir / "roofline_calibration" / "ai_sweep.csv")
    mix_df = read_csv_if_exists(raw_dir / "roofline_calibration" / "mem_compute_mix.csv")
    gemm_df = normalize_gemm_df(read_csv_if_exists(raw_dir / "gemm" / "gemm_sweep.csv"))

    if not hbm_df.empty:
        hbm_df.to_csv(merged_dir / "hbm_stream_triad.csv", index=False)
    if not ai_df.empty:
        ai_df.to_csv(merged_dir / "ai_sweep.csv", index=False)
    if not mix_df.empty:
        mix_df.to_csv(merged_dir / "mem_compute_mix.csv", index=False)
    if not gemm_df.empty:
        gemm_df.to_csv(merged_dir / "gemm_sweep.csv", index=False)

    if args.hbm_ceiling_gbs is not None:
        hbm_ceiling_gbs = args.hbm_ceiling_gbs
    else:
        if hbm_df.empty:
            raise SystemExit(
                "[ERROR] HBM CSV is required unless --hbm-ceiling-gbs is provided."
            )
        ceiling_candidates = [
            c for c in ("bw_best_gbs", "bw_mean_gbs", "bw_median_gbs")
            if c in hbm_df.columns
        ]
        if not ceiling_candidates:
            raise SystemExit("[ERROR] HBM CSV has no recognized bandwidth columns.")
        hbm_ceiling_gbs = float(hbm_df[ceiling_candidates[0]].max())

    if args.compute_ceiling_tflops is not None:
        compute_ceiling_tflops = args.compute_ceiling_tflops
    else:
        candidates = []
        gemm_max = max_compute_tflops(gemm_df)
        ai_max = max_compute_tflops(ai_df)
        mix_max = max_compute_tflops(mix_df)
        for c in (gemm_max, ai_max, mix_max):
            if c is not None:
                candidates.append(c)
        if not candidates:
            raise SystemExit("[ERROR] No compute ceiling source available.")
        compute_ceiling_tflops = max(candidates)

    points = build_points(
        hbm_df=hbm_df,
        ai_df=ai_df,
        mix_df=mix_df,
        gemm_df=gemm_df,
        hbm_ceiling_gbs=hbm_ceiling_gbs,
        compute_ceiling_tflops=compute_ceiling_tflops,
    )

    if points.empty:
        raise SystemExit("[ERROR] No roofline points could be built.")

    points.to_csv(roofline_dir / "roofline_points.csv", index=False)

    ridge_point_flop_per_byte = compute_ceiling_tflops * 1000.0 / hbm_ceiling_gbs
    summary = {
        "hbm_ceiling_gbs": hbm_ceiling_gbs,
        "compute_ceiling_tflops": compute_ceiling_tflops,
        "ridge_point_flop_per_byte": ridge_point_flop_per_byte,
        "num_points": int(len(points)),
        "num_gemm_points": int((points["benchmark"] == "gemm").sum()),
    }

    with open(roofline_dir / "empirical_roofline.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()