#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

from roofline import (
    compute_ai,
    compute_perf_flops,
    compute_perf_gflops,
    compute_ridge_point_ai,
)


def read_csv_single(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in CSV: {path}")
    return rows[0]


def maybe_num(x: str):
    try:
        if "." in x or "e" in x.lower():
            return float(x)
        return int(x)
    except Exception:
        return x


def normalize_row(row: Dict[str, str]) -> Dict[str, Any]:
    return {k: maybe_num(v) for k, v in row.items()}


def point_from_stream_triad(row: Dict[str, Any]) -> Dict[str, Any]:
    # STREAM triad:
    # per element: 2 FLOPs, 12 bytes
    n = int(row["n"])
    best_ms = float(row["best_ms"])

    flops = 2.0 * n
    bytes_moved = 12.0 * n
    time_s = best_ms * 1e-3

    return {
        "benchmark": "mb34_stream_triad",
        "ai_flops_per_byte": compute_ai(flops, bytes_moved),
        "perf_gflops": compute_perf_gflops(flops, time_s),
        "flops": flops,
        "bytes_moved": bytes_moved,
        "time_s": time_s,
    }


def point_from_ai_control(row: Dict[str, Any]) -> Dict[str, Any]:
    # Approx model:
    # per thread: 2 * fmas_per_load FLOPs
    # bytes: 1 load + 1 store = 8 B
    n = int(row["n"])
    fmas_per_load = int(row["fmas_per_load"])
    best_ms = float(row["best_ms"])

    flops = n * (2.0 * fmas_per_load)
    bytes_moved = n * 8.0
    time_s = best_ms * 1e-3

    return {
        "benchmark": "mb10_ai_control",
        "ai_flops_per_byte": compute_ai(flops, bytes_moved),
        "perf_gflops": compute_perf_gflops(flops, time_s),
        "flops": flops,
        "bytes_moved": bytes_moved,
        "time_s": time_s,
    }


def point_from_mem_compute_mix(row: Dict[str, Any]) -> Dict[str, Any]:
    # Approx model:
    # each visited element -> 1 global load + some compute
    # output is one store per active thread, but for simplicity we use:
    # bytes ≈ n * 4
    # flops ≈ n * 2 * fmas_per_load
    n = int(row["n"])
    fmas_per_load = int(row["fmas_per_load"])
    best_ms = float(row["best_ms"])

    flops = n * (2.0 * fmas_per_load)
    bytes_moved = n * 4.0
    time_s = best_ms * 1e-3

    return {
        "benchmark": "mb3_mem_compute_mix",
        "ai_flops_per_byte": compute_ai(flops, bytes_moved),
        "perf_gflops": compute_perf_gflops(flops, time_s),
        "flops": flops,
        "bytes_moved": bytes_moved,
        "time_s": time_s,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build roofline points and ridge-point data from benchmark CSVs."
    )
    parser.add_argument("--triad-csv", type=str, default="")
    parser.add_argument("--ai-control-csv", type=str, default="")
    parser.add_argument("--mem-compute-csv", type=str, default="")
    parser.add_argument("--peak-compute-gflops", type=float, required=True,
                        help="Measured sustained compute peak in GFLOP/s")
    parser.add_argument("--peak-bw-gbs", type=float, required=True,
                        help="Measured sustained HBM bandwidth in GB/s")
    parser.add_argument("--points-out", type=str, required=True)
    parser.add_argument("--ridge-out", type=str, required=True)

    args = parser.parse_args()

    points: List[Dict[str, Any]] = []

    if args.triad_csv:
        row = normalize_row(read_csv_single(Path(args.triad_csv)))
        points.append(point_from_stream_triad(row))

    if args.ai_control_csv:
        row = normalize_row(read_csv_single(Path(args.ai_control_csv)))
        points.append(point_from_ai_control(row))

    if args.mem_compute_csv:
        row = normalize_row(read_csv_single(Path(args.mem_compute_csv)))
        points.append(point_from_mem_compute_mix(row))

    points_out = Path(args.points_out)
    points_out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "benchmark",
        "ai_flops_per_byte",
        "perf_gflops",
        "flops",
        "bytes_moved",
        "time_s",
    ]

    with points_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            writer.writerow(p)

    peak_compute_flops_s = args.peak_compute_gflops * 1e9
    peak_bw_bytes_s = args.peak_bw_gbs * 1e9
    ridge_ai = compute_ridge_point_ai(peak_compute_flops_s, peak_bw_bytes_s)

    ridge = {
        "peak_compute_gflops": args.peak_compute_gflops,
        "peak_bw_gbs": args.peak_bw_gbs,
        "ridge_ai_flops_per_byte": ridge_ai,
    }

    ridge_out = Path(args.ridge_out)
    ridge_out.parent.mkdir(parents=True, exist_ok=True)
    with ridge_out.open("w", encoding="utf-8") as f:
        json.dump(ridge, f, indent=2)

    print(f"[build_roofline_points] Wrote points CSV: {points_out}")
    print(f"[build_roofline_points] Wrote ridge JSON: {ridge_out}")


if __name__ == "__main__":
    main()