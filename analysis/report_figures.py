#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_single_row_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in CSV: {path}")
    return rows[0]


def maybe_make_triad_plot(input_root: Path, output_dir: Path):
    matches = list(input_root.rglob("mb34_stream_triad.csv"))
    if not matches:
        return
    row = read_single_row_csv(matches[0])
    bw = float(row["bw_best_gbs"])

    plt.figure(figsize=(6, 4))
    plt.bar(["stream_triad"], [bw])
    plt.ylabel("GB/s")
    plt.title("Sustained HBM bandwidth (STREAM triad)")
    plt.tight_layout()
    out = output_dir / "triad_bw.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[report_figures] Wrote: {out}")


def maybe_make_ptr_plot(input_root: Path, output_dir: Path):
    matches = list(input_root.rglob("mb34_ptr_chase.csv"))
    if not matches:
        return
    row = read_single_row_csv(matches[0])
    cyc = float(row["cycles_per_iter"])

    plt.figure(figsize=(6, 4))
    plt.bar(["ptr_chase"], [cyc])
    plt.ylabel("Cycles / iteration")
    plt.title("Pointer-chasing effective latency")
    plt.tight_layout()
    out = output_dir / "ptr_chase_cycles.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[report_figures] Wrote: {out}")


def maybe_make_gemm_plot(input_root: Path, output_dir: Path):
    matches = list(input_root.rglob("mb4_mb5_cublaslt_baseline.csv"))
    if not matches:
        return
    row = read_single_row_csv(matches[0])
    tflops = float(row["best_tflops"])

    plt.figure(figsize=(6, 4))
    plt.bar(["cublaslt_baseline"], [tflops])
    plt.ylabel("TFLOP/s")
    plt.title("cuBLASLt baseline GEMM")
    plt.tight_layout()
    out = output_dir / "cublaslt_baseline_tflops.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[report_figures] Wrote: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate summary figures from benchmark CSV files.")
    parser.add_argument("--input-root", required=True, help="Root directory with raw/processed results")
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maybe_make_triad_plot(input_root, output_dir)
    maybe_make_ptr_plot(input_root, output_dir)
    maybe_make_gemm_plot(input_root, output_dir)


if __name__ == "__main__":
    main()