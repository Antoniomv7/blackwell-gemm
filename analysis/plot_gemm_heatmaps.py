#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "case_id",
    "dtype",
    "layout",
    "workspace_bytes",
    "best_tflops",
}


def check_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(
            f"[ERROR] GEMM CSV missing required columns: {sorted(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemm-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    gemm_csv = Path(args.gemm_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(gemm_csv)
    check_columns(df)

    # Un heatmap por combinación dtype/layout.
    for (dtype, layout), group in df.groupby(["dtype", "layout"]):
        pivot = group.pivot_table(
            index="case_id",
            columns="workspace_bytes",
            values="best_tflops",
            aggfunc="max",
        ).sort_index()

        if pivot.empty:
            continue

        plt.figure(figsize=(10, max(5, 0.45 * len(pivot.index))))
        plt.imshow(pivot.values, aspect="auto")
        plt.colorbar(label="Best TFLOP/s")
        plt.xticks(
            range(len(pivot.columns)),
            [str(c) for c in pivot.columns],
            rotation=45,
        )
        plt.yticks(range(len(pivot.index)), list(pivot.index))
        plt.xlabel("Workspace bytes")
        plt.ylabel("Case ID")
        plt.title(f"GEMM heatmap - dtype={dtype}, layout={layout}")
        plt.tight_layout()

        out = output_dir / f"heatmap_{dtype}_{layout}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[INFO] Saved heatmap -> {out}")


if __name__ == "__main__":
    main()