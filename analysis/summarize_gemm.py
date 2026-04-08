#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "case_id",
    "family",
    "dtype",
    "layout",
    "workspace_bytes",
    "best_tflops",
    "mean_tflops",
    "ai_model",
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gemm_csv = Path(args.gemm_csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(gemm_csv)
    check_columns(df)

    summary = (
        df.groupby(
            ["family", "case_id", "dtype", "layout", "workspace_bytes"],
            as_index=False,
        )
        .agg(
            best_tflops_max=("best_tflops", "max"),
            best_tflops_mean=("best_tflops", "mean"),
            mean_tflops_mean=("mean_tflops", "mean"),
            ai_model_mean=("ai_model", "mean"),
        )
        .sort_values(
            by=["best_tflops_max", "mean_tflops_mean"],
            ascending=[False, False],
        )
    )

    summary.to_csv(output, index=False)
    print(f"[INFO] Saved GEMM summary -> {output}")


if __name__ == "__main__":
    main()