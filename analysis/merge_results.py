#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple single-row or multi-row CSV files into one CSV."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV files"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output merged CSV"
    )
    args = parser.parse_args()

    all_rows = []
    all_columns = []

    for inp in args.inputs:
        path = Path(inp)
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")

        rows = read_csv(path)
        for row in rows:
            all_rows.append(row)
            for key in row.keys():
                if key not in all_columns:
                    all_columns.append(key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[merge_results] Wrote merged CSV: {out_path}")


if __name__ == "__main__":
    main()