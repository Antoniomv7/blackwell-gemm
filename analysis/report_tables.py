#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_csvs(root: Path) -> List[Path]:
    return sorted(root.rglob("*.csv"))


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables from benchmark CSV files.")
    parser.add_argument("--input-root", required=True, help="Root directory with raw results")
    parser.add_argument("--output-dir", required=True, help="Directory for summary tables")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs = collect_csvs(input_root)
    if not csvs:
        raise RuntimeError(f"No CSV files found under: {input_root}")

    summary_rows: List[Dict[str, str]] = []

    for csv_path in csvs:
        rows = load_csv(csv_path)
        for row in rows:
            out = dict(row)
            out["source_file"] = str(csv_path)
            summary_rows.append(out)

    # Build superset of columns
    fieldnames: List[str] = []
    for row in summary_rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    write_csv(output_dir / "summary_all.csv", summary_rows, fieldnames)
    print(f"[report_tables] Wrote: {output_dir / 'summary_all.csv'}")


if __name__ == "__main__":
    main()