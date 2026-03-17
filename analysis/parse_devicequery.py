#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_key_value_txt(path: Path) -> dict:
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def load_csv_single_row(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows found in CSV: {path}")
    return rows[0]


def maybe_cast(value: str):
    try:
        if value.isdigit():
            return int(value)
    except Exception:
        pass

    try:
        return int(value)
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        pass

    return value


def normalize_record(d: dict) -> dict:
    return {k: maybe_cast(v) for k, v in d.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Parse MB3.1 device inventory outputs into JSON."
    )
    parser.add_argument("--txt", type=str, default="", help="Path to device_inventory.txt")
    parser.add_argument("--csv", type=str, default="", help="Path to device_inventory.csv")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")

    args = parser.parse_args()

    record = {}

    if args.txt:
        txt_path = Path(args.txt)
        if not txt_path.exists():
            raise FileNotFoundError(f"TXT file not found: {txt_path}")
        record.update(parse_key_value_txt(txt_path))

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        record.update(load_csv_single_row(csv_path))

    if not record:
        raise RuntimeError("No input data provided. Use --txt and/or --csv.")

    record = normalize_record(record)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)

    print(f"[parse_devicequery] Wrote JSON: {out_path}")


if __name__ == "__main__":
    main()