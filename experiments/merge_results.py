"""Merge per-image CSV files and rebuild the unit-level summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PER_IMAGE_COLUMNS = ["image_id", "dataset", "model", "method", "metric", "value", "time_ms"]
UNIT_COLUMNS = ["method", "model", "dataset", "metric", "mean", "std", "n", "config_hash"]


def merge(inputs: list[Path], output: Path, units_output: Path) -> None:
    frames = []
    for path in inputs:
        frame = pd.read_csv(path)
        missing = set(PER_IMAGE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame[PER_IMAGE_COLUMNS])
    if not frames:
        raise ValueError("at least one input CSV is required")

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["image_id", "dataset", "model", "method", "metric"], keep="last"
    )
    merged = merged.sort_values(["method", "model", "dataset", "image_id", "metric"])
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    units = merged.groupby(["method", "model", "dataset", "metric"])["value"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    units["std"] = units["std"].fillna(0.0)
    units["config_hash"] = "merged-see-config-snapshots"
    units = units[UNIT_COLUMNS]
    units_output.parent.mkdir(parents=True, exist_ok=True)
    units.to_csv(units_output, index=False)
    print(f"merged {len(inputs)} files -> {len(merged)} rows; {len(units)} unit rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/per_image.csv"))
    parser.add_argument("--units-output", type=Path, default=Path("results/units.csv"))
    args = parser.parse_args()
    merge(args.inputs, args.output, args.units_output)


if __name__ == "__main__":
    main()
