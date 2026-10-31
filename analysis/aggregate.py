"""Aggregate per-image results into the project unit-level long table.

Example:
    python -m analysis.aggregate --input results/per_image.csv \
        --units results/units.csv --output analysis/data_long.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .schema import UNIT_COLUMNS, UNIT_KEY, read_csv, validate_per_image, validate_units


def aggregate_per_image(
    per_image: pd.DataFrame, existing_units: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return one row per method/model/dataset/metric.

    ``config_hash`` cannot be reconstructed from ``per_image.csv`` because the
    agreed per-image schema does not contain it.  When the runner-generated
    ``units.csv`` is supplied, its hash is joined by the unit key; otherwise the
    field is deliberately left blank rather than guessed.
    """

    validate_per_image(per_image)
    working = per_image.copy()
    working["value"] = pd.to_numeric(working["value"], errors="raise")
    grouped = (
        working.groupby(UNIT_KEY, as_index=False, sort=True)["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index(drop=True)
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    if existing_units is not None:
        validate_units(existing_units)
        hashes = existing_units[UNIT_KEY + ["config_hash"]].copy()
        grouped = grouped.merge(hashes, on=UNIT_KEY, how="left", validate="one_to_one")
        grouped["config_hash"] = grouped["config_hash"].fillna("")
    else:
        grouped["config_hash"] = ""
    return grouped[UNIT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="per-image CSV")
    parser.add_argument("--units", type=Path, help="optional runner-generated units CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_image = read_csv(args.input, "per_image")
    units = read_csv(args.units, "units") if args.units else None
    result = aggregate_per_image(per_image, units)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"aggregated {len(per_image)} per-image rows into {len(result)} unit rows")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
