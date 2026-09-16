"""Validate result CSVs before aggregation or statistical analysis.

Example:
    python -m analysis.validate_results --per-image results/per_image.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .schema import read_csv, validate_per_image, validate_units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-image", type=Path)
    parser.add_argument("--units", type=Path)
    args = parser.parse_args()
    if not args.per_image and not args.units:
        parser.error("provide --per-image, --units, or both")

    report: dict[str, object] = {}
    if args.per_image:
        report["per_image"] = asdict(
            validate_per_image(read_csv(args.per_image, "per_image"))
        )
    if args.units:
        report["units"] = asdict(validate_units(read_csv(args.units, "units")))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
