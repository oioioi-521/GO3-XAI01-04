"""Describe formal timing outliers without editing, excluding or rerunning rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from analysis.validate_kernelshap_formal import DATASETS, MODELS


def audit(per_image_path: Path, units_path: Path) -> dict:
    with per_image_path.open(encoding="utf-8-sig", newline="") as handle:
        timings = [row for row in csv.DictReader(handle)
                   if row["method"] == "kernelshap" and row["metric"] == "efficiency_time_ms"]
    with units_path.open(encoding="utf-8-sig", newline="") as handle:
        summaries = [row for row in csv.DictReader(handle)
                     if row["method"] == "kernelshap" and row["metric"] == "efficiency_time_ms"]
    expected = {(model, dataset) for model in MODELS for dataset in DATASETS}
    if len(timings) != 2760 or len(summaries) != 6 or {
        (row["model"], row["dataset"]) for row in summaries
    } != expected:
        raise ValueError("require all six formal timing units")
    reports = []
    for model in MODELS:
        for dataset in DATASETS:
            rows = [row for row in timings if (row["model"], row["dataset"]) == (model, dataset)]
            if len(rows) != 460 or len({row["image_id"] for row in rows}) != 460:
                raise ValueError("require 460 unique timing image IDs")
            values = [float(row["value"]) for row in rows]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("timing must be finite and positive")
            summary = next(row for row in summaries
                           if (row["model"], row["dataset"]) == (model, dataset))
            mean, std = statistics.mean(values), statistics.stdev(values)
            if int(summary["n"]) != 460 or not (
                math.isclose(mean, float(summary["mean"]), rel_tol=1e-10, abs_tol=1e-8)
                and math.isclose(std, float(summary["std"]), rel_tol=1e-10, abs_tol=1e-8)
            ):
                raise ValueError("timing aggregation disagrees with formal summary")
            largest = sorted(rows, key=lambda row: float(row["value"]), reverse=True)[:3]
            reports.append({"model": model, "dataset": dataset, "n": 460,
                            "mean_ms": mean, "sample_std_ms": std,
                            "median_ms": statistics.median(values),
                            "p95_ms": statistics.quantiles(values, n=100, method="inclusive")[94],
                            "max_ms": max(values),
                            "largest_single_row_fraction_of_total": max(values) / sum(values),
                            "largest_rows": [{"image_id": row["image_id"], "time_ms": float(row["value"])}
                                             for row in largest]})
    return {"source_per_image_sha256": hashlib.sha256(per_image_path.read_bytes()).hexdigest(),
            "source_units_sha256": hashlib.sha256(units_path.read_bytes()).hexdigest(),
            "units": reports, "excluded_rows": 0,
            "scope": "descriptive timing audit; no corrected efficiency or cross-device ranking",
            "limitation": "elapsed wrapper timings alone cannot identify suspend, interruption, retry or hardware causes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/kernelshap_eval"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.root / "per_image.csv", args.root / "units.csv")
    payload = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
