import csv
import statistics

import pytest

from analysis.audit_kernelshap_timing import DATASETS, MODELS, audit


def _inputs(tmp_path):
    timings, summaries = [], []
    for model in MODELS:
        for dataset in DATASETS:
            values = [10.0] * 459 + [10000.0]
            timings.extend({"image_id": str(i), "model": model, "dataset": dataset,
                            "method": "kernelshap", "metric": "efficiency_time_ms", "value": value}
                           for i, value in enumerate(values))
            summaries.append({"model": model, "dataset": dataset, "method": "kernelshap",
                              "metric": "efficiency_time_ms", "mean": statistics.mean(values),
                              "std": statistics.stdev(values), "n": 460})
    paths = [tmp_path / "per_image.csv", tmp_path / "units.csv"]
    for path, rows in zip(paths, (timings, summaries)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return paths


def test_audit_exposes_outlier_without_excluding_it(tmp_path):
    paths = _inputs(tmp_path)
    original = [path.read_bytes() for path in paths]
    report = audit(*paths)
    assert report["excluded_rows"] == 0 and len(report["units"]) == 6
    assert report["units"][0]["median_ms"] == report["units"][0]["p95_ms"] == 10.0
    assert report["units"][0]["max_ms"] == 10000.0
    assert report["units"][0]["largest_rows"][0]["image_id"] == "459"
    assert original == [path.read_bytes() for path in paths]


def test_audit_rejects_wrong_aggregation(tmp_path):
    paths = _inputs(tmp_path)
    # Alter n only, preserving the original timing observations.
    paths[1].write_text(paths[1].read_text().replace(",460", ",40"), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregation"):
        audit(*paths)
