"""Read E's midterm ZIP as data only; do not certify unprovided gate traces/code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image


ROOT = "LIME-same-image-voc8173/"
MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
MAPS = ("lime", "integrated_gradients", "grad_cam", "rise", "occlusion")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(path: Path, project: Path) -> tuple[dict, bytes]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP paths")
        require(all(name.startswith(ROOT) and ".." not in PurePosixPath(name).parts
                    and "\\" not in name and not PurePosixPath(name).is_absolute()
                    for name in (entry.orig_filename for entry in archive.infolist())),
                "unsafe ZIP paths")
        require(all(entry.file_size <= 20_000_000 for entry in archive.infolist())
                and sum(entry.file_size for entry in archive.infolist()) <= 50_000_000,
                "oversized ZIP payload")
        require(archive.testzip() is None, "ZIP CRC failure")

        def read_json(name: str) -> dict:
            return json.loads(archive.read(ROOT + name))

        manifest = read_json("manifest.json")
        require(manifest["split"] == "debug" and manifest["image_id"] == "8173"
                and manifest["model"] == "resnet50"
                and manifest["target"]["class_id"] == 18, "sample identity")
        provenance = manifest["reused_lime_map"]["provenance"]
        require(provenance["method"] == "lime" and provenance["model"] == "resnet50"
                and provenance["dataset"] == "voc" and provenance["split"] == "debug"
                and provenance["image_id"] == "8173"
                and provenance["checkpoint_sha256"] == manifest["checkpoint_sha256"],
                "LIME provenance identity")
        map_report = []
        with np.load(io.BytesIO(archive.read(ROOT + "maps_float32.npz")), allow_pickle=False) as bundle:
            require(set(bundle.files) == set(MAPS), "NPZ map keys")
            for method in MAPS:
                contents = archive.read(ROOT + method + ".npy")
                array = np.load(io.BytesIO(contents), allow_pickle=False)
                require(array.dtype == np.float32 and array.shape == (224, 224)
                        and np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1,
                        f"map contract: {method}")
                require(np.array_equal(array, bundle[method]), f"NPZ disagreement: {method}")
                sha = hashlib.sha256(contents).hexdigest()
                if method == "lime":
                    require(sha == manifest["reused_lime_map"]["sha256"]
                            == provenance["map_sha256"], "LIME map hash")
                map_report.append({"method": method, "sha256": sha, "shape": list(array.shape),
                                   "dtype": str(array.dtype), "constant": bool(np.ptp(array) == 0)})

        # Only use repository-controlled paths, never follow paths supplied by the ZIP.
        checkpoint = project / "models/checkpoints/resnet50_voc20.pt"
        checkpoint_match = (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                            == manifest["checkpoint_sha256"]) if checkpoint.is_file() else None
        require(checkpoint_match is not False, "local checkpoint mismatch")
        input_path = project / "data/voc/raw/008173.jpg"
        input_match = (hashlib.sha256(input_path.read_bytes()).hexdigest()
                       == manifest["image_sha256"]) if input_path.is_file() else None
        require(input_match is not False, "local input image mismatch")
        metadata_path = project / "data/metadata.csv"
        with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
            metadata = [row for row in csv.DictReader(handle)
                        if row["dataset"] == "voc" and row["image_id"] == "8173"]
        require(len(metadata) == 1 and metadata[0]["split"] == "debug"
                and float(metadata[0]["class_id"]) == 18, "frozen metadata identity")

        expected = {f"{model}_{dataset}" for model in MODELS for dataset in DATASETS}
        gates = read_json("handoff_evidence/gate_summary.json")
        status = read_json("handoff_evidence/final_pipeline_status.json")
        require(gates["formal_evaluation_started"] is False
                and gates["formal_budget_frozen"] is False
                and status["formal_evaluation_started"] is False, "formal stage is not frozen")
        require(set(gates["completed_units"]) == expected and set(status["completed_units"]) == expected,
                "completed gate unit set")
        units = gates["units"]
        require(len(units) == 6 and {row["unit"] for row in units} == expected, "gate summary units")
        csv_rows = list(csv.DictReader(io.StringIO(
            archive.read(ROOT + "handoff_evidence/gate_summary.csv").decode("utf-8-sig"))))
        require(len(csv_rows) == 6 and {row["unit"] for row in csv_rows} == expected,
                "gate CSV units")
        csv_by_unit = {row["unit"]: row for row in csv_rows}
        warnings = []
        gate_report = []
        for row in units:
            unit = row["unit"]
            require(row["completed_images"] == row["debug_ids_unique"] == 40
                    and row["map_count"] == 240 and row["constant_map_count"] == 0
                    and row["valid_budget_pairs"] == row["total_budget_pairs"] == 120,
                    f"reported gate counts: {unit}")
            for field in ("completed_images", "map_count", "constant_map_count",
                          "mean_spearman_1000_vs_2000", "p10_spearman_1000_vs_2000",
                          "mean_rmse_1000_vs_2000", "estimated_attribution_seconds"):
                require(math.isclose(float(csv_by_unit[unit][field]), float(row[field]),
                                     rel_tol=1e-10, abs_tol=1e-12), f"gate CSV/JSON: {unit}/{field}")
            require(-1 <= row["p10_spearman_1000_vs_2000"] <= 1
                    and -1 <= row["mean_spearman_1000_vs_2000"] <= 1, f"gate score range: {unit}")
            for budget in ("1000", "2000"):
                pair = row["seed_pair_similarity"][budget]
                require(pair["valid_pairs"] == pair["total_pairs"] == 120
                        and -1 <= pair["p10_spearman"] <= 1
                        and math.isfinite(row["mean_attribution_ms"][budget])
                        and row["mean_attribution_ms"][budget] > 0, f"seed summary: {unit}/{budget}")
            implied_seconds = 120 * sum(row["mean_attribution_ms"].values()) / 1000
            if not math.isclose(implied_seconds, row["estimated_attribution_seconds"], rel_tol=0.01):
                warnings.append(f"{unit}: reported estimated seconds differ from 120 maps per budget x mean ms; source timing logs needed")
            if row["measured_gate_wall_seconds"] is None:
                warnings.append(f"{unit}: gate wall time missing")
            gate_report.append({"unit": unit, "reported_images": 40,
                                "budget_spearman_mean": row["mean_spearman_1000_vs_2000"],
                                "seed_2000_spearman_p10": row["seed_pair_similarity"]["2000"]["p10_spearman"],
                                "reported_estimated_seconds": row["estimated_attribution_seconds"],
                                "implied_attribution_seconds": implied_seconds})

        panel = archive.read(ROOT + "same_image_attribution_comparison.png")
        with Image.open(io.BytesIO(panel)) as picture:
            require(picture.format == "PNG", "panel format")
            dimensions = list(picture.size)
            picture.verify()
        return {"zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "archive_files": len(names), "scope": "midterm sample integrity and supplied summary consistency only",
                "sample": {"image_id": "8173", "target": 18, "split": "debug",
                           "local_checkpoint_hash_matches": checkpoint_match,
                           "local_input_hash_matches": input_match,
                           "panel_dimensions": dimensions, "panel_sha256": hashlib.sha256(panel).hexdigest()},
                "maps": map_report, "reported_gate_summaries": gate_report,
                "formal_eval_started": False, "formal_budget_frozen": False,
                "warnings": warnings,
                "limitations": ["No LIME adapter, six YAMLs, raw gate maps/records, full logs or environment in this ZIP; code and six gate execution not independently accepted",
                                "Budget-seed agreement is not Issue #8 RGB-noise stability",
                                "One qualitative CPU panel; RISE uses 256 masks, not formal 4000 masks",
                                "Sample provenance statements for other methods are not generation-time proof; timing totals not accepted"]}, panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--panel", type=Path)
    args = parser.parse_args()
    report, panel = validate(args.zip, args.project)
    for destination, contents in ((args.report, (json.dumps(report, indent=2) + "\n").encode()),
                                  (args.panel, panel)):
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
