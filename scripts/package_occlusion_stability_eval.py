"""Create a verified local-only lightweight handoff for six formal units."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import torch

from scripts.prepare_occlusion_stability_eval import DEST, ROOT, UNITS, sha256


def main() -> None:
    delivery = DEST / "delivery"
    if delivery.exists():
        raise FileExistsError(f"will not overwrite existing delivery: {delivery}")
    files: set[Path] = {DEST / "backup/backup_manifest.json"}
    files.update(ROOT / relative for relative in (
        "configs/stability_protocol_v1.yaml", "data/DATA_VERSION.json",
        "docs/VOC20_CHECKPOINT_MANIFEST.json", "docs/B_OCCLUSION_EVAL_MANIFEST.json",
        "docs/B_OCCLUSION_STABILITY_EVAL_MANIFEST.json",
        "docs/B_OCCLUSION_STABILITY_EVAL_HANDOFF.md", "requirements.txt",
        "preprocessing/dataset.py", "models/factory.py",
    ))
    for unit in UNITS:
        output = DEST / unit
        report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed" or report["images"] != 460 or report["trace_rows"] != 2300:
            raise ValueError(f"unit not validated: {unit}")
        base = ROOT / "results/occlusion_eval" / unit
        files.update((base / name for name in ("per_image.csv", "units.csv", "predictions.csv",
                                                 "run_log.jsonl", "validation_report.json")))
        files.add(ROOT / "configs" / f"occlusion_{unit}.yaml")
        for path in output.rglob("*"):
            if path.is_file():
                files.add(path)
        files.update(base.rglob("*.provenance.json"))
        if len(list(base.rglob("*.provenance.json"))) != 460:
            raise ValueError(f"missing source provenance records: {unit}")
    delivery.mkdir(parents=True)
    environment_path = delivery / "environment_snapshot.json"
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "git_head_at_packaging": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                           text=True).strip(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("torchvision", "captum", "scipy", "matplotlib", "numpy", "pandas", "PyYAML")},
    }
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    files.add(environment_path)
    paths = sorted(files)
    if any(path.suffix.lower() in (".npy", ".pt", ".pth", ".zip") for path in paths):
        raise ValueError("binary map, checkpoint or archive would enter light package")
    details = [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                "sha256": sha256(p)} for p in paths]
    manifest = {"schema": 1, "kind": "formal_occlusion_stability_light_handoff",
                "reference_maps_included": False, "checkpoints_included": False,
                "files": details}
    manifest_path = delivery / "file_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = delivery / "occlusion_stability_eval_light.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.write(manifest_path, "file_manifest.json")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        if set(bundle.namelist()) != {entry["path"] for entry in details} | {"file_manifest.json"}:
            raise ValueError("ZIP file coverage differs from manifest")
        for entry in details:
            payload = bundle.read(entry["path"])
            if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP content verification failed: {entry['path']}")
    receipt = {"archive": archive.relative_to(ROOT).as_posix(), "bytes": archive.stat().st_size,
               "sha256": sha256(archive), "file_count": len(details),
               "file_manifest_sha256": sha256(manifest_path), "transferred": False}
    (delivery / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
