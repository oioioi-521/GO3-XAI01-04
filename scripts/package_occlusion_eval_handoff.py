"""Build two non-overwriting local delivery ZIPs from a verified file inventory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/occlusion_handoff"
DETAIL = OUT / "B_OCCLUSION_EVAL_FILES.json"
BRIEF = ROOT / "docs/B_OCCLUSION_EVAL_MANIFEST.json"
HANDOFF = ROOT / "docs/B_OCCLUSION_EVAL_HANDOFF.md"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def package(path: Path, files: list[Path]) -> dict:
    if path.exists() or path.with_suffix(".zip.partial").exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_suffix(".zip.partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for file in files:
                archive.write(file, file.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(temporary, "r") as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(files):
                raise ValueError(f"ZIP integrity check failed: {temporary}")
        os.replace(temporary, path)
    except Exception:
        # Preserve partial files for diagnosis; never touch source artifacts.
        raise
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest(path), "files": len(files)}


def main() -> None:
    inventory = json.loads(DETAIL.read_text(encoding="utf-8"))
    brief = json.loads(BRIEF.read_text(encoding="utf-8"))
    if DETAIL.stat().st_size != brief["detailed_manifest"]["bytes"] or digest(DETAIL) != brief["detailed_manifest"]["sha256"]:
        raise ValueError("Detailed manifest mismatch")
    light = {DETAIL, BRIEF, HANDOFF,
             ROOT / "data/DATA_VERSION.json", ROOT / "data/imagenet_class_index.json",
             ROOT / "docs/VOC20_CHECKPOINT_MANIFEST.json", ROOT / "preprocessing/dataset.py"}
    maps = set()
    for unit in inventory["units"]:
        for item in unit["files"]:
            path = ROOT / item["path"]
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"Source drift: {path}")
            (maps if item["kind"] == "float32_map" else light).add(path)
    if len(maps) != 2760:
        raise ValueError(f"Expected 2,760 float maps, got {len(maps)}")
    if (OUT / "package_receipt.json").exists():
        raise FileExistsError("Refusing to overwrite package_receipt.json")
    if any((OUT / name).exists() for name in ("B_OCCLUSION_EVAL_LIGHT.zip", "B_OCCLUSION_EVAL_FLOAT32.zip")):
        raise FileExistsError("Existing handoff ZIP detected")
    receipts = [package(OUT / "B_OCCLUSION_EVAL_LIGHT.zip", sorted(light)),
                package(OUT / "B_OCCLUSION_EVAL_FLOAT32.zip", sorted(maps))]
    receipt = {"schema_version": 1, "experiment_commit": inventory["experiment_commit"],
               "status": "prepared_not_transferred", "packages": receipts,
               "detailed_manifest": brief["detailed_manifest"]}
    (OUT / "package_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
