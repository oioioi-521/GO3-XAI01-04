"""Package only lightweight, validated stability-gate evidence locally.

The archive is deliberately not a float32-map transfer and is never uploaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "results/occlusion_stability_gates"
UNITS = tuple(f"{model}_{dataset}" for model in ("resnet50", "densenet121", "vgg16")
              for dataset in ("imagenet", "voc"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    delivery = GATES / "delivery"
    if delivery.exists():
        raise FileExistsError(f"will not overwrite existing delivery: {delivery}")
    paths: list[Path] = []
    for stage in ("single", "debug40"):
        for unit in UNITS:
            gate = GATES / stage / unit
            report = json.loads((gate / "validation_report.json").read_text(encoding="utf-8"))
            if report["status"] != "passed" or report["images"] != (1 if stage == "single" else 40):
                raise ValueError(f"not validated: {gate}")
            for path in gate.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix == ".npy" or path.suffix == ".pt":
                    continue
                paths.append(path)
    delivery.mkdir(parents=True)
    manifest = {
        "scope": "candidate single and debug40 Occlusion stability engineering gates",
        "maps_included": False,
        "source_root": "results/occlusion_stability_gates/",
        "files": [{"path": p.relative_to(GATES).as_posix(), "bytes": p.stat().st_size,
                   "sha256": sha256(p)} for p in sorted(paths)],
    }
    manifest_path = delivery / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = delivery / "occlusion_stability_gates_light.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in paths:
            output.write(path, path.relative_to(GATES).as_posix())
        output.write(manifest_path, "package_manifest.json")
    with zipfile.ZipFile(archive) as bundle:
        for item in manifest["files"]:
            payload = bundle.read(item["path"])
            if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"archive verification failed: {item['path']}")
    receipt = {"archive": archive.relative_to(ROOT).as_posix(), "bytes": archive.stat().st_size,
               "sha256": sha256(archive), "files": len(paths),
               "manifest_sha256": sha256(manifest_path), "transferred": False}
    (delivery / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
