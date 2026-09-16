"""Verify the frozen W2 data baseline recorded in data/DATA_VERSION.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((DATA_ROOT / "DATA_VERSION.json").read_text(encoding="utf-8"))
    failures = []
    for relative_path, expected in manifest["sha256"].items():
        actual = sha256(DATA_ROOT / relative_path)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{status:8} {relative_path}")
        if actual != expected:
            failures.append(relative_path)

    for dataset, expected in manifest["datasets"].items():
        actual_count = sum(1 for path in (DATA_ROOT / dataset / "raw").iterdir() if path.is_file())
        status = "OK" if actual_count == expected["raw_images"] else "MISMATCH"
        print(f"{status:8} {dataset}/raw ({actual_count} files)")
        if actual_count != expected["raw_images"]:
            failures.append(f"{dataset}/raw")

    if failures:
        raise SystemExit(f"data baseline verification failed: {', '.join(failures)}")
    print(f"verified data baseline: {manifest['version']}")


if __name__ == "__main__":
    main()
