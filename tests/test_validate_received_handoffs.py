from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.validate_received_handoffs import (
    UNITS,
    ReceivedHandoffError,
    _stable_seed,
    validate_eval_assets,
    validate_package_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_eval_assets(root: Path) -> Path:
    units = []
    for index, unit in enumerate(UNITS):
        path = root / "maps" / f"{unit}.npy"
        path.parent.mkdir(exist_ok=True)
        array = np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
        array = np.roll(array, index, axis=0)
        np.save(path, array, allow_pickle=False)
        units.append(
            {
                "unit": unit,
                "processed": 460,
                "skipped": 0,
                "failed": 0,
                "constant_maps": 0,
                "files": [
                    {
                        "path": f"maps/{unit}.npy",
                        "kind": "float32_map",
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                ],
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "units": units}), encoding="utf-8"
    )
    return manifest


def test_validate_eval_assets_checks_float32_maps(tmp_path):
    manifest = _build_eval_assets(tmp_path)
    result = validate_eval_assets(manifest, [tmp_path], expected_maps=6)
    assert result["units"] == 6
    assert result["files"] == 6
    assert result["float32_maps"] == 6
    assert result["map_shape"] == [224, 224]


def test_validate_eval_assets_rejects_hash_drift(tmp_path):
    manifest = _build_eval_assets(tmp_path)
    path = tmp_path / "maps" / f"{UNITS[0]}.npy"
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ReceivedHandoffError, match="size mismatch"):
        validate_eval_assets(manifest, [tmp_path], expected_maps=6)


def test_validate_package_manifest_checks_exact_file_set(tmp_path):
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("validated", encoding="utf-8")
    manifest = {
        "maps_included": False,
        "files": [
            {
                "path": "evidence.txt",
                "bytes": evidence.stat().st_size,
                "sha256": _sha256(evidence),
            }
        ],
    }
    (tmp_path / "package_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert validate_package_manifest(tmp_path) == {
        "files": 1,
        "maps_included": False,
    }
    (tmp_path / "unlisted.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ReceivedHandoffError, match="file set"):
        validate_package_manifest(tmp_path)


def test_stable_seed_matches_received_trace_contract():
    assert _stable_seed("imagenet", "ILSVRC2012_val_00012132", 0) == 1240890485581281775
