"""Bind a float32 attribution map to its base run or verified recomputation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_path(map_path: Path) -> Path:
    return map_path.with_suffix(map_path.suffix + ".provenance.json")


def write_map_provenance(
    map_path: Path,
    *,
    image_id: str,
    unit: Mapping[str, str],
    split: str,
    base_config_hash: str,
    prediction_context: Mapping[str, str],
    binding: str,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "image_id": str(image_id),
        **{key: unit[key] for key in ("dataset", "model", "method")},
        "split": split,
        "base_config_hash": base_config_hash,
        "prediction_context_hash": prediction_context["config_hash"],
        "checkpoint_sha256": prediction_context["checkpoint_sha256"],
        "map_sha256": sha256_file(map_path),
        "binding": binding,
    }
    path = provenance_path(map_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_map_provenance(
    map_path: Path,
    *,
    image_id: str,
    unit: Mapping[str, str],
    split: str,
    base_config_hash: str,
    prediction_context: Mapping[str, str],
) -> bool:
    """Return false for a legacy map; reject a present but mismatched record."""
    path = provenance_path(map_path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid reference map provenance: {path}") from error
    expected = {
        "schema_version": SCHEMA_VERSION,
        "image_id": str(image_id),
        **{key: unit[key] for key in ("dataset", "model", "method")},
        "split": split,
        "base_config_hash": base_config_hash,
        "prediction_context_hash": prediction_context["config_hash"],
        "checkpoint_sha256": prediction_context["checkpoint_sha256"],
        "map_sha256": sha256_file(map_path),
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"reference map provenance mismatch: {map_path}")
    if payload.get("binding") not in {"base_runner", "verified_recomputation"}:
        raise ValueError(f"unknown reference map provenance binding: {path}")
    return True
