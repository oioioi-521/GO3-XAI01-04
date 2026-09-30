import json
from pathlib import Path

from preprocessing.verify_data_version import DATA_ROOT, sha256


def test_tracked_data_hashes_match_frozen_manifest():
    manifest = json.loads((DATA_ROOT / "DATA_VERSION.json").read_text(encoding="utf-8"))

    assert {
        relative_path: sha256(DATA_ROOT / relative_path)
        for relative_path in manifest["sha256"]
    } == manifest["sha256"]


def test_data_files_are_declared_with_portable_lf_endings():
    project_root = Path(__file__).resolve().parents[1]
    attributes = (project_root / ".gitattributes").read_text(encoding="utf-8")

    assert "data/*.csv text eol=lf" in attributes
    assert "data/*.json text eol=lf" in attributes


def test_frozen_text_hash_is_independent_of_windows_checkout_endings(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_bytes(b"image_id,class_id\n1,2\n")
    expected = sha256(path)
    path.write_bytes(b"image_id,class_id\r\n1,2\r\n")
    assert sha256(path) == expected
    path.write_bytes(b"image_id,class_id\r\n1,3\r\n")
    assert sha256(path) != expected
