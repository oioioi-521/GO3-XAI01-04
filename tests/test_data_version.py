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


def test_frozen_hash_ignores_platform_line_endings(tmp_path):
    lf = tmp_path / "sample.csv"
    crlf = tmp_path / "sample-windows.csv"
    lf.write_bytes(b"name,value\na,1\n")
    crlf.write_bytes(b"name,value\r\na,1\r\n")

    assert sha256(lf) == sha256(crlf)
