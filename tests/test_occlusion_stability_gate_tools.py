"""Safety contracts for copying existing debug evidence into isolated gates."""

import csv

import pytest

from scripts import prepare_occlusion_stability_gates as gates


def test_copy_verified_keeps_source_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "ROOT", tmp_path)
    source = tmp_path / "source.bin"
    source.write_bytes(b"original evidence")
    destination = tmp_path / "gate" / "source.bin"
    ledger = []

    gates.copy_verified(source, destination, ledger)
    assert destination.read_bytes() == source.read_bytes()
    assert ledger[0]["sha256"] == gates.sha256(source)
    with pytest.raises(FileExistsError):
        gates.copy_verified(source, destination, ledger)
    assert source.read_bytes() == b"original evidence"


def test_write_subset_records_source_and_selected_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "ROOT", tmp_path)
    source = tmp_path / "all.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "value"])
        writer.writeheader()
        writer.writerows([{"image_id": "one", "value": "1"},
                          {"image_id": "two", "value": "2"}])
    initial_hash = gates.sha256(source)
    destination = tmp_path / "gate" / "subset.csv"
    ledger = []

    gates.write_subset(source, destination, {"two"}, ledger)
    assert gates.csv_rows(destination) == [{"image_id": "two", "value": "2"}]
    assert gates.sha256(source) == initial_hash == ledger[0]["source_sha256"]
    assert ledger[0]["sha256"] == gates.sha256(destination)
    with pytest.raises(FileExistsError):
        gates.write_subset(source, destination, {"one"}, ledger)
