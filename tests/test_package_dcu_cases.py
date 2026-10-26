"""Synthetic packaging fixtures only; these are not formal attribution evidence."""
import copy
import io
import json
import tarfile
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from analysis.package_dcu_cases import (
    METHODS, build_packet, check_map, collect_tar, file_sha, sha,
)


def sample(method="ig", image_id="215", dtype=np.float32):
    record = dict(method=method, image_id=image_id, model="resnet50", dataset="voc",
                  split="eval", checkpoint_sha256="a" * 64, base_config_hash="base",
                  prediction_context_hash="context", target=7)
    npy = io.BytesIO()
    np.save(npy, np.full((224, 224), .5, dtype=dtype), allow_pickle=False)
    png = io.BytesIO()
    Image.new("L", (224, 224), 128).save(png, format="PNG")
    provenance = {k: v for k, v in record.items() if k != "target"}
    provenance["map_sha256"] = sha(npy.getvalue())
    return record, npy.getvalue(), png.getvalue(), json.dumps(provenance).encode()


def test_original_sidecar_does_not_need_fabricated_target():
    r, *data = sample()
    check_map(r, *data)
    assert "target" not in json.loads(data[2])


@pytest.mark.parametrize("field", ["method", "image_id", "checkpoint_sha256", "prediction_context_hash", "map_sha256"])
def test_rejects_mismatched_provenance(field):
    r, npy, png, sidecar = sample()
    p = json.loads(sidecar)
    p[field] = "wrong"
    with pytest.raises(ValueError):
        check_map(r, npy, png, json.dumps(p).encode())


def test_rejects_float64_and_frozen_hash_mismatch():
    with pytest.raises(ValueError, match="dtype"):
        check_map(*sample(dtype=np.float64))
    r, *data = sample()
    r["expected_npy_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="completion"):
        check_map(r, *data)


def archive(tmp_path, members):
    path = tmp_path / "source.tar.gz"
    with tarfile.open(path, "w:gz") as t:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return path


def test_selective_archive_read_and_source_pin(tmp_path):
    path = archive(tmp_path, [("outer/run/a.npy", b"saved"), ("outer/unselected", b"other")])
    selected, digest = collect_tar(path, {"run/a.npy"}, file_sha(path))
    assert selected == {"run/a.npy": ("outer/run/a.npy", b"saved")}
    assert digest == file_sha(path)
    with pytest.raises(ValueError, match="SHA mismatch"):
        collect_tar(path, {"run/a.npy"}, "0" * 64)
    with pytest.raises(ValueError, match="missing"):
        collect_tar(path, {"run/missing.npy"}, digest)


@pytest.mark.parametrize("members", [
    [("../unsafe", b"x")],
    [("a/run/a.npy", b"x"), ("b/run/a.npy", b"x")],
])
def test_unsafe_and_duplicate_archive_paths_rejected(tmp_path, members):
    path = archive(tmp_path, members)
    with pytest.raises(ValueError):
        collect_tar(path, {"run/a.npy"}, file_sha(path))


def packet_fixture():
    request = dict(image_ids=["215", "3803", "7612"], configs={}, records=[])
    sources = {"five": {}, "lime": {}}
    for method in METHODS:
        for image_id in request["image_ids"]:
            r, *data = sample(method, image_id)
            group = "lime" if method == "lime" else "five"
            r["source_group"] = group
            for field, extension, contents in zip(
                ("source_npy_suffix", "source_png_suffix", "source_sidecar_suffix"),
                ("npy", "png", "npy.provenance.json"), data,
            ):
                suffix = f"{method}/{image_id}.{extension}"
                r[field] = suffix
                sources[group][suffix] = ("original/" + suffix, contents)
            request["records"].append(r)
    return request, sources


def test_complete_packet_readback_and_no_overwrite(tmp_path):
    request, sources = packet_fixture()
    path = tmp_path / "cases.zip"
    assert build_packet(request, sources, path)["groups"] == 18
    with ZipFile(path) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert manifest["complete"] is True and manifest["packaged_groups"] == 18
        assert len(manifest["files"]) == 54
        assert z.testzip() is None
        for name, receipt in manifest["files"].items():
            assert sha(z.read(name)) == receipt["sha256"]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        build_packet(request, sources, path)
    assert path.read_bytes() == original


def test_bad_or_incomplete_packet_creates_no_output(tmp_path):
    request, sources = packet_fixture()
    path = tmp_path / "cases.zip"
    incomplete = copy.deepcopy(request)
    incomplete["records"][-1] = incomplete["records"][0]
    with pytest.raises(ValueError, match="incomplete"):
        build_packet(incomplete, sources, path)
    assert not path.exists()
    first = request["records"][0]
    sources["five"][first["source_npy_suffix"]] = ("original", b"invalid npy")
    with pytest.raises(ValueError):
        build_packet(request, sources, path)
    assert not path.exists()
