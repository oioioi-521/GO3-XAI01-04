"""Archive-integrity fixtures are synthetic and contain no experiment outputs."""
import hashlib
import json
from zipfile import ZipFile

import pytest

from analysis.validate_dcu_delivery import audit_bundle


def bundle(tmp_path, *, corrupt=False, extra=False):
    path = tmp_path / "delivery.zip"
    payload = b"frozen data\n"
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.dumps({"files": {"data.txt": {"bytes": len(payload), "sha256": digest}}}).encode()
    checksum = f"{digest}  data.txt\n{hashlib.sha256(manifest).hexdigest()}  manifest.json\n"
    with ZipFile(path, "w") as z:
        z.writestr("delivery/data.txt", b"changed" if corrupt else payload)
        z.writestr("delivery/manifest.json", manifest)
        z.writestr("delivery/SHA256SUMS", checksum)
        if extra:
            z.writestr("delivery/unlisted.py", "raise Exception('never execute')")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_members_and_checksums_verified(tmp_path):
    path, digest = bundle(tmp_path)
    result = audit_bundle(path, digest, "delivery/")
    assert result["manifest_files"] == 1 and result["checksum_entries"] == 2


def test_changed_zip_identity_rejected(tmp_path):
    path, _ = bundle(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        audit_bundle(path, "0" * 64, "delivery/")


@pytest.mark.parametrize("options", [{"corrupt": True}, {"extra": True}])
def test_internal_tamper_or_unlisted_payload_rejected(tmp_path, options):
    path, digest = bundle(tmp_path, **options)
    with pytest.raises(ValueError):
        audit_bundle(path, digest, "delivery/")
