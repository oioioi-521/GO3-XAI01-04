"""Audit the October 26 data-only deliveries without executing packaged code."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd
from PIL import Image
import yaml

from .package_dcu_cases import ARCHIVES, check_map, require
from .validate_dcu_team_handoff import EXPECTED_SHA, ROOT, object_hash

CASE_SHA = "a46fa18f447e104fc707451de4a03e8b2c2eb56679c15a675f6cc5aaacde9eb3"
ANALYSIS_SHA = "a1ae2e245e3a96fd2790c4d61a3bbb8c61c5f25bbc5ee2d06f10436f19c2894a"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_bundle(path, expected_sha, root):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected_sha, "delivery ZIP identity changed")
    with ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP entries")
        require(all(n.startswith(root) and ".." not in PurePosixPath(n).parts
                    and "\\" not in n and ":" not in n for n in names), "unsafe ZIP names")
        require(sum(i.file_size for i in z.infolist()) < 512_000_000
                and all(i.file_size < 100_000_000 for i in z.infolist()), "oversized delivery")
        require(z.testzip() is None, "ZIP CRC failure")
        manifest = json.loads(z.read(root + "manifest.json"))["files"]
        require(set(names) == {root + n for n in manifest} | {root + "manifest.json", root + "SHA256SUMS"},
                "unlisted/missing delivery member")
        for name, spec in manifest.items():
            data = z.read(root + name)
            require(len(data) == spec["bytes"] and sha(data) == spec["sha256"], "manifest payload mismatch: " + name)
        sums = {}
        for line in z.read(root + "SHA256SUMS").decode().splitlines():
            digest, name = line.split(None, 1)
            name = name.lstrip(" *")
            require(name not in sums, "duplicate checksum entry")
            sums[name] = digest
            require(sha(z.read(root + name)) == digest, "checksum payload mismatch: " + name)
        require(set(sums) == set(manifest) | {"manifest.json"}, "checksum coverage")
        return dict(zip_sha256=expected_sha, bytes=len(raw), members=len(names),
                    manifest_files=len(manifest), checksum_entries=len(sums))


def audit(analysis_path, case_path, project):
    analysis_root, case_root = "dcu_analysis_handoff_20261026/", "dcu_cases_voc_resnet50_215_3803_7612/"
    analysis = audit_bundle(analysis_path, ANALYSIS_SHA, analysis_root)
    cases = audit_bundle(case_path, CASE_SHA, case_root)
    request = json.loads((project / "analysis/case_request_20261026.json").read_text(encoding="utf-8"))
    expected = {(r["method"], r["image_id"]): r for r in request["records"]}
    require(request["source_handoff_sha256"] == EXPECTED_SHA and len(expected) == 18, "frozen case request")
    with ZipFile(analysis_path) as a, ZipFile(case_path) as c:
        original = a.read(analysis_root + "archives/dcu_team_handoff_20261014.zip")
        require(sha(original) == EXPECTED_SHA, "nested original handoff identity")
        with ZipFile(io.BytesIO(original)) as old:
            for name in old.namelist():
                require(old.read(name) == a.read(analysis_root + "analysis_input/" + name[len(ROOT):]),
                        "analysis_input changed from original: " + name)
            analysis["identical_original_members"] = len(old.namelist())
            for name, count in (("per_image", 72), ("stability_trace", 90), ("method_predictions", 18),
                                ("predictions", 3), ("cohorts", 3)):
                def read(z, path):
                    return pd.read_csv(io.BytesIO(z.read(path)), dtype=str, keep_default_na=False)
                given = read(c, case_root + "case_tables/" + name + ".csv")
                full = read(old, ROOT + "data/" + name + ".csv")
                subset = full[full.dataset.eq("voc") & full.model.eq("resnet50")
                              & full.image_id.isin(request["image_ids"])]
                require(set(given.columns) == set(full.columns) and len(given) == len(subset) == count,
                        "case table coverage: " + name)
                cols = list(full.columns)
                require(given[cols].sort_values(cols).reset_index(drop=True).equals(
                    subset[cols].sort_values(cols).reset_index(drop=True)), "case table values: " + name)
        index = json.loads(c.read(case_root + "indexes/maps_index.json"))
        require(len(index) == 18 and {(r["method"], r["image_id"]) for r in index} == set(expected),
                "18 unique case groups required")
        source_archives = json.loads(c.read(case_root + "evidence/source_archives.json"))
        require(all(source_archives[g]["sha256"] == h for g, (_, h) in ARCHIVES.items()), "source archive index")
        for r in index:
            e = expected[r["method"], r["image_id"]]
            require(int(r["target"]) == e["target"], "case metadata target")
            for field in ("dataset", "model", "split", "checkpoint_sha256", "base_config_hash", "prediction_context_hash"):
                require(r[field] == e[field], "case identity: " + field)
            check_map(e, *[c.read(case_root + r[k]) for k in ("npy_path", "png_path", "provenance_path")])
            for stem in ("npy", "png", "provenance", "config", "protocol", "stability_config", "image"):
                require(sha(c.read(case_root + r[stem + "_path"])) == r[stem + "_sha256"], "case asset hash: " + stem)
            config = yaml.safe_load(c.read(case_root + r["config_path"]))
            require(object_hash(config) == e["base_config_hash"], "case frozen config")
            require(config == yaml.safe_load(request["configs"][e["config_member"]]), "case config contents")
            with Image.open(io.BytesIO(c.read(case_root + r["image_path"]))) as image:
                image.verify()
        cases.update(verified_groups=18, npy=18, png=18, original_images=3,
                     metadata_targets={r["image_id"]: r["target"] for r in request["records"]},
                     selected_tables_identical_to_frozen_input=True)
    return dict(schema_version=1, date="2026-10-27", status="verified_received", analysis=analysis, cases=cases,
                original_handoff_sha256=EXPECTED_SHA,
                checks="CRC; exact manifest and SHA256SUMS coverage; nested handoff and all 722 original members byte-identical; case table subsets; all 18 original map/PNG/sidecar identities, target/config/context/hash bindings; source photographs",
                limits="No packaged code executed or attribution rerun. Source archive hashes are cross-checked declarations, not rehashed full archives. Three IDs remain B/E candidates; this is not their final acceptance or independent C reproduction.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.analysis, args.cases, args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Verified: analysis input unchanged; 18/18 case groups received; receipt " + str(args.output))


if __name__ == "__main__":
    main()
