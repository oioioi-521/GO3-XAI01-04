"""Select existing VOC/ResNet50 case maps. Never generate or repair attribution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
import yaml

from .validate_dcu_team_handoff import ROOT, METHODS, object_hash, validate

ARCHIVES = {
    "five": ("dcu_full_five_20261013_complete.tar.gz", "852eb7c8c4eb74124b6c599d2fd54bf1f067f9c0579ce9a0542dfada040dbede"),
    "lime": ("lime_vectorized_formal_complete_20261014_v2.tar.gz", "38510f8eeda54ae38ee77acaf04580666779954ff7565b59d9fbd58438fe1cd3"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(handoff, project, ids):
    ids = [str(i) for i in ids]
    require(ids and len(ids) == len(set(ids)) and all(i.isdecimal() for i in ids), "unique numeric image IDs required")
    _, receipt, _ = validate(Path(handoff), Path(project))
    with ZipFile(handoff) as z:
        def rows(name):
            return list(csv.DictReader(io.StringIO(z.read(ROOT + name).decode("utf-8-sig"))))
        metadata = {r["image_id"]: r for r in rows("source/five/release/data/metadata.csv")
                    if r["dataset"] == "voc" and r["split"] == "eval"}
        require(set(ids) <= metadata.keys(), "image outside frozen VOC eval")
        predictions = {r["image_id"]: r for r in rows("data/predictions.csv")
                       if r["dataset"] == "voc" and r["model"] == "resnet50"}
        contexts = {(r["method"], r["image_id"]): r["config_hash"] for r in rows("data/method_predictions.csv")
                    if r["dataset"] == "voc" and r["model"] == "resnet50"}
        configs, records = {}, []
        for method in METHODS:
            folders = ([f"configs/five/{method}-resnet50-voc"] if method != "lime" else
                       sorted({n[len(ROOT):].rsplit("/completion.json", 1)[0] for n in z.namelist()
                               if n.startswith(ROOT + "configs/lime/lime-resnet50-voc__") and n.endswith("/completion.json")}))
            for image_id in ids:
                choices = []
                for folder in folders:
                    name = next(n for n in z.namelist() if n.startswith(ROOT + folder + "/configs/") and n.endswith(".yaml"))
                    content = z.read(name)
                    config = yaml.safe_load(content)
                    if method != "lime" or image_id in config["dataset"]["image_ids"]:
                        choices.append((folder, content, config))
                require(len(choices) == 1, f"missing/duplicate source config: {method}/{image_id}")
                folder, content, config = choices[0]
                digest = object_hash(config)
                key = f"configs/{method}/{digest}_unit.yaml"
                configs[key] = content.decode("utf-8")
                shard = folder.rsplit("/", 1)[1]
                run = f"run/formal/{shard}"
                stem = f"{method}_resnet50_voc/{image_id}"
                npy = f"{run}/maps_float/{stem}.npy"
                png = f"{run}/maps/{stem}.png"
                completion = json.loads(z.read(ROOT + folder + "/completion.json"))
                hashes = completion.get("map_sha256", {})
                p = predictions[image_id]
                records.append(dict(image_id=image_id, method=method, dataset="voc", model="resnet50", split="eval",
                    target=int(float(metadata[image_id]["class_id"])), class_name=metadata[image_id]["class_name"],
                    image_path=metadata[image_id]["image_path"], predicted_class_id=int(p["predicted_class_id"]),
                    model_correct=p["correct"] == "1", checkpoint_sha256=p["checkpoint_sha256"],
                    prediction_context_hash=contexts[method, image_id], base_config_hash=digest,
                    config_member=key, source_group="lime" if method == "lime" else "five", shard=shard,
                    source_npy_suffix=npy, source_png_suffix=png, source_sidecar_suffix=npy + ".provenance.json",
                    expected_npy_sha256=hashes.get(f"maps_float/{stem}.npy"),
                    expected_sidecar_sha256=hashes.get(f"maps_float/{stem}.npy.provenance.json"), map_status="not_received"))
    return dict(schema_version=1, complete=False, requested_groups=len(records), packaged_groups=0,
                image_ids=ids, dataset="voc", model="resnet50", split="eval",
                source_handoff_sha256=receipt["zip_sha256"], records=records, configs=configs,
                required_archives={k: dict(filename=v[0], sha256=v[1]) for k, v in ARCHIVES.items()},
                limits="Request/identity list only until both map sources are supplied; no attribution is generated.")


def collect_tar(path, suffixes, expected_sha):
    """One-pass, bounded selective reads; never extract members to disk."""
    path = Path(path)
    digest = file_sha(path)
    require(digest == expected_sha, "source archive SHA mismatch")
    found = {}
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = member.name
            pure = PurePosixPath(name)
            require(not pure.is_absolute() and ".." not in pure.parts and "\\" not in name and ":" not in name,
                    "unsafe archive path")
            matching = [s for s in suffixes if name == s or name.endswith("/" + s)]
            if not matching:
                continue
            require(len(matching) == 1 and matching[0] not in found, "duplicate selected archive member")
            require(member.isfile() and member.size <= 5_000_000, "selected member is not a bounded regular file")
            contents = archive.extractfile(member).read()
            require(len(contents) == member.size, "truncated archive member")
            found[matching[0]] = (name, contents)
    require(set(found) == set(suffixes), "selected map/PNG/sidecar missing: " + str(sorted(set(suffixes)-set(found))))
    return found, digest


def check_map(record, npy, png, sidecar):
    array = np.load(io.BytesIO(npy), allow_pickle=False)
    require(isinstance(array, np.ndarray) and array.dtype == np.float32 and array.shape == (224, 224)
            and np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1, "NPY dtype/shape/range")
    with Image.open(io.BytesIO(png)) as im:
        require(im.format == "PNG" and im.size == (224, 224), "PNG format/size")
        im.verify()
    provenance = json.loads(sidecar)
    for field in ("image_id", "method", "model", "dataset", "split", "checkpoint_sha256", "base_config_hash"):
        require(provenance.get(field) == record[field], f"sidecar {field} mismatch")
    require(provenance.get("prediction_context_hash") == record["prediction_context_hash"]
            and provenance.get("map_sha256") == sha(npy), "sidecar map/prediction binding")
    for field, value in (("expected_npy_sha256", npy), ("expected_sidecar_sha256", sidecar)):
        require(record.get(field) is None or sha(value) == record[field], f"frozen completion {field} mismatch")


def build_packet(request, sources, output):
    """All groups validated before creating output. Refuse existing outputs."""
    expected = {(m, i) for m in METHODS for i in request["image_ids"]}
    actual = [(r["method"], r["image_id"]) for r in request["records"]]
    require(len(actual) == len(expected) and set(actual) == expected, "incomplete request")
    payload, index = {}, []
    for r in request["records"]:
        source = sources[r["source_group"]]
        names = [r["source_npy_suffix"], r["source_png_suffix"], r["source_sidecar_suffix"]]
        data = [source[n][1] for n in names]
        check_map(r, *data)
        files = []
        for suffix, content, extension in zip(names, data, ("npy", "png", "npy.provenance.json")):
            target = f"maps/{r['method']}/{r['image_id']}.{extension}"
            require(target not in payload, "duplicate output member")
            payload[target] = content
            files.append(dict(path=target, sha256=sha(content), bytes=len(content), source_member=source[suffix][0]))
        index.append({**r, "map_status": "verified_received", "files": files})
    for name, content in request["configs"].items():
        require(object_hash(yaml.safe_load(content)) == PurePosixPath(name).name.split("_")[0], "config content hash")
        payload[name] = content.encode("utf-8")
    manifest = {k: v for k, v in request.items() if k not in ("records", "configs", "limits")}
    manifest.update(complete=True, packaged_groups=len(index), records=index,
                    limits="Saved base maps only; target comes from frozen metadata, not a field fabricated in original sidecars. No rerun or proof of historical generation.",
                    files={n: dict(sha256=sha(b), bytes=len(b)) for n, b in payload.items()})
    payload["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    payload["README.md"] = ("# VOC eval / ResNet50 cases\n\n"
        f"{len(index)} groups: original float32 NPY, corresponding saved PNG, original sidecars and config snapshots.\n"
        "Verify manifest.json file hashes and dataset/image_id/metadata target/checkpoint/config before use.\n"
        "PNG is a saved quantized attribution map, not a newly rendered overlay. Use NPY for numerical work.\n"
        "Original photographs are not included. No attribution was rerun.\n").encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle, ZipFile(handle, "w", ZIP_DEFLATED) as z:
        for name, content in sorted(payload.items()):
            z.writestr(name, content)
    with ZipFile(output) as z:
        require(z.testzip() is None and all(sha(z.read(n)) == sha(b) for n, b in payload.items()), "packet readback mismatch")
    return dict(path=str(output), sha256=file_sha(output), groups=len(index), files=len(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--image-ids", nargs="+", default=["215", "3803", "7612"])
    parser.add_argument("--request-only", type=Path)
    parser.add_argument("--five-archive", type=Path)
    parser.add_argument("--lime-archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    request = prepare(args.handoff, args.project, args.image_ids)
    if args.request_only:
        require(not (args.output or args.five_archive or args.lime_archive), "request-only does not package maps")
        args.request_only.parent.mkdir(parents=True, exist_ok=True)
        with args.request_only.open("x", encoding="utf-8") as f:
            json.dump(request, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Identity request prepared: {request['requested_groups']} groups; 0 maps received")
        return
    require(args.output and args.five_archive and args.lime_archive, "both original archives and --output are required")
    sources = {}
    for group, path in (("five", args.five_archive), ("lime", args.lime_archive)):
        suffixes = {r[field] for r in request["records"] if r["source_group"] == group
                    for field in ("source_npy_suffix", "source_png_suffix", "source_sidecar_suffix")}
        sources[group], digest = collect_tar(path, suffixes, ARCHIVES[group][1])
        request["required_archives"][group]["verified_sha256"] = digest
    print(json.dumps(build_packet(request, sources, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
