"""Check local Demo assets, frozen metric bindings, and localhost delivery."""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import platform
import threading
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "deliverables/e_m5/demo"
EXPECTED = "14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    assert digest(args.handoff.read_bytes()) == EXPECTED
    with ZipFile(args.handoff) as archive:
        member = next(n for n in archive.namelist() if n.endswith("/data/per_image_model_correct.csv"))
        rows = list(csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig"))))
    cases = json.loads((DEMO / "cases.json").read_text(encoding="utf-8"))["cases"]
    assert [str(c["image_id"]) for c in cases] == ["215", "3803", "7612"]
    assets = ["index.html", "cases.json"]
    checked = []
    for case in cases:
        assert case["dataset"] == "voc" and case["model"] == "resnet50" and case["split"] == "eval"
        assert digest((DEMO / case["input_image"]).read_bytes()) == case["input_image_sha256"]
        assets.append(case["input_image"])
        assert {m["method"] for m in case["methods"]} == {"ig", "gradcam", "occlusion", "rise", "kernelshap", "lime"}
        for method in case["methods"]:
            for field, key in (("map_png", "map_png_sha256"), ("map_provenance_sidecar", "map_provenance_sha256")):
                assert digest((DEMO / method[field]).read_bytes()) == method[key]
                assets.append(method[field])
            for metric in ("faithfulness_morf_auc_raw", "stability_spearman", "stability_valid_rate", "efficiency_time_ms"):
                matches = [r for r in rows if r["image_id"] == case["image_id"] and r["model"] == "resnet50"
                           and r["dataset"] == "voc" and r["method"] == method["method"] and r["metric"] == metric]
                assert len(matches) == 1, (case["image_id"], method["method"], metric)
                assert float(matches[0]["value"]) == method[metric]
        checked.append(dict(image_id=case["image_id"], maps=6, sidecars=6, original_sha256=case["input_image_sha256"], frozen_metric_values=24))

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(QuietHandler, directory=str(DEMO))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for asset in assets:
            with urlopen(f"http://127.0.0.1:{server.server_port}/" + asset, timeout=5) as response:
                assert response.status == 200 and response.read() == (DEMO / asset).read_bytes()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    receipt = dict(date="2026-11-21", actor="C", status="passed_local_assets_and_http", platform=platform.platform(),
        input_sha256=EXPECTED, cases=checked, frozen_metric_values=72, localhost_get_checks=len(assets),
        sidecar_bytes="LF checkout required by .gitattributes; raw SHA checks, no normalized hashing",
        browser_interaction_verified=False, independent_device_verified=False,
        browser_limitation="Available browser control failed to start: CreateProcessWithLogonW error 1920.",
        pending="Second-device browser: originals, each case 6/6 maps, case switching, source disclosure and static fallback; actual group rehearsal.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Verified: 3 originals, 18 PNG, 18 sidecars, 72 frozen metrics, 41 localhost GETs")


if __name__ == "__main__":
    main()
