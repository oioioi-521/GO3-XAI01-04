"""Read-only D audit of E's fixed PR snapshot against received case files.

The snapshot is extracted under ignored results/, never checked out over E's
branch. Does not run code from the received ZIP or perform inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

from .validate_dcu_delivery import CASE_SHA
from .validate_dcu_team_handoff import EXPECTED_SHA

HEAD = "755edca74c88ca74ce6384baa75f31bc396f7ad3"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def audit(snapshot, case_zip, handoff, restore=False):
    assert "results" in snapshot.parts, "review extraction must stay under ignored results"
    assert sha(case_zip.read_bytes()) == CASE_SHA
    assert sha(handoff.read_bytes()) == EXPECTED_SHA
    demo = snapshot / "demo"
    listing = json.loads((demo/"cases.json").read_text(encoding="utf-8"))
    assert listing["source_handoff_zip"]["sha256"] == EXPECTED_SHA
    with ZipFile(handoff) as z:
        member = next(n for n in z.namelist() if n.endswith("/data/per_image_model_correct.csv"))
        values = {(r["image_id"],r["method"],r["metric"]):float(r["value"]) for r in
                  csv.DictReader(io.StringIO(z.read(member).decode())) if r["dataset"] == "voc" and r["model"] == "resnet50"}
    expected = {"215": ("7","cat"), "3803": ("2","bird"), "7612": ("11","dog")}
    records=[]
    with ZipFile(case_zip) as z:
        root="dcu_cases_voc_resnet50_215_3803_7612/"
        for c in listing["cases"]:
            iid=c["image_id"]
            assert (c["target_class_id"],c["target_class_name"]) == expected[iid]
            assert c["dataset"] == "voc" and c["split"] == "eval" and c["model"] == "resnet50"
            assert c["model_correct"] and c["predicted_class_id"] == c["target_class_id"]
            image=z.read(root+f"dataset/voc/raw/{int(iid):06d}.jpg")
            assert sha(image) == c["input_image_sha256"]
            if restore:
                (demo/c["input_image"]).write_bytes(image)
            for m in c["methods"]:
                method=m["method"]
                prefix=root+f"maps/voc/resnet50/{method}/{iid}"
                for extension,key,pathkey in ((".png","map_png_sha256","map_png"),
                                               (".npy.provenance.json","map_provenance_sha256","map_provenance_sidecar")):
                    original=z.read(prefix+extension)
                    delivered=(demo/m[pathkey]).read_bytes()
                    assert sha(delivered) == sha(original) == m[key]
                npy=z.read(prefix+".npy")
                assert sha(npy) == m["map_npy_sha256"]
                a=np.load(io.BytesIO(npy),allow_pickle=False)
                assert a.dtype == np.float32 and a.shape == (224,224) and np.isfinite(a).all()
                for key,hashkey in (("source_config_path",None),("source_protocol_path","source_protocol_sha256"),
                                    ("source_stability_config_path","source_stability_config_sha256")):
                    original=z.read(root+m[key])
                    assert sha(original) == (m[hashkey] if hashkey else m["config"]["config_snapshot_sha256"])
                for metric in ("faithfulness_morf_auc_raw","stability_spearman","stability_valid_rate","efficiency_time_ms"):
                    np.testing.assert_allclose(m[metric],values[iid,method,metric],rtol=1e-12,atol=1e-12)
                records.append(dict(image_id=iid,method=method,png_sha256=m["map_png_sha256"],
                                    npy_sha256=m["map_npy_sha256"],sidecar_sha256=m["map_provenance_sha256"]))
    assert len(records) == 18 and len({(r["image_id"],r["method"]) for r in records}) == 18
    pptx=next(snapshot.glob("*v6.pptx"));pdf=next(snapshot.glob("*v6.pdf"))
    with ZipFile(pptx) as z:
        slides=sorted([n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")],
                      key=lambda n:int(n.split("slide")[-1][:-4]))
        texts=[" | ".join(ET.fromstring(z.read(n)).itertext()) for n in slides]
        assert len(texts) == 12 and "共同命中组" in texts[3] and "p_adj=.837" in texts[3]
        media={sha(z.read(n)) for n in z.namelist() if n.startswith("ppt/media/")}
        profile=Path("analysis/dcu_20261028/method_model_profiles.png")
        assert sha(profile.read_bytes()) in media
    return dict(pr="https://github.com/oioioi-521/GO3-XAI01-04/pull/26",reviewed_head=HEAD,
        scope="D local static/content/source review; not independent-device rehearsal or teacher/final acceptance",
        pptx_sha256=sha(pptx.read_bytes()),pdf_sha256=sha(pdf.read_bytes()),pptx_slides=12,
        case_zip_sha256=CASE_SHA,handoff_sha256=EXPECTED_SHA,
        byte_identical_pngs=18,byte_identical_sidecars=18,float32_npy_hashes_and_arrays_verified=18,
        frozen_metric_values_matched=72,config_protocol_stability_bindings_checked=54,
        source_images_hash_verified=3,old_D_profile_image_byte_identical=True,
        source_images_restored_only_under_ignored_snapshot=restore,
        records=records,review_code_sha256=sha(Path(__file__).read_bytes()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_zip",type=Path)
    parser.add_argument("--snapshot",type=Path,default=Path("results/e_review_20261029/deliverables/e_m5"))
    parser.add_argument("--handoff",type=Path,default=Path("results/dcu_delivery_20261027/dcu_team_handoff_20261014.zip"))
    parser.add_argument("--restore-images",action="store_true")
    args=parser.parse_args()
    result=audit(args.snapshot,args.case_zip,args.handoff,args.restore_images)
    Path("analysis/dcu_20261029/e_v6_review_receipt.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("E snapshot audit passed: 18 PNG, 18 sidecars, 18 float32 NPY, 72 metrics, 54 config bindings, 3 inputs; 12 slides")


if __name__ == "__main__":
    main()
