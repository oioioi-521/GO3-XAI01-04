"""Read-only D review of the fixed E notes correction and A status update.

Requires the four commits below in the local Git object database. Run from the
repository root with ``python -m analysis.review_notes_closeout_20261120``.
No teammate code is executed and no experiment or previous receipt is changed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
OLD_E = "d00553f9ca7fc41208e24296619c1406a339d8da"
NEW_E = "e2003bf78b8f15cbf65294580ad453d67d7da2e1"
OLD_A = "3cfd4a8250b7f820e241bbcae3df93aa374f6c8a"
NEW_A = "138b865e161d4808ab696864ce68b8daf5edafd5"
EXPECTED_NOTES = [f"ppt/notesSlides/notesSlide{i}.xml" for i in (1, 6, 11, 12)]


def git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT
    )


def blob(ref: str, path: str) -> bytes:
    return git("show", f"{ref}:{path}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def tree(ref: str, prefix: str) -> dict[str, str]:
    entries = git("ls-tree", "-rz", ref, "--", prefix).split(b"\0")
    return {
        path.decode("utf-8"): meta.decode("ascii")
        for entry in entries if entry
        for meta, path in [entry.split(b"\t", 1)]
    }


def package(data: bytes) -> dict[str, bytes]:
    with ZipFile(io.BytesIO(data)) as archive:
        require(archive.testzip() is None, "PPTX ZIP CRC failed")
        require(len(archive.namelist()) == len(set(archive.namelist())), "Duplicate ZIP entries")
        entries = {name: archive.read(name) for name in archive.namelist()}
    for name, content in entries.items():
        if name.endswith((".xml", ".rels")):
            ET.fromstring(content)
    return entries


def xml_text(entries: dict[str, bytes], name: str) -> str:
    return " | ".join(ET.fromstring(entries[name]).itertext())


def main() -> None:
    prefix = "deliverables/e_m5/"
    names = list(tree(NEW_E, prefix))
    ppt = next(name for name in names if name.endswith("_v7.pptx"))
    pdf = next(name for name in names if name.endswith("_v7.pdf"))
    old_bytes, new_bytes = blob(OLD_E, ppt), blob(NEW_E, ppt)
    old, new = package(old_bytes), package(new_bytes)
    require(old.keys() == new.keys(), "PPTX package entry set changed")
    changed = sorted(name for name in new if new[name] != old[name])
    require(changed == sorted(EXPECTED_NOTES), f"Unexpected PPTX changes: {changed}")
    slide_names = [name for name in new if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
    note_names = [name for name in new if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)]
    require(len(slide_names) == len(note_names) == 12, "Expected 12 slides and 12 notes")
    texts = {i: xml_text(new, f"ppt/notesSlides/notesSlide{i}.xml") for i in range(1, 13)}
    for i, text in texts.items():
        require("v7 尚未 commit/push" not in text, f"Stale publication status in note {i}")
    for i in (1, 12):
        require("v7" in texts[i] and "已进入" in texts[i] and "#26" in texts[i], f"Release status missing in note {i}")
    # Manual reading of the full four notes supplements these targeted checks.
    require("ImageNet" in texts[6] and "均值" in texts[6] and "秩" in texts[6], "P6 task/estimand wording missing")
    require("确认" in texts[11], "P11 case confirmation missing")
    require("独立" in texts[12] and "计时" in texts[12], "P12 pending acceptance boundary missing")
    for token in (".121215", ".1148", "9,999", ".0014", "IID", ".8374", "不是 p=0"):
        require(token in texts[4] + texts[12], f"Missing retained statistical boundary: {token}")
    pdf_bytes = blob(NEW_E, pdf)
    require(pdf_bytes == blob(OLD_E, pdf), "PDF changed")
    demo = tree(NEW_E, prefix + "demo/")
    require(demo == tree(OLD_E, prefix + "demo/"), "Tracked Demo files/modes changed")
    claimed = json.loads(blob(NEW_E, prefix + "E_V7_NOTES_REVIEW_20261104.json"))
    require(digest(old_bytes) == claimed["source_pptx_sha256"], "Old PPTX identity mismatch")
    require(digest(new_bytes) == claimed["pptx_sha256"], "New PPTX identity mismatch")
    require(digest(pdf_bytes) == claimed["pdf_sha256"], "PDF identity mismatch")
    require(changed == sorted(claimed["changed_package_entries"]), "A receipt package changes mismatch")
    report_path = "docs/FINAL_REPORT_DRAFT.md"
    report_old = blob(OLD_A, report_path).decode("utf-8")
    report_new = blob(NEW_A, report_path).decode("utf-8")
    tables = lambda text: [line for line in text.splitlines() if line.startswith("|")]
    require(tables(report_old) == tables(report_new), "Report tables changed")
    changed_report_lines = set(report_new.splitlines()) - set(report_old.splitlines())
    require(len(changed_report_lines) == 4, "Report edit scope requires fresh review")
    require(NEW_E in report_new, "Report does not point to corrected E version")
    for path in ("analysis/dcu_20261015/", "analysis/dcu_20261028/", "analysis/dcu_20261029/", "analysis/plot_final_pareto_3d.py"):
        require(tree(OLD_A, path) == tree(NEW_A, path), f"Previously reviewed analysis changed: {path}")
    output = ROOT / "analysis/dcu_20261120"
    output.mkdir(exist_ok=True)
    # Raw artifacts and extracted notes remain ignored review material.
    snapshot = ROOT / "results/notes_closeout_20261120"
    snapshot.mkdir(exist_ok=True)
    for i in (1, 6, 11, 12):
        (snapshot / f"note{i}.txt").write_text(texts[i] + "\n", encoding="utf-8")
    receipt = {
        "date": "2026-11-20", "actor": "D", "status": "passed",
        "old_E_head": OLD_E, "E_head": NEW_E, "old_A_head": OLD_A, "A_head": NEW_A,
        "pptx_path": ppt, "old_pptx_sha256": digest(old_bytes),
        "pptx_sha256": digest(new_bytes), "pdf_sha256": digest(pdf_bytes),
        "changed_package_entries": changed, "unchanged_package_entries": len(new) - len(changed),
        "zip_crc_and_all_xml_parse": True, "slides": len(slide_names), "notes": len(note_names),
        "pdf_bytes_unchanged": True, "tracked_demo_files_unchanged": len(demo),
        "report_table_lines_unchanged": len(tables(report_new)),
        "report_changed_lines": len(changed_report_lines),
        "report_sha256": digest(blob(NEW_A, report_path)),
        "analysis_and_pareto_export_source_unchanged": True,
        "A_correction_receipt_matches": True, "stale_P12_publication_status_removed": True,
        "scope": "Fixed-ref byte/XML/text review only. No new rendering, PowerPoint application, numerical bootstrap rerun, independent-device Demo, C/B/E signoff or timed rehearsal.",
        "review_code_sha256": digest(Path(__file__).read_bytes()),
    }
    (output / "notes_closeout_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
