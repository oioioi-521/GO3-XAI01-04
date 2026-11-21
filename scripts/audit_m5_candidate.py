"""Audit fixed A/B/D/E source ancestry and file identity in C's candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "29895378e930138e042fd29ad029d0f14e8b9186"
SOURCES = {
    "C_gate_fix": "eb275ed535a1ba660608a9606fd3847f83460405",
    "A_report_v4": "138b865e161d4808ab696864ce68b8daf5edafd5",
    "B_cases_review": "73310e85614193e2e40d456a007ae16fc725fd8b",
    "D_statistics_closeout": "468a686de1700e89456a9dcddeb4b2cf175afcf9",
    "E_display_v7": "e2003bf78b8f15cbf65294580ad453d67d7da2e1",
}
AUTHORIZED_D_UPDATES = {
    "analysis/report_dcu_statistics.py", "docs/DCU_STATISTICAL_ANALYSIS_20261015.md",
    "docs/D_DELIVERY_INDEX.md",
}
C_DELIVERY_DOCS = {"README.md", "docs/C_M5_DELIVERY.md", "docs/C_M5_INTEGRATION.md",
                   "docs/C_M5_REPRODUCTION.md", "docs/C_M5_PR_DESCRIPTION.md"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8").strip()


def tree(ref):
    rows = subprocess.check_output(["git", "ls-tree", "-rz", ref], cwd=ROOT).split(b"\0")
    return {path.decode("utf-8"): meta.decode("ascii").split()[2]
            for row in rows if row for meta, path in [row.split(b"\t", 1)]}


def audit():
    baseline, current = tree(BASE), tree("HEAD")
    sources = {label: tree(sha) for label, sha in SOURCES.items()}
    protected = [p for p in baseline if p.startswith(("configs/", "data/", "experiments/", "models/",
        "analysis/", "scripts/", "docs/D_", "docs/DCU_", "docs/LIME_DCU_"))]
    changed = [p for p in protected if baseline[p] != current.get(p)]
    unexpected = [p for p in changed if p not in AUTHORIZED_D_UPDATES or
                  current.get(p) != sources["D_statistics_closeout"].get(p)]
    inventories = {}
    for label, sha in SOURCES.items():
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=ROOT).returncode == 0
        paths = sorted(p for p, blob in sources[label].items() if baseline.get(p) != blob)
        rows = []
        for path in paths:
            final = current.get(path)
            matches = [name for name, files in sources.items() if final is not None and final == files.get(path)]
            approved = bool(matches) or path == ".gitattributes" or path in C_DELIVERY_DOCS
            rows.append(dict(path=path, source_blob=sources[label][path], candidate_blob=final,
                matches_fixed_sources=matches, reconciled_attributes=path == ".gitattributes",
                C_maintained_delivery_document=path in C_DELIVERY_DOCS, passed=approved))
        inventories[label] = dict(sha=sha, ancestor=ancestor, files=rows)
    # The first integration preserved these histories; retain that requirement.
    old = {str(n): subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=ROOT).returncode == 0
           for n, sha in ((14, "a2246ea66efb58815ef324cb3938162fba941c7f"),
                          (15, "cb1f1abb98cec0e69dc6231897f81a6ff26cda44"),
                          (16, "d7a4d6e8bb7862bad32ae6323a975ce72d196b51"))}
    passed = (not unexpected and all(old.values()) and
              all(s["ancestor"] and all(r["passed"] for r in s["files"]) for s in inventories.values()))
    return dict(status="passed" if passed else "failed", baseline=BASE, candidate_commit=git("rev-parse", "HEAD"),
        protected_files=len(protected), unchanged_protected_files=len(protected)-len(changed),
        authorized_D_report_updates=changed, unexpected_protected_changes=unexpected,
        historical_PR_ancestry=old, sources=inventories,
        scope="Committed Git blob and ancestry audit. C's attributes union and new delivery tools are separate; no experiment provenance or human signoff inferred.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("status", "protected_files", "unchanged_protected_files", "authorized_D_report_updates")}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
