"""Record every PR-only path and verify ancestry and protected DCU/LIME files."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "29895378e930138e042fd29ad029d0f14e8b9186"
HEADS = {"14": "a2246ea66efb58815ef324cb3938162fba941c7f",
         "15": "cb1f1abb98cec0e69dc6231897f81a6ff26cda44",
         "16": "d7a4d6e8bb7862bad32ae6323a975ce72d196b51"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8").strip()


def tree(ref: str) -> dict[str, str]:
    rows = git("ls-tree", "-r", ref).splitlines()
    return {row.split("\t", 1)[1]: row.split("\t", 1)[0].split()[2] for row in rows}


def audit() -> dict:
    baseline = tree(BASE)
    sources = {number: tree(head) for number, head in HEADS.items()}
    paths = sorted(set(baseline).union(*(set(source) for source in sources.values())))
    existing = [path for path in paths if (ROOT / path).is_file()]
    hashes = subprocess.check_output(["git", "hash-object", "--stdin-paths"], cwd=ROOT,
                                     input="\n".join(existing) + "\n", encoding="utf-8").splitlines()
    current_files = dict(zip(existing, hashes, strict=True))
    inventories = {}
    for number, head in HEADS.items():
        ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", head, "HEAD"],
                                  cwd=ROOT).returncode == 0
        paths = git("diff", "--name-only", f"{BASE}...{head}").splitlines()
        rows = []
        for path in paths:
            before, incoming, final = baseline.get(path), sources[number].get(path), current_files.get(path)
            if final == incoming:
                decision = "already_in_baseline" if before == final else "integrated"
            elif final == before:
                decision = "retained_current_equivalent_or_newer"
            else:
                decision = "reconciled_in_candidate"
            rows.append(dict(path=path, decision=decision, baseline_blob=before,
                             source_blob=incoming, candidate_blob=final))
        inventories[number] = dict(head=head, ancestor_of_candidate=ancestor, files=rows,
                                   counts=dict(Counter(row["decision"] for row in rows)))
    prefixes = ("configs/", "data/", "experiments/", "models/", "analysis/", "scripts/")
    protected = [path for path in baseline
                 if path.startswith(prefixes) or path.startswith(("docs/D_", "docs/DCU_", "docs/LIME_DCU_"))]
    changes = [path for path in protected if current_files.get(path) != baseline[path]]
    passed = not changes and all(item["ancestor_of_candidate"] for item in inventories.values())
    return dict(status="passed" if passed else "failed", baseline=BASE,
                candidate_commit=git("rev-parse", "HEAD"), pull_requests=inventories,
                protected_baseline_files=len(protected), protected_changes=changes,
                note="Inventory includes every branch-only path since its merge base. Historical documents retain original run identity; README and handoff index define current status.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("analysis/m5_integration_20261026.json"))
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "protected_baseline_files", "protected_changes")}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
