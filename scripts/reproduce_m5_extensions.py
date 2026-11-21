"""Independently rerun D's October 28/29 analyses from the frozen DCU ZIP."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ZIP = "14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc"


def sha256(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    handoff = args.handoff.resolve(strict=True)
    if sha256(handoff) != EXPECTED_ZIP:
        raise ValueError("Frozen ZIP SHA-256 mismatch")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("Use a new output directory under this checkout's results/")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="running", input_sha256=EXPECTED_ZIP,
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        working_tree_at_start=subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        python=platform.python_version(),
        dependencies={n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "PyYAML", "matplotlib", "pytest")},
        commands=[], comparisons={}, artifacts={},
        scope="C-executed CPU statistics from frozen observations, including full published resampling budgets; no GPU attribution, teacher acceptance, independent-device Demo or group rehearsal.")
    started = time.monotonic()
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "matplotlib-cache"), PYTHONUTF8="1")

    def run(label, arguments):
        command = [sys.executable, *arguments]
        print(label, flush=True)
        log_path = output / (label + ".log")
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["commands"].append(dict(label=label, arguments=arguments, exit_code=result.returncode, log_sha256=sha256(log_path)))
        if result.returncode:
            raise RuntimeError(f"{label} failed: see {log_path}")

    try:
        rm = output / "repeated"
        mixed = output / "mixed"
        rm.mkdir()
        mixed.mkdir()
        rm_json = rm / "repeated_measures.json"
        mixed_json = mixed / "mixed_factorial.json"
        run("01_repeated", ["-m", "analysis.dcu_repeated_measures", str(handoff), "--output", str(rm_json), "--bootstrap", "10000"])
        run("02_mixed", ["-m", "analysis.dcu_mixed_factorial", str(handoff), "--output", str(mixed_json), "--bootstrap", "9999", "--ci-resamples", "10000"])
        run("03_repeated_report", ["-m", "analysis.report_dcu_repeated_measures", "--input", str(rm_json), "--report", str(rm / "report.md")])
        # The mixed report CLI hardcodes a tracked docs/ output. Call its pure
        # authoring functions with explicit destinations to preserve that file.
        code = ("import json; from pathlib import Path; from analysis.report_dcu_mixed_factorial import figures,report,appendix; "
                "p=Path(__import__('sys').argv[1]); d=json.loads(p.read_text(encoding='utf-8')); figures(d,p.parent); "
                "(p.parent/'report.md').write_text(report(d),encoding='utf-8',newline='\\n'); "
                "(p.parent/'mixed_factorial_tables.md').write_text(appendix(d),encoding='utf-8',newline='\\n')")
        run("04_mixed_report", ["-c", code, str(mixed_json)])
        run("05_mixed_independent_formula_check", ["-m", "analysis.verify_dcu_mixed_factorial", str(handoff), "--result", str(mixed_json)])
        for label, rebuilt, published in (
            ("repeated", rm_json, ROOT / "analysis/dcu_20261028/repeated_measures.json"),
            ("mixed", mixed_json, ROOT / "analysis/dcu_20261029/mixed_factorial.json"),
        ):
            actual = json.loads(rebuilt.read_text(encoding="utf-8"))
            expected = json.loads(published.read_text(encoding="utf-8"))
            receipt["comparisons"][label] = {
                "sections": {k: dict(exact_match=actual.get(k) == v,
                    rows=len(v) if isinstance(v, list) else None)
                    for k, v in expected.items() if k != "environment"},
                "unexpected_sections": sorted(set(actual) - set(expected)),
                "published_environment": expected.get("environment"),
                "actual_environment": actual.get("environment"),
                "published_sha256": sha256(published),
            }
        files = {
            "repeated/method_model_profiles.png": "analysis/dcu_20261028/method_model_profiles.png",
            "repeated/method_model_interaction.png": "analysis/dcu_20261028/method_model_interaction.png",
            "repeated/direct_did_tables.md": "analysis/dcu_20261028/direct_did_tables.md",
            "repeated/report.md": "docs/D_M5_REPEATED_MEASURES_20261028.md",
            "mixed/observed_task_differences.png": "analysis/dcu_20261029/observed_task_differences.png",
            "mixed/mixed_factorial_tables.md": "analysis/dcu_20261029/mixed_factorial_tables.md",
            "mixed/report.md": "docs/D_M5_MIXED_FACTORIAL_20261029.md",
        }
        receipt["file_comparisons"] = {}
        for name, reference in files.items():
            a, b = output / name, ROOT / reference
            text = a.suffix == ".md"
            receipt["file_comparisons"][name] = dict(
                exact_match=(a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")) if text else a.read_bytes() == b.read_bytes(),
                comparison="text with universal newlines" if text else "raw bytes",
                published_sha256=sha256(b), rebuilt_sha256=sha256(a))
        passed = (all(v["exact_match"] for s in receipt["comparisons"].values() for v in s["sections"].values())
                  and not any(s["unexpected_sections"] for s in receipt["comparisons"].values())
                  and all(v["exact_match"] for v in receipt["file_comparisons"].values()))
        receipt["status"] = "passed" if passed else "differences_require_review"
    except Exception as error:
        receipt.update(status="failed", error=str(error))
    finally:
        receipt["duration_seconds"] = round(time.monotonic() - started, 3)
        receipt["artifacts"] = {str(p.relative_to(output)).replace("\\", "/"): sha256(p)
            for p in output.rglob("*") if p.is_file() and "matplotlib-cache" not in p.parts and p.name != "receipt.json"}
        (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(status=receipt["status"], duration_seconds=receipt["duration_seconds"])), flush=True)
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
