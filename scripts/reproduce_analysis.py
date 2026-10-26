"""Rebuild the frozen DCU analysis with the current interpreter, without torch."""
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

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ZIP = "14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc"
TABLES = ("summaries", "method_omnibus", "method_pairs", "pareto",
          "correlations", "model_omnibus", "model_pairs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_tables(expected: dict, actual: dict) -> dict:
    """Exact semantic equality; environment metadata and JSON EOL are separate."""
    return {name: {"rows": len(actual[name]), "exact_match": actual[name] == expected[name]}
            for name in TABLES}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    handoff = args.handoff.resolve(strict=True)
    if sha256(handoff) != EXPECTED_ZIP:
        raise ValueError("ZIP SHA-256 does not match the frozen M5 analysis input")
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "results"):
        raise ValueError("output must be a new directory under this checkout's results/")
    output.mkdir(parents=True, exist_ok=False)
    statistics = output / "statistics.json"
    report = output / "statistical_report.md"
    commands = [
        [sys.executable, "-m", "analysis.dcu_statistics", str(handoff), "--output", str(statistics)],
        [sys.executable, "-m", "analysis.plot_dcu_statistics", "--input", str(statistics),
         "--output-dir", str(output / "figures")],
        [sys.executable, "-m", "analysis.report_dcu_statistics", "--input", str(statistics),
         "--report", str(report)],
    ]
    environment = dict(os.environ, MPLCONFIGDIR=str(output / "matplotlib-cache"),
                       MPLBACKEND="Agg", PYTHONUTF8="1")
    for index, command in enumerate(commands, 1):
        print(f"step {index}/3: {command[2]}", flush=True)
        with (output / f"step_{index}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    actual = json.loads(statistics.read_text(encoding="utf-8"))
    published_path = ROOT / "analysis/dcu_20261015/statistics.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))
    comparison = compare_tables(published, actual)
    text_checks = {}
    for label, source, rebuilt in (
        ("report", ROOT / "docs/DCU_STATISTICAL_ANALYSIS_20261015.md", report),
        ("paired_quality_tables", published_path.with_name("paired_quality_tables.md"),
         output / "paired_quality_tables.md"),
    ):
        text_checks[label] = source.read_text(encoding="utf-8") == rebuilt.read_text(encoding="utf-8")
    figures = sorted((output / "figures").glob("*.png"))
    source_checks = {key: actual[key] == published[key]
                     for key in ("analysis_code_sha256", "validator_code_sha256")}
    passed = (all(row["exact_match"] for row in comparison.values())
              and actual["protocol"] == published["protocol"]
              and actual["receipt"] == published["receipt"]
              and all(source_checks.values())
              and all(text_checks.values()) and len(figures) == 4)
    receipt = {
        "status": "passed" if passed else "differences_require_review",
        "input_sha256": EXPECTED_ZIP,
        "code_commit": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain"),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "PyYAML", "matplotlib", "pytest")},
        "torch_installed": any(d.metadata["Name"].lower() == "torch"
                               for d in importlib.metadata.distributions()),
        "tables": comparison,
        "protocol_exact_match": actual["protocol"] == published["protocol"],
        "text_exact_match_after_eol_normalization": text_checks,
        "source_hash_matches_published": source_checks,
        "validation_exact_match": actual["receipt"] == published["receipt"],
        "validation": actual["receipt"],
        "commands": [["python", *[part.replace(str(handoff), "<handoff.zip>")
                       .replace(str(output), "<output-dir>") for part in c[1:]]]
                     for c in commands],
        "artifacts": {str(p.relative_to(output)): sha256(p) for p in
                      [statistics, report, output / "paired_quality_tables.md", *figures]},
        "figure_byte_matches_published": {
            p.name: sha256(p) == sha256(published_path.parent / p.name) for p in figures},
        "scope": "Independent table analysis and figure regeneration; no archive code execution, GPU attribution, or unsaved noise-map replication.",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "tables": comparison}, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
