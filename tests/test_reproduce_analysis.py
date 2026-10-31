"""Comparison must detect both changed values and missing experiment rows."""
from scripts.reproduce_analysis import TABLES, compare_tables
import json

import pytest

from scripts import reproduce_analysis


def test_comparison_detects_changed_value_and_missing_row():
    expected = {name: [{"value": 0.25}, {"value": 0.5}] for name in TABLES}
    actual = {name: [dict(row) for row in rows] for name, rows in expected.items()}
    assert all(row["exact_match"] for row in compare_tables(expected, actual).values())
    actual["summaries"][0]["value"] = 0.26
    actual["method_pairs"].pop()
    result = compare_tables(expected, actual)
    assert not result["summaries"]["exact_match"]
    assert not result["method_pairs"]["exact_match"]
    assert result["method_pairs"]["rows"] == 1
    assert result["pareto"]["exact_match"]


@pytest.mark.parametrize("changed_figure", [False, True])
def test_final_receipt_requires_four_byte_identical_figures(tmp_path, monkeypatch, changed_figure):
    root = tmp_path / "project"
    published = root / "analysis/dcu_20261015"
    published.mkdir(parents=True)
    (root / "docs").mkdir()
    expected = {name: [{"value": 0.25}] for name in TABLES}
    expected.update(protocol={"fixture": True}, receipt={"fixture": True},
                    analysis_code_sha256="fixture-analysis", validator_code_sha256="fixture-validator")
    (published / "statistics.json").write_text(json.dumps(expected))
    (published / "paired_quality_tables.md").write_text("fixture paired table\n")
    (root / "docs/DCU_STATISTICAL_ANALYSIS_20261015.md").write_text("fixture report\n")
    for i in range(4):
        (published / f"figure{i}.png").write_bytes(b"fixture PNG bytes")
    handoff = tmp_path / "handoff.zip"
    handoff.write_bytes(b"fixture input")
    output = root / "results/reproduction"

    def fake_analysis(command, **kwargs):
        if command[2] == "analysis.dcu_statistics":
            (output / "statistics.json").write_text(json.dumps(expected))
        elif command[2] == "analysis.plot_dcu_statistics":
            (output / "figures").mkdir()
            for i in range(4):
                content = b"changed PNG bytes" if changed_figure and i == 0 else b"fixture PNG bytes"
                (output / "figures" / f"figure{i}.png").write_bytes(content)
        elif command[2] == "analysis.report_dcu_statistics":
            (output / "statistical_report.md").write_text("fixture report\n")
            (output / "paired_quality_tables.md").write_text("fixture paired table\n")
        else:
            raise AssertionError(command)

    monkeypatch.setattr(reproduce_analysis, "ROOT", root)
    monkeypatch.setattr(reproduce_analysis, "EXPECTED_ZIP", reproduce_analysis.sha256(handoff))
    monkeypatch.setattr(reproduce_analysis.sys, "argv", ["reproduce_analysis", "--handoff", str(handoff),
                                                        "--output-dir", str(output)])
    monkeypatch.setattr(reproduce_analysis.subprocess, "run", fake_analysis)
    monkeypatch.setattr(reproduce_analysis, "git", lambda *args: "fixture")
    monkeypatch.setattr(reproduce_analysis.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(reproduce_analysis.importlib.metadata, "distributions", lambda: [])
    if changed_figure:
        with pytest.raises(SystemExit) as error:
            reproduce_analysis.main()
        assert error.value.code == 1
    else:
        reproduce_analysis.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert len(receipt["figure_byte_matches_published"]) == 4
    assert receipt["figure_byte_matches_published"]["figure0.png"] is not changed_figure
    assert receipt["status"] == ("differences_require_review" if changed_figure else "passed")
