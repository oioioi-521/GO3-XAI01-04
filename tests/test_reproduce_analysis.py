"""Comparison must detect both changed values and missing experiment rows."""
from scripts.reproduce_analysis import TABLES, compare_tables


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
