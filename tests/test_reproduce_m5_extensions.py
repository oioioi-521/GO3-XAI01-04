import pytest

from scripts.reproduce_m5_extensions import compare_values


def test_roundoff_is_disclosed_and_accepted():
    result = compare_values({"F": 300202928.57873356}, {"F": 300202928.5787342})
    assert not result["exact_match"]
    assert result["within_tolerance"]
    assert result["float_differences"] == 1
    assert not compare_values({"p_uncorrected": 1e-145}, {"p_uncorrected": 1e-140})["within_tolerance"]


@pytest.mark.parametrize("expected,actual", [
    ({"p_value": 1e-145}, {"p_value": 1e-140}),
    ({"bootstrap_exceedances": 12}, {"bootstrap_exceedances": 13}),
    ({"ci": [-1e-15, 1.]}, {"ci": [1e-15, 1.]}),
    ({"mean": 0.}, {"mean": 1e-15}),
    ({"n": 332}, {"n": 332.}),
    ({"sha": "abc"}, {"sha": "abd"}),
    ({"v": [1., 2.]}, {"v": [1.]}),
    ({"v": float("nan")}, {"v": float("nan")}),
    ({"v": 1.}, {"v": 1., "extra": True}),
])
def test_material_or_structural_difference_is_rejected(expected, actual):
    assert not compare_values(expected, actual)["within_tolerance"]
