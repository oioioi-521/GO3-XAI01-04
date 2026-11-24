"""Fixed-seed synthetic null/power diagnostics, not formal experiment evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .dcu_repeated_measures import effect_coordinates, rm_effect


def calibrate(trials=1000, n=60, seed=20261028):
    rng = np.random.default_rng(seed)
    rows = []
    pattern = np.array([-1., 0, 1])[:, None] * np.linspace(-1, 1, 6)[None, :]
    for name in ("spherical_gaussian_null", "anisotropic_gaussian_null", "centered_exponential_null", "injected_interaction"):
        hits = 0
        for _ in range(trials):
            y = rng.exponential(size=(n, 3, 6)) - 1 if name == "centered_exponential_null" else rng.normal(size=(n, 3, 6))
            if name == "anisotropic_gaussian_null":
                y *= np.linspace(.2, 4., 18).reshape(3, 6)
            if name == "injected_interaction":
                y += pattern
            result = rm_effect(effect_coordinates(y)["method_model"])
            hits += result["p_value"] < .05
        ci = stats.binomtest(hits, trials).proportion_ci()
        rows.append(dict(scenario=name, trials=trials, n=n, rejected=hits, rejection_rate=hits/trials,
                         binomial_ci95=[float(ci.low), float(ci.high)]))
    # Broad fixed-seed software alarms, not a proof of exact 5% calibration.
    passed = all(r["rejection_rate"] <= .10 for r in rows[:3]) and rows[3]["rejection_rate"] >= .9
    return dict(seed=seed, alpha=.05, diagnostics=rows, broad_alarm_passed=passed,
                analysis_code_sha256=hashlib.sha256(Path(__file__).with_name("dcu_repeated_measures.py").read_bytes()).hexdigest(),
                limits="Synthetic reference diagnostics only. Does not validate distributional assumptions or exact Type I error for the observed dataset, all endpoints, or other sample sizes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("analysis/dcu_20261028/synthetic_diagnostics.json"))
    args = parser.parse_args()
    data = calibrate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if not data["broad_alarm_passed"]:
        raise RuntimeError("synthetic broad alarm failed; inspect diagnostics")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
