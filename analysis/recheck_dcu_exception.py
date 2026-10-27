"""Recheck B's IG/KernelSHAP exception using frozen CSV and the existing estimators."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .dcu_statistics import holm, interval, method_matrix, paired_test, seed_for, stratified_boot_means
from .validate_dcu_team_handoff import ALL_METRICS, DATASETS, METHODS, MODELS, require, validate


def recheck(handoff, project, statistics):
    frames, receipt, classes = validate(handoff, project)
    results = []
    for cohort in ("model_correct", "common_correct", "all"):
        family = []
        for dataset, model in itertools.product(DATASETS, MODELS):
            wide = method_matrix(frames[cohort], dataset, model)
            for metric in ALL_METRICS[:2]:
                direction = -1 if metric == ALL_METRICS[0] else 1
                for a, b in itertools.combinations(METHODS, 2):
                    r = paired_test(wide[a, metric], wide[b, metric], direction)
                    family.append(dict(dataset=dataset, model=model, metric=metric, a=a, b=b, **r))
        require(len(family) == 180, "quality family must have 180 tests")
        for row, p in zip(family, holm([r["p_value"] for r in family])):
            row["p_holm"] = float(p)
        row = next(r for r in family if r["dataset"] == "imagenet" and r["model"] == "densenet121"
                   and r["metric"] == ALL_METRICS[0] and r["a"] == "ig" and r["b"] == "kernelshap")
        original = next(r for r in statistics["method_pairs"] if r["cohort"] == cohort
                        and all(r[k] == row[k] for k in ("dataset", "model", "metric", "a", "b")))
        wide = method_matrix(frames[cohort], "imagenet", "densenet121")
        columns = [("ig", ALL_METRICS[0]), ("kernelshap", ALL_METRICS[0])]
        values = wide[columns].to_numpy(float)
        labels = classes.loc[[("imagenet", i) for i in wide.index]].to_numpy()
        boot = stratified_boot_means(values, labels, statistics["protocol"]["bootstrap_resamples"],
                                    seed_for(cohort, "imagenet", "densenet121", "bootstrap"))
        ci = interval(boot[:, 1] - boot[:, 0])
        for field in ("n", "wins_a", "losses_a", "ties", "mean_advantage_a", "rank_biserial_a", "p_value", "p_holm"):
            require(np.isclose(row[field], original[field], rtol=1e-10, atol=1e-14), "recomputed field mismatch: " + field)
        require(np.allclose(ci, original["mean_advantage_ci95"], rtol=1e-10, atol=1e-13), "bootstrap CI mismatch")
        results.append(dict(cohort=cohort, mean_ig=float(values[:, 0].mean()), mean_kernelshap=float(values[:, 1].mean()),
                            mean_advantage_ci95=ci, **row))
    return dict(date="2026-10-27", status="recomputed_and_matched", receipt=receipt,
                original_statistics_sha256=hashlib.sha256((project / "analysis/dcu_20261015/statistics.json").read_bytes()).hexdigest(),
                analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                resamples=statistics["protocol"]["bootstrap_resamples"], family_size_per_cohort=180, results=results,
                checks="CSV-derived paired counts, mean differences, rank effects, Wilcoxon p, all 180 quality p-values and Holm per cohort; class-stratified paired bootstrap CI at the existing seed",
                limits="Software reanalysis of frozen observations, not a new GPU run or teacher/human acceptance. Marginal mean CIs are not simultaneous Holm intervals. Not an interaction test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("analysis/dcu_20261027/ig_kernelshap_recheck.json"))
    args = parser.parse_args()
    statistics = json.loads((args.project / "analysis/dcu_20261015/statistics.json").read_text(encoding="utf-8"))
    result = recheck(args.handoff, args.project, statistics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("All three cohorts and original 180-test quality families matched: " + str(args.output))


if __name__ == "__main__":
    main()
