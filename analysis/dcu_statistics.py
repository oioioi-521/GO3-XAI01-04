"""Paired, image-level analysis of the separately received 36-unit DCU matrix.

No GPU computation. No pooling datasets or treating five noise repeats as n=5.
All exports are derived summaries, not the delivered per-image/trace tables.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy import stats

from .validate_dcu_team_handoff import (
    ALL_METRICS, DATASETS, METHODS, METRICS, MODELS, validate,
)

SEED = 20261015
DIRECTION = {"faithfulness_morf_auc_raw": -1, "stability_spearman": 1,
             "efficiency_time_ms": -1, "stability_valid_rate": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("\0".join(map(str, (SEED,) + parts)).encode()).digest()[:8], "big")


def holm(p_values):
    values = np.asarray(p_values, dtype=float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "invalid p-values")
    if not len(values):
        return values
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def adjust_families(rows, field="p_value", output="p_holm"):
    families = sorted({r["family"] for r in rows})
    for family in families:
        selected = [r for r in rows if r["family"] == family and r.get(field) is not None]
        for row, p in zip(selected, holm([r[field] for r in selected])):
            row[output] = float(p)
            row["family_size"] = len(selected)


def stratified_boot_means(values, labels, resamples, seed):
    """Fixed-class-count case bootstrap; same draw for all paired columns."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=str)
    require(values.ndim == 2 and len(values) == len(labels) and len(values) >= 2,
            "bootstrap shape/n")
    require(np.isfinite(values).all() and resamples >= 20, "bootstrap values/resamples")
    rng = np.random.default_rng(seed)
    result = np.zeros((resamples, values.shape[1]), dtype=float)
    for label in sorted(set(labels)):
        group = values[labels == label]
        counts = rng.multinomial(len(group), np.full(len(group), 1 / len(group)), size=resamples)
        result += counts @ group
    return result / len(values)


def interval(values):
    low, high = np.quantile(values, [.025, .975], method="linear")
    return [float(low), float(high)]


def paired_test(a, b, direction):
    """Positive effects favor A. Wilcoxon is a signed-rank, not mean test."""
    raw = direction * (np.asarray(a, float) - np.asarray(b, float))
    require(raw.ndim == 1 and len(raw) >= 2 and np.isfinite(raw).all(), "paired input")
    d = np.round(raw, 12)
    nonzero = d[d != 0]
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero), method="average")
        rb = float(np.sum(ranks * np.sign(nonzero)) / np.sum(ranks))
        # Explicit normal approximation handles ties/zeros, never label exact.
        w = stats.wilcoxon(d, zero_method="wilcox", correction=True,
                           alternative="two-sided", method="approx")
        statistic, p = float(w.statistic), max(1e-300, float(w.pvalue))
        sign_p = max(1e-300, float(stats.binomtest(wins, wins + losses, .5).pvalue))
    else:
        rb, statistic, p, sign_p = 0., 0., 1., 1.
    sd = float(np.std(raw, ddof=1))
    dz = float(np.mean(raw) / sd) if sd > 0 else (0. if np.all(raw == 0) else None)
    return dict(n=len(raw), wins_a=wins, losses_a=losses, ties=ties,
                mean_advantage_a=float(np.mean(raw)), median_advantage_a=float(np.median(raw)),
                rank_biserial_a=rb, cohen_dz_a=dz,
                paired_superiority_a=(wins + .5 * ties) / len(raw),
                wilcoxon_statistic=statistic, p_value=p, sign_p_value=sign_p,
                p_method="two-sided Wilcoxon normal approximation, continuity/tie correction; zeros dropped",
                difference_round_decimals=12)


def friedman(matrix, names, direction, permutations, seed):
    matrix = np.asarray(matrix, float)
    require(matrix.ndim == 2 and matrix.shape[1] == len(names) >= 3
            and len(matrix) >= 2 and np.isfinite(matrix).all(), "Friedman input")
    rank = stats.rankdata(matrix, axis=1, method="average")
    mean_ranks = stats.rankdata(-direction * matrix, axis=1, method="average").mean(0)
    if np.all(np.ptp(matrix, axis=1) == 0):
        q, p_asym, p = 0., 1., 1.
    else:
        result = stats.friedmanchisquare(*matrix.T)
        q, p_asym = float(result.statistic), max(1e-300, float(result.pvalue))
        require(permutations >= 20, "at least 20 permutations required")
        # Rank-sum squared distance has the same ordering as tie-corrected Q;
        # the within-block ties and correction remain unchanged by permutation.
        centered = rank.sum(0) - len(rank) * (rank.shape[1] + 1) / 2
        observed = float(centered @ centered)
        rng, exceed = np.random.default_rng(seed), 0
        for start in range(0, permutations, 128):
            count = min(128, permutations - start)
            order = rng.random((count,) + rank.shape).argsort(axis=2)
            sums = np.take_along_axis(rank[None, :, :], order, axis=2).sum(1)
            sums -= len(rank) * (rank.shape[1] + 1) / 2
            simulated = np.sum(sums * sums, axis=1)
            exceed += int(np.count_nonzero(simulated >= observed - 1e-8))
        p = (exceed + 1) / (permutations + 1)
    return dict(n=len(matrix), k=len(names), statistic=q, p_value=p,
                asymptotic_p_diagnostic=p_asym,
                kendall_w=min(1., max(0., q / (len(matrix) * (len(names) - 1)))),
                mean_ranks_best_is_1=dict(zip(names, map(float, mean_ranks))),
                permutations=permutations, p_method="within-image rank permutation with plus-one correction")


def pareto_flags(costs):
    """Minimize raw costs; support (..., alternatives, objectives), keep ties."""
    costs = np.asarray(costs, float)
    require(costs.ndim >= 2 and np.isfinite(costs).all(), "Pareto costs must be finite")
    competitor, candidate = costs[..., None, :, :], costs[..., :, None, :]
    dominated = np.all(competitor <= candidate, axis=-1) & np.any(competitor < candidate, axis=-1)
    return ~np.any(dominated, axis=-1)


def method_matrix(frame, dataset, model):
    selected = frame[frame.dataset.eq(dataset) & frame.model.eq(model)]
    require(not selected.duplicated(["image_id", "method", "metric"]).any(), "duplicate matched key")
    columns = pd.MultiIndex.from_product([METHODS, ALL_METRICS])
    wide = selected.pivot(index="image_id", columns=["method", "metric"], values="value").reindex(columns=columns).sort_index()
    require(len(wide) >= 2 and not wide.isna().any().any(), "incomplete method pairing")
    return wide


def analyze(frames, classes, resamples=10000, permutations=4999):
    summaries, omnibus, pairs, pareto, correlations = [], [], [], [], []
    for cohort in ("model_correct", "common_correct", "all"):
        frame = frames[cohort]
        for dataset, model in itertools.product(DATASETS, MODELS):
            wide = method_matrix(frame, dataset, model)
            labels = classes.loc[[(dataset, image_id) for image_id in wide.index]].to_numpy()
            values = wide.to_numpy(float).reshape(len(wide), len(METHODS), len(ALL_METRICS))
            boot = stratified_boot_means(values.reshape(len(wide), -1), labels, resamples,
                                        seed_for(cohort, dataset, model, "bootstrap")).reshape(resamples, len(METHODS), len(ALL_METRICS))
            means = values.mean(0)
            context = dict(cohort=cohort, dataset=dataset, model=model)
            for m, method in enumerate(METHODS):
                for j, metric in enumerate(ALL_METRICS):
                    summaries.append(dict(**context, method=method, metric=metric, n=len(wide),
                        mean=float(means[m, j]), std=float(values[:, m, j].std(ddof=1)),
                        median=float(np.median(values[:, m, j])), mean_ci95=interval(boot[:, m, j])))
            for j, metric in enumerate(METRICS):
                timing = metric == "efficiency_time_ms"
                family = f"{cohort}/method/" + ("recorded_timing_exploratory" if timing else "quality")
                result = friedman(values[:, :, j], METHODS, DIRECTION[metric], permutations,
                                  seed_for(cohort, dataset, model, metric, "permutation"))
                omnibus.append(dict(**context, metric=metric, family=family + "/omnibus", **result))
                for a, b in itertools.combinations(range(len(METHODS)), 2):
                    test = paired_test(values[:, a, j], values[:, b, j], DIRECTION[metric])
                    pairs.append(dict(**context, metric=metric, a=METHODS[a], b=METHODS[b], family=family,
                        mean_advantage_ci95=interval(DIRECTION[metric] * (boot[:, a, j] - boot[:, b, j])), **test))
            quality = means[:, :2] * [1, -1]
            costs = means[:, :3] * [1, -1, 1]
            qflag, tflag = pareto_flags(quality), pareto_flags(costs)
            qfreq = pareto_flags(boot[:, :, :2] * [1, -1]).mean(0)
            tfreq = pareto_flags(boot[:, :, :3] * [1, -1, 1]).mean(0)
            for m, method in enumerate(METHODS):
                dominated_by = [METHODS[k] for k in range(len(METHODS))
                    if np.all(costs[k] <= costs[m]) and np.any(costs[k] < costs[m])]
                pareto.append(dict(**context, method=method, n=len(wide),
                    faithfulness=float(means[m, 0]), stability=float(means[m, 1]), recorded_ms=float(means[m, 2]),
                    quality_front=bool(qflag[m]), recorded_time_3d_front=bool(tflag[m]),
                    quality_front_bootstrap_frequency=float(qfreq[m]),
                    recorded_time_front_bootstrap_frequency=float(tfreq[m]), dominated_by=dominated_by))
            for m, method in enumerate(METHODS):
                for a, b in itertools.combinations(range(len(METRICS)), 2):
                    x, y = values[:, m, a], values[:, m, b]
                    row = dict(**context, method=method, metric_a=METRICS[a], metric_b=METRICS[b],
                               n=len(wide), family=f"{cohort}/correlations", status="constant",
                               pearson_r=None, pearson_p=None, spearman_rho=None, spearman_p=None)
                    if np.ptp(x) > 0 and np.ptp(y) > 0:
                        pr, sr = stats.pearsonr(x, y), stats.spearmanr(x, y)
                        row.update(status="ok", pearson_r=float(pr.statistic), pearson_p=max(1e-300, float(pr.pvalue)),
                                   spearman_rho=float(sr.statistic), spearman_p=max(1e-300, float(sr.pvalue)))
                    correlations.append(row)
            print(f"analyzed {cohort}/{dataset}/{model}: n={len(wide)}", flush=True)
    # Cross-model comparisons ONLY on common-correct matched images. Never pool
    # each model's different correct cohort, or treat model observations as IID.
    common = frames["common_correct"]
    cross_omnibus, cross_pairs = [], []
    for dataset in DATASETS:
        selected = common[common.dataset.eq(dataset)]
        columns = pd.MultiIndex.from_product([MODELS, METHODS, ALL_METRICS])
        wide = selected.pivot(index="image_id", columns=["model", "method", "metric"], values="value").reindex(columns=columns).sort_index()
        require(not wide.isna().any().any(), "incomplete cross-model pairing")
        values = wide.to_numpy(float).reshape(len(wide), len(MODELS), len(METHODS), len(ALL_METRICS))
        labels = classes.loc[[(dataset, image_id) for image_id in wide.index]].to_numpy()
        boot = stratified_boot_means(wide.to_numpy(float), labels, resamples,
                                    seed_for(dataset, "cross-model-bootstrap")).reshape(resamples, len(MODELS), len(METHODS), len(ALL_METRICS))
        for m, method in enumerate(METHODS):
            for j, metric in enumerate(METRICS):
                family = "cross_model/" + ("recorded_timing_exploratory" if j == 2 else "quality")
                context = dict(cohort="common_correct", dataset=dataset, method=method, metric=metric)
                result = friedman(values[:, :, m, j], MODELS, DIRECTION[metric], permutations,
                                  seed_for(dataset, method, metric, "cross-permutation"))
                cross_omnibus.append(dict(**context, family=family + "/omnibus", **result))
                for a, b in itertools.combinations(range(len(MODELS)), 2):
                    result = paired_test(values[:, a, m, j], values[:, b, m, j], DIRECTION[metric])
                    cross_pairs.append(dict(**context, a=MODELS[a], b=MODELS[b], family=family,
                        mean_advantage_ci95=interval(DIRECTION[metric] * (boot[:, a, m, j] - boot[:, b, m, j])), **result))
    for rows in (omnibus, pairs, cross_omnibus, cross_pairs):
        adjust_families(rows)
        if rows and "sign_p_value" in rows[0]:
            adjust_families(rows, "sign_p_value", "sign_p_holm")
    # Pearson and Spearman form ONE family per cohort, not 216 isolated tests.
    for cohort in ("model_correct", "common_correct", "all"):
        endpoints = [(r, field) for r in correlations if r["cohort"] == cohort
                     for field in ("pearson_p", "spearman_p") if r[field] is not None]
        for (row, field), p in zip(endpoints, holm([r[f] for r, f in endpoints])):
            row[field + "_holm"] = float(p)
            row["family_size"] = len(endpoints)
    return dict(summaries=summaries, method_omnibus=omnibus, method_pairs=pairs,
                pareto=pareto, correlations=correlations, model_omnibus=cross_omnibus, model_pairs=cross_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=4999)
    args = parser.parse_args()
    require(args.bootstrap >= 1000 and args.permutations >= 999, "insufficient production resampling")
    frames, receipt, classes = validate(args.handoff, args.project)
    results = analyze(frames, classes, args.bootstrap, args.permutations)
    results["receipt"] = receipt
    results["protocol"] = dict(seed=SEED, bootstrap_resamples=args.bootstrap,
        bootstrap="class-stratified original-image case bootstrap; shared draws across methods/metrics; marginal percentile 95% CIs, not simultaneous CIs",
        permutations=args.permutations, multiplicity="Holm within each named family, pooled across both datasets/3 models; separate quality and recorded timing; sign tests corrected separately",
        main_cohort="model_correct", sensitivity_cohorts=["common_correct", "all"],
        metric_directions=DIRECTION, effect_sign="positive advantage/rank-biserial/dz favors a",
        valid_rate="QC summary only, not a fourth rank endpoint",
        correlation="within method/model/dataset; Pearson+Spearman Holm together per cohort; asymptotic exploratory p-values",
        inference="conditional on frozen classes/correct cohorts and fixed budgets; original images treated as independent; Wilcoxon assumes symmetric differences, exact sign tests supplied as sensitivity",
        timing="recorded two-round shared-load wrapper times; descriptive/exploratory only, no causal hardware speedup",
        anova="ordinary pooled factorial ANOVA deliberately not used: repeated images and different dataset/output semantics",
        limits="not a new GPU run, budget convergence proof, independent replication of unsaved noise maps, or universal method ranking")
    results["environment"] = dict(python=platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "PyYAML")})
    results["analysis_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    results["validator_code_sha256"] = hashlib.sha256(
        Path(__file__).with_name("validate_dcu_team_handoff.py").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
