"""Image-blocked method/model ANOVA and direct interaction contrasts, CPU only.

Each dataset is analyzed separately. This is NOT the complete mixed three-factor
dataset/method/model analysis requested by the course.
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
from scipy.linalg import helmert

from .dcu_statistics import DIRECTION, adjust_families, holm, interval, paired_test, stratified_boot_means
from .validate_dcu_team_handoff import DATASETS, METHODS, MODELS, require, validate

QUALITY = ("faithfulness_morf_auc_raw", "stability_spearman")
SEED = 20261028


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("\0".join(map(str, (SEED,) + parts)).encode()).digest()[:8], "big")


def effect_coordinates(values):
    """Orthogonal projections of (image, model, method) into error strata."""
    y = np.asarray(values, float)
    require(y.ndim == 3 and len(y) >= 3 and min(y.shape[1:]) >= 2 and np.isfinite(y).all(), "balanced finite image/model/method array required")
    n, k, m = y.shape
    hk, hm = helmert(k), helmert(m)
    return dict(method=np.sqrt(k) * (y.mean(1) @ hm.T),
                model=np.sqrt(m) * (y.mean(2) @ hk.T),
                method_model=np.einsum("ak,nkm,bm->nab", hk, y, hm).reshape(n, -1))


def rm_effect(coordinates):
    """Univariate repeated-measures F with effect-specific GG correction.

    SS_effect=n*||mean(z)||²; SS_error=sum ||z_i-mean(z)||².
    epsilon=tr(S)²/[d*tr(S²)] in the orthonormal contrast space.
    Never silently replace this effect's subject interaction error by pooled OLS.
    """
    z = np.asarray(coordinates, float)
    require(z.ndim == 2 and len(z) >= 3 and z.shape[1] >= 1 and np.isfinite(z).all(), "invalid repeated-measures coordinates")
    n, d = z.shape
    mu = z.mean(0)
    residual = z - mu
    ss_effect, ss_error = float(n * (mu @ mu)), float(np.sum(residual ** 2))
    threshold = 1e-25 * max(1., float(np.sum(z ** 2)))
    result = dict(n=n, df_effect=d, df_error=(n - 1) * d, ss_effect=ss_effect, ss_error=ss_error,
                  F=None, epsilon_gg=None, df_effect_gg=None, df_error_gg=None,
                  p_uncorrected=None, p_value=None, partial_eta_squared=None, covariance_rank=0)
    if ss_error <= threshold:
        # Zero contrasts provide no evidence; constant nonzero contrasts have no
        # estimable error variance. Keep a visible status, never emit Inf/NaN.
        result.update(status="all_zero" if ss_effect <= threshold else "zero_error_variance",
                      p_value=1. if ss_effect <= threshold else None)
        return result
    covariance = residual.T @ residual / (n - 1)
    epsilon = float(np.clip(np.trace(covariance) ** 2 / (d * np.sum(covariance ** 2)), 1 / d, 1))
    f = ss_effect * (n - 1) / ss_error
    result.update(status="ok", F=f, epsilon_gg=epsilon, df_effect_gg=d * epsilon,
                  df_error_gg=(n - 1) * d * epsilon,
                  p_uncorrected=max(1e-300, float(stats.f.sf(f, d, (n - 1) * d))),
                  p_value=max(1e-300, float(stats.f.sf(f, d * epsilon, (n - 1) * d * epsilon))),
                  partial_eta_squared=ss_effect / (ss_effect + ss_error),
                  covariance_rank=int(np.linalg.matrix_rank(covariance)))
    return result


def repeated_cube(frame, dataset):
    part = frame[frame.dataset.eq(dataset) & frame.metric.isin(QUALITY)]
    require(not part.duplicated(["image_id", "model", "method", "metric"]).any(), "duplicate repeated-measures key")
    columns = pd.MultiIndex.from_product([MODELS, METHODS, QUALITY])
    wide = part.pivot(index="image_id", columns=["model", "method", "metric"], values="value").reindex(columns=columns).sort_index()
    require(len(wide) >= 3 and not wide.isna().any().any(), "incomplete repeated-measures matrix")
    values = wide.to_numpy(float).reshape(len(wide), len(MODELS), len(METHODS), len(QUALITY))
    require(np.isfinite(values).all(), "nonfinite repeated-measures matrix")
    return wide.index, values


def analyze(frames, classes, resamples=10000):
    omnibus, cells, marginals, contrasts, interactions, cohorts = [], [], [], [], [], []
    for cohort in ("common_correct", "all"):
        for dataset in DATASETS:
            ids, values = repeated_cube(frames[cohort], dataset)
            labels = classes.loc[[(dataset, i) for i in ids]].to_numpy()
            class_counts = pd.Series(labels).value_counts()
            cohorts.append(dict(cohort=cohort, dataset=dataset, n=len(ids), classes=len(class_counts),
                                singleton_classes=int((class_counts == 1).sum()),
                                image_ids_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest()))
            boot = stratified_boot_means(values.reshape(len(ids), -1), labels, resamples,
                                        seed_for(cohort, dataset, "image-bootstrap")).reshape(resamples, *values.shape[1:])
            # A large number of singleton ImageNet classes makes the fixed-class
            # estimand particularly conditional. Also expose ordinary paired
            # image-bootstrap CIs; never suppress that sensitivity result.
            iid_boot = stratified_boot_means(values.reshape(len(ids), -1), np.repeat("images", len(ids)), resamples,
                                            seed_for(cohort, dataset, "iid-image-bootstrap")).reshape(resamples, *values.shape[1:])
            means = values.mean(0)
            for j, metric in enumerate(QUALITY):
                context = dict(cohort=cohort, dataset=dataset, metric=metric, n=len(ids))
                for name, z in effect_coordinates(values[:, :, :, j]).items():
                    omnibus.append(dict(**{k: v for k, v in context.items() if k != "n"}, effect=name,
                                         family=cohort + "/rm_quality_omnibus", **rm_effect(z)))
                for k, model in enumerate(MODELS):
                    for m, method in enumerate(METHODS):
                        cells.append(dict(**context, model=model, method=method, mean=float(means[k, m, j]),
                                          mean_ci95=interval(boot[:, k, m, j]), iid_image_ci95=interval(iid_boot[:, k, m, j])))
                for axis, levels in (("method", METHODS), ("model", MODELS)):
                    observed = means[:, :, j].mean(0 if axis == "method" else 1)
                    sampled = boot[:, :, :, j].mean(1 if axis == "method" else 2)
                    iid_sampled = iid_boot[:, :, :, j].mean(1 if axis == "method" else 2)
                    for i, level in enumerate(levels):
                        marginals.append(dict(**context, factor=axis, level=level, mean=float(observed[i]),
                                              mean_ci95=interval(sampled[:, i]), iid_image_ci95=interval(iid_sampled[:, i])))
                interaction = means[:, :, j] - means[:, :, j].mean(0, keepdims=True) - means[:, :, j].mean(1, keepdims=True) + means[:, :, j].mean()
                sampled = boot[:, :, :, j]
                boot_interaction = sampled - sampled.mean(1, keepdims=True) - sampled.mean(2, keepdims=True) + sampled.mean((1, 2), keepdims=True)
                iid_sampled = iid_boot[:, :, :, j]
                iid_interaction = iid_sampled - iid_sampled.mean(1, keepdims=True) - iid_sampled.mean(2, keepdims=True) + iid_sampled.mean((1, 2), keepdims=True)
                for k, model in enumerate(MODELS):
                    for m, method in enumerate(METHODS):
                        interactions.append(dict(**context, model=model, method=method,
                            centered_cell_effect=float(interaction[k, m]), mean_ci95=interval(boot_interaction[:, k, m]),
                            iid_image_ci95=interval(iid_interaction[:, k, m])))
                for (u, v), (a, b) in itertools.product(itertools.combinations(range(len(MODELS)), 2),
                                                      itertools.combinations(range(len(METHODS)), 2)):
                    direction = DIRECTION[metric]
                    gap_u = direction * (values[:, u, a, j] - values[:, u, b, j])
                    gap_v = direction * (values[:, v, a, j] - values[:, v, b, j])
                    test = paired_test(gap_u, gap_v, 1)
                    bootstrap_did = direction * ((boot[:, u, a, j] - boot[:, u, b, j]) - (boot[:, v, a, j] - boot[:, v, b, j]))
                    iid_did = direction * ((iid_boot[:, u, a, j] - iid_boot[:, u, b, j]) - (iid_boot[:, v, a, j] - iid_boot[:, v, b, j]))
                    contrasts.append(dict(**{k: v for k, v in context.items() if k != "n"},
                        method_a=METHODS[a], method_b=METHODS[b], model_u=MODELS[u], model_v=MODELS[v],
                        mean_did_ci95=interval(bootstrap_did), iid_image_ci95=interval(iid_did),
                        family=cohort + "/direct_did_quality", **test))
            print(f"Repeated-measures analysis: {cohort}/{dataset}, n={len(ids)}", flush=True)
    for cohort in ("common_correct", "all"):
        selected = [r for r in omnibus if r["cohort"] == cohort]
        require(len(selected) == 12, "12-test omnibus family required")
        # Undefined tests remain visible and consume their prespecified slot.
        for row, p in zip(selected, holm(
                [r["p_value"] if r["p_value"] is not None else 1. for r in selected])):
            row.update(p_holm=float(p) if row["p_value"] is not None else None, family_size=12)
    adjust_families(contrasts)
    adjust_families(contrasts, "sign_p_value", "sign_p_holm")
    return dict(cohorts=cohorts, omnibus=omnibus, cell_means=cells, marginal_means=marginals,
                centered_interactions=interactions, direct_did=contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("analysis/dcu_20261028/repeated_measures.json"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    require(args.bootstrap >= 1000, "production bootstrap >=1000 required")
    frames, receipt, classes = validate(args.handoff, args.project)
    data = analyze(frames, classes, args.bootstrap)
    data.update(receipt=receipt, protocol=dict(seed=SEED, bootstrap_resamples=args.bootstrap,
        main_cohort="common_correct", sensitivity_cohort="all", quality_metrics=QUALITY,
        global_test="two-within-factor repeated-measures ANOVA; effect-specific subject interaction error; GG-corrected F p-values always used; Holm over 12 tests per cohort",
        effect="partial eta squared=SS_effect/(SS_effect+SS_subject_x_effect); not generalized eta squared, not causal explained variance",
        uncertainty="class-stratified original-image bootstrap, same draws for every model/method/metric; marginal percentile 95% CIs, not simultaneous intervals; singleton classes contribute no within-class bootstrap variation",
        ci_sensitivity="also 10000 ordinary paired original-image bootstrap draws (bootstrap_resamples argument controls both); iid_image_ci95 allows class proportions to vary; neither interval includes model-training uncertainty",
        direct_contrast="[benefit(a)-benefit(b)] on model_u minus same gap on model_v; benefit=-AUC or +Spearman. Positive means relative advantage of a is larger on u; not a universal method win",
        followup_test="direct paired DID signed-rank tests (normal approximation) and sign sensitivity; each Holm over 180 quality contrasts per cohort; rank p-values do not test the mean represented by the bootstrap CI",
        assumptions="independent original images and adequate sampling; F/GG is an approximate univariate RM test, not a distribution-free test; GG corrects sphericity df, not nonnormality, selection bias or training uncertainty",
        scope="datasets separate; three fixed models/checkpoints; no dataset-factor test, complete three-factor ANOVA, recorded-time inference, new GPU run, or teacher acceptance"),
        environment=dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "PyYAML")}),
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved " + str(args.output))


if __name__ == "__main__":
    main()
