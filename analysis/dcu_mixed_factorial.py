"""Dataset-between, model/method-within mixed factorial analysis of frozen CSV.

Raw-score cross-dataset hypotheses describe the two observed evaluation tasks;
they are not a common latent-quality or causal dataset comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy import stats

from .dcu_repeated_measures import effect_coordinates
from .dcu_statistics import holm, interval, stratified_boot_means
from .validate_dcu_team_handoff import DATASETS, METHODS, METRICS, MODELS, require, validate

SEED = 20261029


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("\0".join(map(str, (SEED,) + parts)).encode()).digest()[:8], "big")


def groups_check(groups, weights):
    arrays = [np.asarray(z, float) for z in groups]
    w = np.asarray(weights, float)
    require(len(arrays) == len(w) == 2 and np.isfinite(w).all() and np.any(w != 0), "two groups and contrast weights required")
    require(all(z.ndim == 2 and len(z) >= 3 and z.shape[1] > 0 and np.isfinite(z).all() for z in arrays)
            and arrays[0].shape[1] == arrays[1].shape[1], "finite group contrast matrices required")
    return arrays, w


def mixed_anova(groups, weights):
    """Equal-cell (Type III) hypothesis in an effect's subject-within-group stratum.

    weights=(.5,.5) tests the equally weighted dataset marginal within effect;
    weights=(1,-1) tests the between-dataset difference in that within effect.
    Covariance homogeneity is a nominal assumption, checked by separate WTS.
    """
    groups, w = groups_check(groups, weights)
    d = groups[0].shape[1]
    sizes = np.array([len(z) for z in groups])
    means = [z.mean(0) for z in groups]
    beta = sum(c * mu for c, mu in zip(w, means))
    residuals = [z - mu for z, mu in zip(groups, means)]
    scatter = sum(e.T @ e for e in residuals)
    v = float(np.sum(w ** 2 / sizes))
    ss_hypothesis = float(beta @ beta / v)
    ss_error = float(np.trace(scatter))
    ndf = int(sizes.sum() - 2)
    result = dict(n_by_dataset=list(map(int, sizes)), contrast_dimension=d, ss_hypothesis=ss_hypothesis,
                  ss_error=ss_error, df1=d, df2=ndf * d, F=None, epsilon_gg=None,
                  df1_gg=None, df2_gg=None, p_value=None, p_uncorrected=None,
                  partial_eta_squared=None, weights=list(map(float, w)))
    threshold = 1e-25 * max(1., sum(float(np.sum(z ** 2)) for z in groups))
    if ss_error <= threshold:
        result.update(status="all_zero" if ss_hypothesis <= threshold else "zero_error_variance",
                      p_value=1. if ss_hypothesis <= threshold else None)
        return result
    covariance = scatter / ndf
    eps = float(np.clip(np.trace(covariance) ** 2 / (d * np.sum(covariance ** 2)), 1/d, 1))
    f = ss_hypothesis * ndf / ss_error
    result.update(status="ok", F=f, epsilon_gg=eps, df1_gg=d * eps, df2_gg=ndf * d * eps,
                  p_value=max(1e-300, float(stats.f.sf(f, d * eps, ndf * d * eps))),
                  p_uncorrected=max(1e-300, float(stats.f.sf(f, d, ndf * d))),
                  partial_eta_squared=ss_hypothesis / (ss_hypothesis + ss_error))
    return result


def parametric_wald(groups, weights, iterations=9999, seed=SEED):
    """Heteroscedastic WTS, recalibrated by Gaussian parametric bootstrap.

    Uses exact Gaussian normal-mean/Wishart-scatter sufficient-statistic draws
    instead of materializing every synthetic subject. Covariances are recomputed
    on EACH draw. No equal-covariance or observed-normality assumption is imposed
    for the asymptotic bootstrap justification; finite fourth moments and
    independent subjects plus full-rank group contrast covariances are required.
    """
    groups, w = groups_check(groups, weights)
    require(iterations >= 99, "at least 99 bootstrap draws required")
    sizes = [len(z) for z in groups]
    d = groups[0].shape[1]
    covariance = [np.atleast_2d(np.cov(z, rowvar=False, ddof=1)) for z in groups]
    pooled_diagonal = np.mean([np.diag(s) for s in covariance], axis=0)
    result = dict(status="ok", WTS=None, df=d, p_value=None, asymptotic_p=None,
                  bootstrap_iterations=iterations, bootstrap_exceedances=None,
                  mc_se=None, covariance_ranks=[], covariance_condition_numbers=[])
    if np.any(pooled_diagonal <= 0):
        result["status"] = "nonpositive_group_variance"
        return result
    # A shared nonsingular scaling is invariant for WTS and improves numerical
    # conditioning. Never normalize datasets separately or add arbitrary jitter.
    scale = np.sqrt(pooled_diagonal)
    scaled = [z / scale for z in groups]
    covariance = [np.atleast_2d(np.cov(z, rowvar=False, ddof=1)) for z in scaled]
    for s in covariance:
        eigen = np.linalg.eigvalsh(s)
        rank = int(np.sum(eigen > eigen[-1] * 1e-12))
        result["covariance_ranks"].append(rank)
        result["covariance_condition_numbers"].append(float(eigen[-1]/eigen[0]) if rank == d else None)
    if min(result["covariance_ranks"]) < d or min(sizes) <= d:
        result["status"] = "singular_or_insufficient_group_covariance"
        return result
    beta = sum(c * z.mean(0) for c, z in zip(w, scaled))
    variance = sum(c*c*s/n for c, s, n in zip(w, covariance, sizes))
    observed = float(beta @ np.linalg.solve(variance, beta))
    rng = np.random.default_rng(seed)
    means, covariances = [], []
    for s, n in zip(covariance, sizes):
        means.append((rng.standard_normal((iterations, d)) @ np.linalg.cholesky(s).T) / np.sqrt(n))
        scatter = np.asarray(stats.wishart.rvs(df=n-1, scale=s, size=iterations, random_state=rng)).reshape(iterations, d, d)
        covariances.append(scatter / (n-1))
    star_beta = sum(c * mean for c, mean in zip(w, means))
    star_v = sum(c*c*s/n for c, s, n in zip(w, covariances, sizes))
    solution = np.linalg.solve(star_v, star_beta[..., None])[..., 0]
    simulated = np.einsum("bi,bi->b", star_beta, solution)
    require(np.isfinite(simulated).all(), "invalid bootstrap WTS")
    exceed = int(np.count_nonzero(simulated >= observed - 1e-12))
    p = (exceed + 1) / (iterations + 1)
    result.update(WTS=observed, p_value=p, asymptotic_p=max(1e-300, float(stats.chi2.sf(observed, d))),
                  bootstrap_exceedances=exceed, mc_se=float(np.sqrt(p*(1-p)/(iterations+1))))
    return result


def cube(frame, dataset):
    part = frame[frame.dataset.eq(dataset) & frame.metric.isin(METRICS)]
    require(not part.duplicated(["image_id", "model", "method", "metric"]).any(), "duplicate mixed-design key")
    columns = pd.MultiIndex.from_product([MODELS, METHODS, METRICS])
    wide = part.pivot(index="image_id", columns=["model", "method", "metric"], values="value").reindex(columns=columns).sort_index()
    require(len(wide) >= 3 and not wide.isna().any().any(), "incomplete mixed-design matrix")
    y = wide.to_numpy(float, copy=True).reshape(len(wide), len(MODELS), len(METHODS), len(METRICS))
    require(np.isfinite(y).all() and (y[:, :, :, 2] >= 0).all(), "invalid score/time")
    y[:, :, :, 2] = np.log1p(y[:, :, :, 2])
    return wide.index, y


def analyze(frames, classes, iterations=9999, resamples=10000):
    tests, deltas, cohorts = [], [], []
    for cohort in ("common_correct", "all"):
        arrays, bootstrap, iid_bootstrap = [], [], []
        for dataset in DATASETS:
            ids, y = cube(frames[cohort], dataset)
            labels = classes.loc[[(dataset, i) for i in ids]].to_numpy()
            counts = pd.Series(labels).value_counts()
            cohorts.append(dict(cohort=cohort, dataset=dataset, n=len(ids), classes=len(counts),
                                singleton_classes=int((counts==1).sum()), subject_key="dataset + image_id"))
            arrays.append(y)
            for target, label, key in ((bootstrap, labels, "class"), (iid_bootstrap, np.repeat("images", len(ids)), "iid")):
                target.append(stratified_boot_means(y.reshape(len(y), -1), label, resamples,
                              seed_for(cohort, dataset, key)).reshape(resamples, *y.shape[1:]))
        for j, metric in enumerate(METRICS):
            ys = [y[:, :, :, j] for y in arrays]
            coordinates = [effect_coordinates(y) for y in ys]
            grand = [np.sqrt(y.shape[1]*y.shape[2]) * y.mean((1,2))[:, None] for y in ys]
            hypotheses = [("dataset", grand, (1., -1.))]
            for name in ("method", "model", "method_model"):
                groups = [z[name] for z in coordinates]
                hypotheses.extend([(name, groups, (.5, .5)), ("dataset_" + name, groups, (1., -1.))])
            family = cohort + ("/mixed_quality" if j < 2 else "/recorded_logtime_exploratory")
            for effect, groups, weights in hypotheses:
                nominal = mixed_anova(groups, weights)
                robust = parametric_wald(groups, weights, iterations, seed_for(cohort, metric, effect, "WTS"))
                beta = sum(c*z.mean(0) for c,z in zip(weights,groups))
                tests.append(dict(cohort=cohort, metric=metric, effect=effect, family=family,
                                  nominal_anova=nominal, heteroscedastic_wts=robust,
                                  contrast_mean=list(map(float,beta)),
                                  contrast_rms_per_cell=float(np.sqrt(np.sum(beta**2)/(len(MODELS)*len(METHODS))))))
            mean = [y.mean(0) for y in ys]
            for k, model in enumerate(MODELS):
                for m, method in enumerate(METHODS):
                    delta = mean[0][k,m] - mean[1][k,m]
                    ci = interval(bootstrap[0][:,k,m,j] - bootstrap[1][:,k,m,j])
                    iid_ci = interval(iid_bootstrap[0][:,k,m,j] - iid_bootstrap[1][:,k,m,j])
                    deltas.append(dict(cohort=cohort, metric=metric, model=model, method=method,
                        mean_imagenet=float(mean[0][k,m]), mean_voc=float(mean[1][k,m]),
                        difference_imagenet_minus_voc=float(delta), conditional_ci95=ci, iid_image_ci95=iid_ci,
                        interpretation="nominal observed-score task difference, not a common latent-quality effect"))
            print(f"Mixed factorial: {cohort}/{metric}, 7 effects", flush=True)
    for family in sorted({r["family"] for r in tests}):
        selected = [r for r in tests if r["family"] == family]
        require(len(selected) == (14 if family.endswith("mixed_quality") else 7), "prespecified mixed family size")
        for field in ("nominal_anova", "heteroscedastic_wts"):
            ps = [r[field]["p_value"] if r[field]["p_value"] is not None else 1. for r in selected]
            for r,p in zip(selected,holm(ps)):
                r[field]["p_holm"] = float(p) if r[field]["p_value"] is not None else None
                r[field]["family_size"] = len(selected)
    return dict(cohorts=cohorts, tests=tests, observed_dataset_differences=deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("analysis/dcu_20261029/mixed_factorial.json"))
    parser.add_argument("--bootstrap", type=int, default=9999)
    parser.add_argument("--ci-resamples", type=int, default=10000)
    args = parser.parse_args()
    require(args.bootstrap >= 999 and args.ci_resamples >= 1000, "production resampling too small")
    frames, receipt, classes = validate(args.handoff, args.project)
    data = analyze(frames, classes, args.bootstrap, args.ci_resamples)
    data.update(receipt=receipt, protocol=dict(seed=SEED, dataset_between=DATASETS, model_within=MODELS,
        method_within=METHODS, main_cohort="common_correct", sensitivity="all", p_bootstrap=args.bootstrap,
        ci_resamples=args.ci_resamples, nominal="equal-cell Type III mixed ANOVA; subject nested in dataset; effect-specific subject-within-dataset error; pooled GG epsilon",
        robust="heteroscedastic WTS with independent group Gaussian parametric bootstrap; each bootstrap covariance re-estimated using normal/Wishart sufficient statistics; plus-one correction; no pseudoinverse or jitter to conceal singularities",
        primary_inference="report nominal ANOVA and heteroscedastic WTS together; 14 quality tests per cohort; recorded log1p(ms) separate exploratory 7-test family; Holm separately for each procedure",
        metric_scales={METRICS[0]:"raw target-probability MoRF AUC", METRICS[1]:"input-noise Spearman", METRICS[2]:"log1p(recorded wrapper milliseconds)"},
        contrast="dataset=ImageNet minus VOC; within main effects average datasets EQUALLY, regardless of n; dataset interactions difference of corresponding within contrasts",
        uncertainty="independent image bootstrap in each dataset, paired over models/methods/metrics; class-stratified and IID-image percentile marginal CIs; singleton-class and fixed-checkpoint limitations retained",
        assumptions="independent original-image vectors, finite fourth moments and positive definite group contrast covariance for WTS; bootstrap is asymptotic, not an exact finite-sample test; nominal ANOVA additionally uses pooled covariance and GG approximation",
        scope="all 7 factorial terms computed for nominal raw task scores, not causal dataset effects or a common quality scale. Softmax/sigmoid, class space, image population, preprocessing and shared two-round timing confounds remain. Teacher/A/C/E acceptance and final freeze are separate."),
        environment=dict(python=platform.python_version(), **{n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","PyYAML")}),
        analysis_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print("Saved " + str(args.output))


if __name__ == "__main__":
    main()
