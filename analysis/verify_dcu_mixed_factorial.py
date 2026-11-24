"""Independent QR-basis / grouped-OLS audit of the frozen mixed-design output.

Does not call the mixed ANOVA, WTS, projection, or Holm implementations under
audit. Uses the existing input validator only to recover frozen observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .validate_dcu_team_handoff import DATASETS, METHODS, METRICS, MODELS, validate


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=2e-10)


def audit(handoff, project, result):
    frames, receipt, _ = validate(handoff, project)
    data = json.loads(result.read_text(encoding="utf-8"))
    # QR of simple differences supplies a distinct basis from Helmert.
    q3 = np.linalg.qr(np.eye(3)[:, 1:] - np.eye(3)[:, :1])[0]
    q6 = np.linalg.qr(np.eye(6)[:, 1:] - np.eye(6)[:, :1])[0]
    bases = dict(dataset=np.ones((18,1))/np.sqrt(18),
                 method=np.kron(np.ones((3,1))/np.sqrt(3),q6),
                 model=np.kron(q3,np.ones((6,1))/np.sqrt(6)), method_model=np.kron(q3,q6))
    count = 0
    for cohort in ("common_correct", "all"):
        for metric in METRICS:
            groups = []
            for dataset in DATASETS:
                f = frames[cohort]
                sub = f[f.dataset.eq(dataset) & f.metric.eq(metric)]
                wide = sub.pivot(index="image_id", columns=["model","method"], values="value")
                y = wide.reindex(columns=pd.MultiIndex.from_product([MODELS,METHODS])).sort_index().to_numpy(float,copy=True)
                if metric == METRICS[2]:
                    y = np.log1p(y)
                groups.append(y)
            rows = [r for r in data["tests"] if r["cohort"] == cohort and r["metric"] == metric]
            assert len(rows) == 7
            for r in rows:
                effect = r["effect"]
                between = effect == "dataset" or effect.startswith("dataset_")
                base = effect if effect == "dataset" else effect.removeprefix("dataset_")
                z = [y @ bases[base] for y in groups]
                ns = [len(a) for a in z]
                x = np.zeros((sum(ns),2))
                x[:ns[0],0], x[ns[0]:,1] = 1,1
                joined = np.concatenate(z)
                coefficients = np.linalg.lstsq(x,joined,rcond=None)[0]
                residual = joined - x @ coefficients
                w = np.array([1,-1] if between else [.5,.5])
                b = w @ coefficients
                scatter = residual.T @ residual
                d, ndf = len(b), len(joined)-2
                h = float((b @ b)/(w @ np.linalg.solve(x.T@x,w)))
                error = float(np.trace(scatter))
                f = h*ndf/error
                eps = float(np.clip(np.trace(scatter)**2/(d*np.sum(scatter**2)),1/d,1))
                nominal = r["nominal_anova"]
                close([h,error,f,eps,h/(h+error),d*eps,ndf*d*eps],
                      [nominal[k] for k in ("ss_hypothesis","ss_error","F","epsilon_gg","partial_eta_squared","df1_gg","df2_gg")])
                close(max(1e-300,float(stats.f.sf(f,d*eps,ndf*d*eps))),nominal["p_value"])
                cov = [np.atleast_2d(np.cov(a,rowvar=False,ddof=1)) for a in z]
                v = sum(wi*wi*s/n for wi,s,n in zip(w,cov,ns))
                wts = float(b @ np.linalg.solve(v,b))
                robust = r["heteroscedastic_wts"]
                close(wts,robust["WTS"])
                assert robust["covariance_ranks"] == [d,d]
                close((robust["bootstrap_exceedances"]+1)/(robust["bootstrap_iterations"]+1),robust["p_value"])
                count += 1
            means = [g.mean(0).reshape(3,6) for g in groups]
            for r in data["observed_dataset_differences"]:
                if r["cohort"] != cohort or r["metric"] != metric:
                    continue
                i,j = MODELS.index(r["model"]), METHODS.index(r["method"])
                close([means[0][i,j],means[1][i,j],means[0][i,j]-means[1][i,j]],
                      [r["mean_imagenet"],r["mean_voc"],r["difference_imagenet_minus_voc"]])
                for key in ("conditional_ci95","iid_image_ci95"):
                    assert np.isfinite(r[key]).all() and r[key][0] <= r[key][1]
    for family in {r["family"] for r in data["tests"]}:
        rows = [r for r in data["tests"] if r["family"] == family]
        for procedure in ("nominal_anova","heteroscedastic_wts"):
            order = sorted(rows,key=lambda r:r[procedure]["p_value"])
            running = 0
            for i,r in enumerate(order):
                running = max(running,min(1.,(len(order)-i)*r[procedure]["p_value"]))
                close(running,r[procedure]["p_holm"])
    assert count == 42 and len(data["observed_dataset_differences"]) == 108
    assert data["analysis_code_sha256"] == hashlib.sha256((project/"analysis/dcu_mixed_factorial.py").read_bytes()).hexdigest()
    assert receipt == data["receipt"]
    return dict(status="passed", factorial_rows_independently_recomputed=count,
        observed_dataset_cell_differences_checked=108, full_rank_group_covariances_checked=84,
        input_receipt_identical=True, raw_observations_modified=False,
        algorithm="independent simple-difference QR basis, group-dummy least squares, raw covariance WTS; separate Holm reconstruction",
        checks="SS_H/error/F/GG epsilon/GG dfs/partial eta squared/nominal p/observed WTS/plus-one bootstrap arithmetic/108 means/CI order/source SHA",
        limits="Does not independently repeat every bootstrap draw or validate substantive dataset comparability; not C/teacher acceptance.",
        input=receipt, result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
        verification_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handoff",type=Path)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result",type=Path,default=Path("analysis/dcu_20261029/mixed_factorial.json"))
    args=parser.parse_args()
    receipt=audit(args.handoff,args.project,args.result)
    (args.result.parent/"validation_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print("Passed: 42 independent factorial rows, 108 cell differences, all families")


if __name__ == "__main__":
    main()
