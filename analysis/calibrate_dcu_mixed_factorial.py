"""Synthetic heteroscedastic WTS diagnostics; no experiment observations used."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .dcu_mixed_factorial import parametric_wald


def calibrate(trials=300, iterations=499, seed=20261029):
    rng=np.random.default_rng(seed)
    rows=[]
    for scenario in ("unequal_covariance_gaussian_null", "unequal_covariance_exponential_null", "injected_group_contrast"):
        hit=0
        for i in range(trials):
            if scenario == "unequal_covariance_exponential_null":
                a=rng.exponential(size=(60,3))-1
                b=(rng.exponential(size=(90,3))-1)*[3,1,2]
            else:
                a=rng.normal(size=(60,3))
                b=rng.normal(size=(90,3))*[3,1,2]
                if scenario == "injected_group_contrast":
                    b += [2,1,2]
            r=parametric_wald([a,b],(1,-1),iterations,seed+i)
            hit += r['p_value'] < .05
        ci=stats.binomtest(hit,trials).proportion_ci()
        rows.append(dict(scenario=scenario,trials=trials,n_by_group=[60,90],dimension=3,
                         bootstrap_iterations=iterations,rejected=hit,rate=hit/trials,
                         binomial_ci95=[float(ci.low),float(ci.high)]))
    return dict(seed=seed,alpha=.05,diagnostics=rows,broad_alarm_passed=all(x['rate']<=.10 for x in rows[:2]) and rows[2]['rate']>=.9,
                analysis_code_sha256=hashlib.sha256(Path(__file__).with_name('dcu_mixed_factorial.py').read_bytes()).hexdigest(),
                limits="Finite synthetic diagnostics for three-dimensional group contrasts only; not a proof for all seven hypotheses, real-data assumptions, exact finite-sample level, or teacher acceptance.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('analysis/dcu_20261029/mixed_synthetic_diagnostics.json'))
    args=parser.parse_args()
    data=calibrate()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    if not data['broad_alarm_passed']:
        raise RuntimeError('synthetic WTS diagnostic alarm; inspect results')
    print(json.dumps(data))


if __name__=='__main__':
    main()
