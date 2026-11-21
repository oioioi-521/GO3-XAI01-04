"""Mixed-design reference and heteroscedastic bootstrap tests (not course data)."""
import numpy as np
import pytest
from scipy import stats

from analysis.dcu_mixed_factorial import analyze, cube, mixed_anova, parametric_wald
from analysis.dcu_repeated_measures import effect_coordinates
from analysis.validate_dcu_team_handoff import METRICS
from tests.test_dcu_repeated_measures import synthetic_frames


def reference_groups():
    # Public StataCorp example, Winer table 7.13: period-major/dial-minor.
    a = np.array([[45,53,60,40,52,57,28,37,46], [35,41,50,30,37,47,25,32,41],
                  [60,65,75,58,54,70,40,47,50]], float).reshape(3,3,3)
    b = np.array([[50,48,61,25,34,51,16,23,35], [42,45,55,30,37,43,22,27,37],
                  [56,60,77,40,39,57,31,29,46]], float).reshape(3,3,3)
    return [a,b]


def test_stata_seven_factorial_effects_and_error_strata():
    y = reference_groups()
    c = [effect_coordinates(a) for a in y]
    grand = [3*a.mean((1,2))[:,None] for a in y]
    cases = [("dataset",grand,(1,-1),468.1666667,2491.1111111,1.,.4348),
             ("model",[z['model'] for z in c],(.5,.5),3722.3333333,234.8888889,.6476,.0003),
             ("dataset_model",[z['model'] for z in c],(1,-1),333.,234.8888889,.6476,.0569),
             ("method",[z['method'] for z in c],(.5,.5),2370.3333333,105.5555556,.9171,None),
             ("dataset_method",[z['method'] for z in c],(1,-1),50.3333333,105.5555556,.9171,.2152),
             ("method_model",[z['method_model'] for z in c],(.5,.5),10.6666667,127.1111111,.5134,.7295),
             ("dataset_method_model",[z['method_model'] for z in c],(1,-1),11.3333333,127.1111111,.5134,.7156)]
    for _, groups, weights, ss, error, eps, p in cases:
        r = mixed_anova(groups,weights)
        assert r['ss_hypothesis'] == pytest.approx(ss,abs=1e-6)
        assert r['ss_error'] == pytest.approx(error,abs=1e-6)
        assert r['epsilon_gg'] == pytest.approx(eps,abs=5e-5)
        if p is not None:
            assert r['p_value'] == pytest.approx(p,abs=5e-5)


def test_unbalanced_type_three_uses_equal_dataset_weights():
    rng=np.random.default_rng(91)
    groups=[rng.normal(size=(13,2))+.2,rng.normal(size=(37,2))-.1]
    x=np.column_stack([np.ones(50), np.r_[np.ones(13),-np.ones(37)]])
    y=np.vstack(groups)
    beta=np.linalg.lstsq(x,y,rcond=None)[0]
    variance=np.linalg.inv(x.T@x)[0,0]
    r=mixed_anova(groups,(.5,.5))
    assert r['ss_hypothesis'] == pytest.approx(float(beta[0]@beta[0]/variance))
    assert r['ss_error'] == pytest.approx(float(np.sum((y-x@beta)**2)))
    assert not np.allclose(beta[0],y.mean(0))


def test_scalar_mixed_anova_is_pooled_t_and_wts_is_welch_t_squared():
    rng=np.random.default_rng(29)
    a=rng.normal(size=(40,1));b=rng.normal(size=(73,1))*3+.4
    nominal=mixed_anova([a,b],(1,-1))
    robust=parametric_wald([a,b],(1,-1),iterations=2999,seed=3)
    pooled=stats.ttest_ind(a[:,0],b[:,0],equal_var=True)
    welch=stats.ttest_ind(a[:,0],b[:,0],equal_var=False)
    assert nominal['F']==pytest.approx(pooled.statistic**2)
    assert nominal['p_value']==pytest.approx(pooled.pvalue)
    assert robust['WTS']==pytest.approx(welch.statistic**2)
    assert abs(robust['p_value']-welch.pvalue)<.035
    assert robust['p_value']>0


def test_wts_scale_invariance_and_reproducibility():
    rng=np.random.default_rng(109)
    groups=[rng.normal(size=(30,3)),rng.normal(size=(45,3))*[1,2,3]+.2]
    first=parametric_wald(groups,(1,-1),iterations=199,seed=71)
    assert first==parametric_wald(groups,(1,-1),iterations=199,seed=71)
    scaled=parametric_wald([g*[.001,1000,4] for g in groups],(1,-1),iterations=199,seed=71)
    assert first['WTS']==pytest.approx(scaled['WTS'])
    assert first['p_value']==scaled['p_value']


def test_singular_covariance_not_repaired_with_jitter_or_hidden_dropped_df():
    x=np.arange(20.)
    r=parametric_wald([np.c_[x,x],np.c_[x,x]],(1,-1),iterations=99)
    assert r['status']=='singular_or_insufficient_group_covariance'
    assert r['WTS'] is None and r['p_value'] is None


def test_two_dataset_keys_do_not_pair_equal_numeric_image_ids():
    frames,classes=synthetic_frames()
    for name,frame in list(frames.items()):
        timing=frame[frame.metric.eq(METRICS[0])].copy()
        timing['metric']=METRICS[2]
        timing['value']=np.exp(np.abs(timing['value']))
        import pandas as pd
        frames[name]=pd.concat([frame,timing],ignore_index=True)
    result=analyze(frames,classes,iterations=99,resamples=40)
    assert len(result['tests'])==42
    assert len(result['observed_dataset_differences'])==108
    assert [r['n'] for r in result['cohorts']]==[12,12,12,12]
    assert {r['effect'] for r in result['tests']}=={'dataset','method','model','method_model','dataset_method','dataset_model','dataset_method_model'}
    for r in result['tests']:
        assert r['nominal_anova']['family_size']==(7 if r['metric']==METRICS[2] else 14)
        assert r['heteroscedastic_wts']['p_holm']>=r['heteroscedastic_wts']['p_value']
    with pytest.raises(ValueError,match='incomplete'):
        cube(frames['all'].iloc[1:],'imagenet')
