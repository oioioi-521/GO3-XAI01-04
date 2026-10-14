"""Independently recompute map equivalence and timing summaries of a probe."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, previous_high=None):
    jobs = [f'{m}-{d}' for m in ('resnet50', 'densenet121', 'vgg16') for d in ('imagenet', 'voc')]
    completed = json.loads((root/'COMPLETE.json').read_text())
    assert completed['status'] == 'probe_complete' and not completed['formal_started']
    for path in (root/'preflight.json', root/'postrun/preflight.json'):
        assert json.loads(path.read_text())['status'] == 'passed'
    result = {'integrity': 'passed', 'units': {}, 'maps_verified': 0, 'pairs_verified': 0,
              'saved_high_budget_anchors_verified': 0}
    for job in jobs:
        unit = root/job
        manifest = json.loads((unit/'manifest.json').read_text())
        summary = json.loads((unit/'summary.json').read_text())
        records = json.loads((unit/'records.json').read_text())
        comparisons = json.loads((unit/'comparisons.json').read_text())
        assert summary['records'] == records and summary['comparisons'] == comparisons
        assert summary['state'] == 'complete' and not summary['formal_started']
        lookup, maps = {}, {}
        for row in records:
            key = (row['case'], row['budget'], row['seed'], row['variant'])
            assert key not in lookup
            path = unit/row['map_path']
            assert path.parent == unit and sha(path) == row['map_sha256']
            a = np.load(path, allow_pickle=False)
            assert a.shape == (224,224) and a.dtype == np.float32 and np.isfinite(a).all()
            assert 0 <= a.min() <= a.max() <= 1 and row['seconds'] > 0
            lookup[key] = row; maps[key] = a; result['maps_verified'] += 1
        # Reference and same-batch prototype are mandatory even if large batches OOM.
        for index, image_id in enumerate(manifest['selected']):
            for budget in (2000,4000,8000):
                for variant in ('reference32','vector32'):
                    assert (image_id,budget,42+index,variant) in lookup
                if previous_high is not None:
                    old_unit = previous_high/f'lime-{job}'
                    old = next(r for r in (json.loads(p.read_text()) for p in (old_unit/'records').glob('*.json')) if r['image_id']==image_id)
                    saved = next(r for r in old['runs'] if r['budget']==budget and r['seed']==42+index)
                    path=old_unit/saved['map_path']
                    assert sha(path)==saved['map_sha256']
                    assert np.array_equal(np.load(path,allow_pickle=False),maps[image_id,budget,42+index,'reference32'])
                    result['saved_high_budget_anchors_verified'] += 1
        for repeat in range(5):
            case = f"noise{repeat}_{manifest['selected'][0]}"
            for variant in ('reference32','vector32'): assert (case,2000,42,variant) in lookup
        seen, exact, passed = set(), {}, {}
        max_error, minimum_rho = {}, {}
        for c in comparisons:
            key = (c['case'],c['budget'],c['seed'],c['variant'])
            reference = (*key[:3], 'reference32')
            assert key not in seen; seen.add(key)
            a, b = maps[reference], maps[key]
            delta = a.astype(np.float64)-b.astype(np.float64)
            maximum, rmse = float(abs(delta).max()),float(np.sqrt((delta**2).mean()))
            r2_abs = abs(lookup[reference]['diagnostics']['weighted_r2'] - lookup[key]['diagnostics']['weighted_r2'])
            rho = None if np.ptp(a)<=1e-12 or np.ptp(b)<=1e-12 else float(spearmanr(a.ravel(),b.ravel()).statistic)
            exact_value = bool(np.array_equal(a,b))
            numeric = maximum<=1e-5 and rmse<=1e-6 and r2_abs<=1e-6
            assert exact_value == c['exact'] and numeric == c['numeric_pass']
            assert maximum == c['max_abs'] and rmse == c['rmse'] and r2_abs == c['r2_abs']
            assert (rho is None and c['spearman'] is None) or np.isclose(rho, c['spearman'], rtol=0, atol=1e-12)
            assert np.isclose(c['speedup'],lookup[reference]['seconds']/lookup[key]['seconds'])
            variant = c['variant']
            exact[variant] = exact.get(variant,0)+int(exact_value)
            passed[variant] = passed.get(variant,0)+int(numeric)
            max_error[variant] = max(max_error.get(variant,0),maximum)
            if rho is not None: minimum_rho[variant] = min(minimum_rho.get(variant,1),rho)
            result['pairs_verified'] += 1
        assert len(comparisons) == len([k for k in lookup if k[-1]!='reference32'])
        times = {}
        for budget in (2000,4000,8000):
            times[str(budget)] = {}
            for variant in ('reference32','vector32','vector64','vector128'):
                values=[r['seconds'] for r in records if r['case'] in manifest['selected'] and r['budget']==budget and r['variant']==variant]
                if values: times[str(budget)][variant] = dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)))
        result['units'][job] = dict(records=len(records),pairs=len(comparisons),exact_pairs=exact,numeric_pass_pairs=passed,
            maximum_map_error=max_error, minimum_spearman=minimum_rho,times=times,
            same_batch_noise_seconds={v:float(np.mean([r['seconds'] for r in records if r['case'].startswith('noise') and r['variant']==v])) for v in ('reference32','vector32')},
            matches_saved_high_budget=sum(r.get('matches_saved_high_budget',False) for r in records),
            optional_variant_failures=summary['failures'],summary_sha256=sha(unit/'summary.json'),wall_seconds=summary['wall_seconds'])
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-high',type=Path)
    args=p.parse_args();result=audit(args.root,args.previous_high)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='units'}))
