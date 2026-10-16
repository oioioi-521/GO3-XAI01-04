"""Independently verify benchmark coverage/maps and compare scaling and batching."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def audit_run(root):
    summary = json.loads((root / 'summary.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    args = manifest['arguments']
    workers = [json.loads(p.read_text()) for p in sorted(root.glob('worker-*/result.json'))]
    assert len(workers) == args['workers'], root
    assert sorted(i for w in workers for i in w['indices']) == list(range(args['images']))
    rows = [r for w in workers for r in w['rows']]
    assert rows == summary['rows'] or sorted(rows, key=lambda r:(r['image_id'],r['repeat'])) == sorted(summary['rows'], key=lambda r:(r['image_id'],r['repeat']))
    assert len(rows) == args['images'] * args['repeats']
    assert len({(r['image_id'],r['repeat']) for r in rows}) == len(rows)
    for row in rows:
        p = root / row['map_file']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['map_sha256'], p
        a = np.load(p, allow_pickle=False)
        assert a.dtype == np.float32 and a.shape == (224,224) and np.isfinite(a).all(), p
        assert np.isfinite(row['attribution_ms']) and row['attribution_ms'] > 0
        if 'morf_auc' in row:
            assert np.isfinite(row['morf_auc'])
    wall = max(w['ended'] for w in workers) - min(w['started'] for w in workers)
    assert np.isclose(wall, summary['steady_wall_seconds'], rtol=1e-12)
    assert np.isclose(len(rows) / wall, summary['calls_per_second'], rtol=1e-12)
    assert np.isclose(np.mean([r['attribution_ms'] for r in rows]), summary['mean_attribution_ms'])
    return summary


def compare_maps(left_root, right_root):
    left = audit_run(left_root)
    right = audit_run(right_root)
    for field in ('method','model','dataset','images','calls'):
        assert left[field] == right[field], field
    a = {(r['image_id'],r['repeat']):r for r in left['rows']}
    b = {(r['image_id'],r['repeat']):r for r in right['rows']}
    assert a.keys() == b.keys(), 'incompatible sample identity'
    details=[]
    for key in sorted(a):
        assert a[key]['target'] == b[key]['target']
        x = np.load(left_root / a[key]['map_file']).astype(np.float64)
        y = np.load(right_root / b[key]['map_file']).astype(np.float64)
        constant = np.ptp(x) == 0 or np.ptp(y) == 0
        details.append(dict(image_id=key[0], repeat=key[1],
                            max_abs=float(np.max(np.abs(x-y))),
                            mean_abs=float(np.mean(np.abs(x-y))),
                            spearman=None if constant else float(spearmanr(x.ravel(),y.ravel()).statistic),
                            constant=bool(constant)))
    correlations=[r['spearman'] for r in details if r['spearman'] is not None]
    return dict(left=left_root.name, right=right_root.name,
                steady_speedup=right['calls_per_second']/left['calls_per_second'],
                total_speedup=left['total_wall_seconds']/right['total_wall_seconds'],
                max_map_abs=max(r['max_abs'] for r in details),
                min_spearman=min(correlations) if correlations else None,
                details=details)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--cpu-reference',type=Path)
    args=p.parse_args()
    if not (args.root / 'COMPLETE').exists():
        raise ValueError('batch has not completed')
    runs={f.parent.name:audit_run(f.parent) for f in sorted(args.root.glob('*/summary.json'))}
    cards=runs['stress']['workers']
    expected={f'gate-{model}-{method}' for model in ('resnet50','densenet121','vgg16')
              for method in ('gradcam','ig','occlusion','kernelshap','rise')}
    expected.update(f'{method}-{count}card' for method in ('kernelshap','rise','kernelshap-b128')
                    for count in (1,cards))
    expected.update(('stress','sustained-stress'))
    assert set(runs) == expected, ('missing or unexpected runs', expected.symmetric_difference(runs))
    comparisons=[]
    for method in ('kernelshap','rise'):
        comparisons.append(compare_maps(args.root / f'{method}-1card',args.root / f'{method}-{cards}card'))
    comparisons.append(compare_maps(args.root/'kernelshap-1card',args.root/'kernelshap-b128-1card'))
    comparisons.append(compare_maps(args.root/'kernelshap-b128-1card',args.root/f'kernelshap-b128-{cards}card'))
    if args.cpu_reference:
        comparisons.append(compare_maps(args.cpu_reference,args.root/'gate-resnet50-gradcam'))
    report=dict(status='verified',runs={name:{k:v for k,v in s.items() if k!='rows'} for name,s in runs.items()},comparisons=comparisons)
    (args.root/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))


if __name__=='__main__':
    main()
