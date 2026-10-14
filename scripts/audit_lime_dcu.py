"""Verify the independent LIME pilot's downloaded evidence and extrapolate cost."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.stability import stable_seed, stability_similarity
from preprocessing.dataset import get_split


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_run(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    args = manifest['args']
    summary = json.loads((path / 'summary.json').read_text())
    assert summary['status'] == 'complete'
    assert manifest['metadata_sha256'] == digest(ROOT / 'data/metadata.csv')
    assert manifest['script_sha256'] == digest(ROOT / 'scripts/benchmark_lime_dcu.py')
    workers = [json.loads(p.read_text()) for p in sorted(path.glob('worker-*/result.json'))]
    assert len(workers) == args['workers']
    frozen = get_split(args['dataset'], 'debug').iloc[:args['images']]
    targets = dict(zip(frozen.image_id.astype(str), frozen.class_id.astype(int)))
    rows = [row for worker in workers for row in worker['rows']]
    assert rows == summary['rows']
    expected = {(image, cycle, repeat) for image in targets for cycle in range(args['cycles'])
                for repeat in range(-1, args['noise_repeats'])}
    assert {(r['image_id'], r['cycle'], r['repeat']) for r in rows} == expected
    assert len(rows) == len(expected) == summary['calls']
    assert len(list(path.glob('worker-*/*.npy'))) == len(rows)
    devices = args['devices'].split(',')
    maps = {}
    for worker in workers:
        env = worker['environment']
        assert env['device_count'] == 1 and env['hip']
        assert env['visible_devices']['HIP_VISIBLE_DEVICES'] == devices[worker['rank']]
        assert env['visible_devices']['CUDA_VISIBLE_DEVICES'] == devices[worker['rank']]
        if args['dataset'] == 'voc':
            checkpoints = json.loads((ROOT/'docs/VOC20_CHECKPOINT_MANIFEST.json').read_text())['checkpoints']
            checkpoint = next(c for c in checkpoints if c['model'] == args['model'])
            assert env['checkpoint_sha256'] == checkpoint['sha256']
    for row in rows:
        filename = path / row['map_file']
        assert digest(filename) == row['map_sha256'], filename
        array = np.load(filename, allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (224,224)
        assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1
        key = row['image_id'], row['cycle'], row['repeat']
        maps[key] = array
        assert row['target'] == targets[row['image_id']]
        assert row['prediction_samples'] == args['samples']
        phases = ['segmentation_ms','prediction_ms','regression_ms','generation_and_other_ms']
        assert all(np.isfinite(row[k]) and row[k] >= 0 for k in phases)
        assert np.isclose(sum(row[k] for k in phases), row['attribution_ms'])
        if row['repeat'] == -1:
            if args['morf']:
                assert np.isfinite(row['morf_auc']) and row['morf_ms'] > 0
        else:
            assert row['noise_seed'] == stable_seed(args['dataset'],row['image_id'],row['repeat'])
    for row in rows:
        if row['repeat'] >= 0:
            candidate = maps[(row['image_id'],row['cycle'],row['repeat'])]
            reference = maps[(row['image_id'],row['cycle'],-1)]
            similarity = stability_similarity(candidate,reference)
            assert similarity.status == row['stability_status']
            assert np.isclose(similarity.score,row['stability_score'],rtol=1e-9,atol=1e-10)
    wall = max(w['ended'] for w in workers) - min(w['started'] for w in workers)
    assert np.isclose(wall,summary['steady_wall_seconds'],rtol=1e-12)
    assert np.isclose(len(rows)/wall,summary['calls_per_second'],rtol=1e-12)
    for key,value in summary['means'].items():
        assert np.isclose(value,np.mean([r[key] for r in rows]),rtol=1e-12)
    clean = {k:v for k,v in summary.items() if k != 'rows'}
    times = np.array([r['attribution_ms'] for r in rows])
    clean['p50_ms'],clean['p95_ms'],clean['max_ms'] = map(float,np.percentile(times,[50,95,100]))
    clean['constant_maps'] = sum(r['constant'] for r in rows)
    clean['superpixels_range'] = [min(r['superpixels'] for r in rows),max(r['superpixels'] for r in rows)]
    clean['mean_morf_ms'] = float(np.mean([r['morf_ms'] for r in rows if 'morf_ms' in r])) if args['morf'] else None
    if args['noise_repeats'] == 5:
        clean['extrapolated_460_images_hours'] = wall / args['images'] * 460 / 3600
    return clean, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args = parser.parse_args()
    assert (args.root/'COMPLETE').read_text().strip() == 'completed'
    paths = sorted(args.root.glob('*/summary.json'))
    expected = {f'{model}-{dataset}-8card' for model in ('resnet50','densenet121','vgg16')
                for dataset in ('imagenet','voc')} | {'scaling-1','scaling-8'}
    assert {p.parent.name for p in paths} == expected
    runs, maps = {}, {}
    for path in paths:
        runs[path.parent.name],maps[path.parent.name] = audit_run(path.parent)
    a,b = maps['scaling-1'],maps['scaling-8']
    assert a.keys() == b.keys()
    correlations, differences = [], []
    for key in a:
        differences.append(float(np.abs(a[key]-b[key]).max()))
        if np.ptp(a[key]) and np.ptp(b[key]):
            correlations.append(float(spearmanr(a[key].ravel(),b[key].ravel()).statistic))
    scaling = dict(steady_speedup=runs['scaling-1']['steady_wall_seconds']/runs['scaling-8']['steady_wall_seconds'],
                   total_speedup=runs['scaling-1']['total_wall_seconds']/runs['scaling-8']['total_wall_seconds'],
                   max_map_abs=max(differences),min_spearman=min(correlations) if correlations else None)
    hours = sum(r.get('extrapolated_460_images_hours',0) for r in runs.values())
    report = dict(status='verified',runs=runs,scaling=scaling,
                  measured_calls=sum(r['calls'] for r in runs.values()),
                  full_lime_16560_calls_hours=hours,
                  full_lime_hours_with_25_percent_margin=hours*1.25,
                  limitations='8 debug images per cell, not formal eval; official LIME candidate, not E adapter')
    (args.root/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
