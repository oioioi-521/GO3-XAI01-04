"""Compare LIME budgets on exactly the frozen 40 debug images; never run eval.

Before execution, run --plan-only and inspect plan.json. The plan reports missing
assets and computes a forward-pass count, not an invented time estimate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run_unit import (_load_config, _resolve, _config_hash,
    _select_device, _synchronize, _black_baseline, _build_attributor)
from experiments.predictions import build_prediction_context
from models import load_model
from preprocessing.dataset import MetadataDataset
from experiments.lime_gate_integrity import (
    SCHEMA, IDENTITY_KEYS, asset_identity, check_sample, existing_records,
    file_sha, output_lock, read_json, require, source_path, validate_record,
)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def compare_maps(a, b):
    if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return {'status': 'constant_map', 'spearman': None, 'rmse': float(np.sqrt(np.mean((a-b)**2)))}
    value = float(spearmanr(a.ravel(), b.ravel()).statistic)
    return {'status': 'valid' if np.isfinite(value) else 'undefined',
            'spearman': value if np.isfinite(value) else None,
            'rmse': float(np.sqrt(np.mean((a-b)**2)))}


def execution_identity(config):
    """Pin this reference gate's code and machine/device, not historical runs."""
    source = Path(__file__).resolve().parents[1]
    files = ('experiments/lime_budget_gate.py', 'experiments/lime_gate_integrity.py',
             'experiments/attribution/lime.py', 'experiments/attribution/gradient_common.py',
             'experiments/run_unit.py', 'experiments/predictions.py',
             'preprocessing/dataset.py', 'models/factory.py')
    # Normalize text line endings so identical Git source on Windows/Linux has
    # the same code identity. Machine/device identity is deliberately separate.
    code = {name: hashlib.sha256((source / name).read_text(encoding='utf-8').encode()).hexdigest()
            for name in files}
    device = _select_device(config['runtime']['device'])
    identity = dict(machine_sha256=hashlib.sha256(platform.node().encode()).hexdigest(),
                    platform=platform.platform(), python=platform.python_version(),
                    device=str(device), cuda=torch.version.cuda, hip=torch.version.hip,
                    threads=torch.get_num_threads(), code_sha256=code,
                    thread_environment={k: os.environ.get(k) for k in
                                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                         'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')})
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        identity.update(device_name=properties.name, device_memory_bytes=properties.total_memory,
                        device_uuid=str(getattr(properties, 'uuid', 'unavailable')))
    return identity


def plan(config, budgets, seeds, output):
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    cfg = copy.deepcopy(config)
    if cfg['method'] != 'lime':
        raise ValueError('budget gate only accepts lime')
    if (any(type(b) is not int for b in budgets) or any(type(s) is not int for s in seeds)
            or len(set(budgets)) < 2 or min(budgets) < 2 or len(set(seeds)) < 3):
        raise ValueError('use at least two distinct budgets and three distinct seeds')
    if any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError('seeds must fit uint32')
    cfg['dataset']['split'] = 'debug'
    cfg['runtime']['max_images'] = 40
    dataset = MetadataDataset(cfg['dataset']['name'], split='debug')
    if len(dataset) != 40 or not dataset.rows.image_id.is_unique:
        raise ValueError('expected exactly 40 unique frozen debug images')
    frozen = json.loads((ROOT / 'data/DATA_VERSION.json').read_text())
    metadata_sha = hashlib.sha256((ROOT / 'data/metadata.csv').read_bytes()).hexdigest()
    if metadata_sha != frozen['sha256']['metadata.csv']:
        raise ValueError('metadata.csv does not match frozen DATA_VERSION.json')
    missing, assets = [], []
    for index, row in enumerate(dataset.rows.itertuples()):
        path = source_path(ROOT, row.image_path)
        if not path.is_file():
            missing.append(str(path))
            assets.append(None)
            continue
        before = file_sha(path)
        asset = asset_identity(ROOT, row, dataset[index])
        require(asset['source_image_sha256'] == before, 'source changed while preprocessing')
        assets.append(asset)
    checkpoint = cfg['model'].get('checkpoint')
    if checkpoint and not _resolve(checkpoint).is_file():
        missing.append(str(_resolve(checkpoint)))
    context = None
    if not checkpoint or _resolve(checkpoint).is_file():
        try:
            context = build_prediction_context(dataset=cfg['dataset']['name'], split='debug',
                model_config=cfg['model'], checkpoint=_resolve(checkpoint) if checkpoint else None)
        except FileNotFoundError as error:
            missing.append(str(error))
    versions = {p: importlib.metadata.version(p) for p in
                ('torch', 'torchvision', 'lime', 'scikit-image', 'scikit-learn', 'scipy', 'numpy')}
    identity = dict(config=cfg, budgets=sorted(set(budgets)), seeds=sorted(set(seeds)),
                    metadata_sha256=metadata_sha, prediction_context=context,
                    image_ids=dataset.rows.image_id.tolist(), versions=versions,
                    gate_schema_version=SCHEMA, image_assets=assets,
                    execution_identity=execution_identity(cfg))
    digest = _config_hash(identity)
    report = dict(**identity, config_hash=digest, output=str(output), missing_assets=missing,
                  expected_images=40, attribution_calls=41*len(set(seeds))*len(set(budgets)),
                  masked_forwards=41*len(set(seeds))*sum(set(budgets)),
                  expected_duration='unknown until real single-image timing on the chosen machine',
                  formal_budget_status='pending review; no automatic freezing')
    return cfg, dataset, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[1000, 2000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--single-image-ms', type=float,
                        help='measured attribution ms at the smallest budget, same machine/model')
    args = parser.parse_args()
    with output_lock(_resolve(str(args.output))):
        run_gate(args)


def run_gate(args):
    config = _load_config(_resolve(str(args.config)))
    output = _resolve(str(args.output))
    cfg, dataset, report = plan(config, args.budgets, args.seeds, output)
    if args.single_image_ms is not None:
        if not np.isfinite(args.single_image_ms) or args.single_image_ms <= 0:
            raise ValueError('single-image-ms must be finite and positive')
        report['estimated_attribution_seconds'] = (args.single_image_ms / 1000 * 41
            * len(set(args.seeds)) * sum(set(args.budgets)) / min(args.budgets))
        report['estimate_caveat'] = 'linear sampling extrapolation; excludes I/O and startup; validate measured gate wall time'
    old_plan = output / 'plan.json'
    if old_plan.exists():
        saved = read_json(old_plan, output)
        require(saved.get('gate_schema_version') == SCHEMA, 'legacy gate is read-only; choose a new output directory')
        require(_config_hash({k: saved[k] for k in IDENTITY_KEYS}) == saved['config_hash'] == report['config_hash'],
                'output contains a different gate plan; choose another directory')
    elif any(output.glob('records/*.json')) or any(output.glob('maps_float/*.npy')) or (output/'summary.json').exists():
        raise ValueError('orphan gate artifacts without a plan; choose another directory')
    if args.plan_only:
        # Do not replace a completed gate's plan or summary in inspection mode.
        if not old_plan.exists():
            atomic_json(old_plan, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if report['missing_assets']:
        raise FileNotFoundError('missing frozen assets; see plan.json')
    if args.single_image_ms is None:
        raise ValueError('run a real single-image smoke, then supply --single-image-ms before the long gate')
    require(_config_hash({k: report[k] for k in IDENTITY_KEYS}) == report['config_hash'], 'current plan hash mismatch')
    completed = existing_records(output, report, compare_maps)
    # Malformed saved records must fail BEFORE overwriting any receipt, loading
    # models or warming up. A fully complete resume never calls the attributor.
    if not old_plan.exists():
        atomic_json(old_plan, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    device = _select_device(cfg['runtime']['device'])
    torch.manual_seed(cfg['runtime']['seed'])
    model_cfg = cfg['model']
    activation = model_cfg.get('output_activation', 'sigmoid' if cfg['dataset']['name']=='voc' else 'softmax')
    if cfg['attribution']['output_activation'] != activation:
        raise ValueError('activation mismatch')
    methods = {}
    if len(completed) < 40:
        model = load_model(name=model_cfg['name'], device=device, weights=model_cfg['weights'],
            num_classes=model_cfg['num_classes'],
            checkpoint=str(_resolve(model_cfg['checkpoint'])) if model_cfg.get('checkpoint') else None,
            output_activation=activation)
        current_context = build_prediction_context(dataset=cfg['dataset']['name'], split='debug',
            model_config=model_cfg, checkpoint=_resolve(model_cfg['checkpoint']) if model_cfg.get('checkpoint') else None)
        require(current_context == report['prediction_context'], 'weights/prediction context changed')
        baseline = _black_baseline(device, torch.float32)
        for budget in report['budgets']:
            for seed in report['seeds']:
                params = {**cfg['attribution'], 'num_samples': budget, 'seed': seed}
                methods[budget, seed] = _build_attributor('lime', model, params)
        # Exactly one discarded warm-up per setting only when there is work.
        first = dataset[0]
        check_sample(ROOT, dataset.rows.iloc[0], first, report['image_assets'][0])
        for method in methods.values():
            method.attribute(first['image'][None].to(device), first['target'], baseline)
    records = []
    started = time.perf_counter()
    for index in range(40):
        row = dataset.rows.iloc[index]
        safe_id = hashlib.sha256(str(row.image_id).encode()).hexdigest()[:16]
        record_path = output / 'records' / f'{safe_id}.json'
        if str(row.image_id) in completed:
            records.append(completed[str(row.image_id)])
            continue
        sample = dataset[index]
        check_sample(ROOT, row, sample, report['image_assets'][index])
        x = sample['image'][None].to(device)
        maps, runs = {}, []
        try:
            for (budget, seed), method in methods.items():
                _synchronize(device); t0 = time.perf_counter()
                result = method.attribute(x, sample['target'], baseline)
                _synchronize(device); ms = (time.perf_counter() - t0)*1000
                arr = result.numpy()
                if arr.dtype != np.float32 or arr.shape != x.shape[-2:] or not np.isfinite(arr).all():
                    raise ValueError('invalid attribution map')
                maps[budget, seed] = arr
                path = output / 'maps_float' / f'{safe_id}_{budget}_{seed}.npy'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.with_suffix('.tmp').open('wb') as handle:
                    np.save(handle, arr, allow_pickle=False)
                path.with_suffix('.tmp').replace(path)
                runs.append({**method.last_diagnostics, 'budget': budget, 'seed': seed,
                    'time_ms': ms, 'map_path': str(path.relative_to(output)),
                    'map_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            comparisons = [dict(budget=budget, seed=seed, reference_budget=max(report['budgets']),
                **compare_maps(maps[budget, seed], maps[max(report['budgets']), seed]))
                for budget in report['budgets'][:-1] for seed in report['seeds']]
            variance = [dict(budget=budget,
                pixel_variance_mean=float(np.stack([maps[budget,s] for s in report['seeds']]).var(0,ddof=1).mean()),
                seed_pairs=[dict(seed_a=a, seed_b=b, **compare_maps(maps[budget,a],maps[budget,b]))
                            for i,a in enumerate(report['seeds']) for b in report['seeds'][i+1:]])
                        for budget in report['budgets']]
            record = dict(image_id=sample['image_id'], target=sample['target'],
                config_hash=report['config_hash'], runs=runs, budget_comparisons=comparisons,
                seed_variance=variance, gate_schema_version=SCHEMA,
                source_image_sha256=report['image_assets'][index]['source_image_sha256'],
                preprocessed_image_sha256=report['image_assets'][index]['preprocessed_image_sha256'])
            check_sample(ROOT, row, sample, report['image_assets'][index])
            validate_record(record, report, report['image_assets'][index], output, compare_maps)
            atomic_json(record_path, record)
            records.append(record)
            print(f"{index+1}/40 {sample['image_id']} saved", flush=True)
        except Exception as error:
            with (output/'failures.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(image_id=sample['image_id'], error=repr(error)))+'\n')
            raise
    # Recheck all input bytes/tensors/code and all maps before final exports.
    _, _, final_plan = plan(config, args.budgets, args.seeds, output)
    require(final_plan['config_hash'] == report['config_hash'], 'input/code/environment changed during gate')
    checked = existing_records(output, report, compare_maps)
    require(len(checked) == 40, 'incomplete final gate')
    records = [checked[str(i)] for i in report['image_ids']]
    rows = [dict(image_id=r['image_id'], **v) for r in records for v in r['runs']]
    pd.DataFrame(rows).to_csv(output/'timings.csv', index=False)
    comparisons = [dict(image_id=r['image_id'], **v) for r in records for v in r['budget_comparisons']]
    pd.DataFrame(comparisons).to_csv(output/'comparisons.csv', index=False)
    variance = [dict(image_id=r['image_id'], budget=v['budget'], pixel_variance_mean=v['pixel_variance_mean'])
                for r in records for v in r['seed_variance']]
    pd.DataFrame(variance).to_csv(output/'variance.csv', index=False)
    similarity_summary = {}
    for budget in report['budgets'][:-1]:
        selected = [r for r in comparisons if r['budget'] == budget]
        values = [r['spearman'] for r in selected if r['status'] == 'valid']
        similarity_summary[str(budget)] = dict(reference_budget=max(report['budgets']),
            valid_pairs=len(values), total_pairs=len(selected),
            mean_spearman=float(np.mean(values)) if values else None,
            p10_spearman=float(np.quantile(values, .1)) if values else None,
            mean_rmse=float(np.mean([r['rmse'] for r in selected])))
    atomic_json(output/'summary.json', dict(config_hash=report['config_hash'], n_images=len(records),
        gate_schema_version=SCHEMA, processed=40-len(completed), skipped=len(completed),
        warmup_calls=len(methods), run_count=len(rows), budget_comparison_count=len(comparisons),
        seed_pair_count=sum(len(v['seed_pairs']) for r in records for v in r['seed_variance']),
        current_invocation_wall_seconds=time.perf_counter()-started,
        budget_status='pending human review; compare per-image similarity, degeneration, variance and timing',
        similarity_by_budget=similarity_summary,
        mean_pixel_variance={str(b): float(np.mean([v['pixel_variance_mean'] for v in variance if v['budget']==b]))
                             for b in report['budgets']},
        mean_time_ms={str(b): float(np.mean([v['time_ms'] for v in rows if v['budget']==b])) for b in report['budgets']}))


if __name__ == '__main__':
    main()
