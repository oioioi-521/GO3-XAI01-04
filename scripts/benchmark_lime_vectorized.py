"""Bounded, paired original/vectorized LIME probe; never starts formal eval.

Three debug images per model/dataset, 2000/4000/8000 samples, reference batch32
and vectorized batch32/64/128. Also pair five protocol noise repeats at 2000.
Six isolated workers at most, each with a 10% memory cap and two CPU threads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

JOBS = [f"{m}-{d}" for m in ("resnet50", "densenet121", "vgg16") for d in ("imagenet", "voc")]
VARIANTS = [("reference32", 32), ("vector32", 32), ("vector64", 64), ("vector128", 128)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def worker(args):
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from experiments.attribution.lime import LIME
    from experiments.run_unit import _load_config, _black_baseline
    from experiments.stability import perturb_rgb, stable_seed
    from models import load_model
    from preprocessing.dataset import MetadataDataset
    from lime_vectorized import VectorizedLIME

    torch.set_num_threads(2)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    torch.manual_seed(42)
    root, out = args.release, args.output / args.worker
    out.mkdir(parents=True, exist_ok=False)
    model_name, dataset_name = args.worker.split('-')
    cfg = _load_config(root / f'configs/lime_{model_name}_{dataset_name}.yaml')
    dataset = MetadataDataset(dataset_name, split='debug')
    ids = dataset.rows.image_id.astype(str).tolist()
    assert len(ids) == 40 and len(set(ids)) == 40
    selected = [ids[0], ids[20], ids[-1]]
    if dataset_name == 'imagenet':
        tail = 'ILSVRC2012_val_00016006' if model_name == 'vgg16' else 'ILSVRC2012_val_00011253'
        assert tail in ids
        if tail not in selected: selected[-1] = tail
    mc = cfg['model']
    model = load_model(name=model_name, device=torch.device('cuda:0'), weights=mc['weights'],
                       num_classes=mc['num_classes'], output_activation=mc['output_activation'],
                       checkpoint=str(root / mc['checkpoint']) if mc.get('checkpoint') else None)
    baseline = _black_baseline(torch.device('cuda:0'), torch.float32)
    settings = dict(cfg['attribution'])
    records, comparisons, failures = [], [], []
    disabled = set()
    started = time.time()
    dump(out / 'manifest.json', dict(job=args.worker, selected=selected, config=cfg,
         budgets=[2000, 4000, 8000], seeds=[42, 43, 44], variants=VARIANTS,
         noise_repeats=5, noise_budget=2000, torch=torch.__version__, hip=torch.version.hip,
         device=torch.cuda.get_device_name(0), visible_device=os.environ['HIP_VISIBLE_DEVICES'],
         memory_fraction=.10, threads=2, started_at=started,
         numeric_tolerance=dict(max_abs=1e-5, rmse=1e-6, r2_abs=1e-6),
         caveat='bounded performance/equivalence probe, not a formal result or full budget gate'))

    def run(variant, batch, x, target, budget, seed, case, warmup=False):
        method = (LIME if variant == 'reference32' else VectorizedLIME)(
            model=model, **{**settings, 'num_samples': budget, 'batch_size': batch, 'seed': seed})
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        result = method.attribute(x, target, baseline)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        if warmup: return None
        a = result.numpy()
        assert a.shape == (224, 224) and a.dtype == np.float32 and np.isfinite(a).all()
        path = out / f'{case}_{budget}_{seed}_{variant}.npy'
        np.save(path, a, allow_pickle=False)
        row = dict(case=case, budget=budget, seed=seed, variant=variant, batch=batch,
                   seconds=seconds, peak_allocated=torch.cuda.max_memory_allocated(),
                   map_path=path.name, map_sha256=sha(path), diagnostics=method.last_diagnostics)
        records.append(row)
        dump(out / 'records.json', records)
        print(json.dumps({k: row[k] for k in ('case', 'budget', 'variant', 'seconds')}), flush=True)
        return a, row

    def compare(reference, candidate):
        a, ar = reference; b, br = candidate
        diff = a.astype(np.float64) - b.astype(np.float64)
        maximum, rmse = float(np.abs(diff).max()), float(np.sqrt(np.mean(diff**2)))
        r2diff = abs(ar['diagnostics']['weighted_r2'] - br['diagnostics']['weighted_r2'])
        constant = bool(np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12)
        rho = None if constant else float(spearmanr(a.ravel(), b.ravel()).statistic)
        comparisons.append(dict(case=ar['case'], budget=ar['budget'], seed=ar['seed'],
            variant=br['variant'], exact=bool(np.array_equal(a, b)), max_abs=maximum, rmse=rmse,
            spearman=rho, r2_abs=r2diff, numeric_pass=maximum <= 1e-5 and rmse <= 1e-6 and r2diff <= 1e-6,
            speedup=ar['seconds'] / br['seconds']))
        dump(out / 'comparisons.json', comparisons)

    first = dataset[ids.index(selected[0])]
    for variant, batch in VARIANTS:
        try:
            run(variant, batch, first['image'][None].cuda(), int(first['target']), 256, 42, 'warmup', True)
        except torch.cuda.OutOfMemoryError as e:
            if variant in ('reference32', 'vector32'): raise
            disabled.add(variant)
            failures.append(dict(variant=variant, stage='warmup', error=repr(e)))
            torch.cuda.empty_cache()
    for index, image_id in enumerate(selected):
        sample = dataset[ids.index(image_id)]
        x = sample['image'][None].cuda(); target = int(sample['target']); seed = 42 + index
        for budget in (2000, 4000, 8000):
            paired = {}
            variants = VARIANTS if index % 2 == 0 else list(reversed(VARIANTS))
            for variant, batch in variants:
                if variant in disabled: continue
                dump(out / 'status.json', dict(state='running', case=image_id, budget=budget,
                     variant=variant, completed_calls=len(records), updated_at=time.time()))
                try:
                    paired[variant] = run(variant, batch, x, target, budget, seed, image_id)
                except torch.cuda.OutOfMemoryError as e:
                    if variant in ('reference32', 'vector32'): raise
                    disabled.add(variant); failures.append(dict(variant=variant, case=image_id, budget=budget, error=repr(e)))
                    torch.cuda.empty_cache()
            for variant in paired:
                if variant != 'reference32': compare(paired['reference32'], paired[variant])
            oldroot = root / f'results/dcu-lime-20261013/high_budget/lime-{args.worker}'
            old = next(r for r in (json.loads(p.read_text()) for p in (oldroot / 'records').glob('*.json')) if r['image_id'] == image_id)
            anchor = next(r for r in old['runs'] if r['budget'] == budget and r['seed'] == seed)
            path = oldroot / anchor['map_path']; assert sha(path) == anchor['map_sha256']
            paired['reference32'][1]['matches_saved_high_budget'] = bool(np.array_equal(np.load(path), paired['reference32'][0]))
            dump(out / 'records.json', records)

    sample = first
    for repeat in range(5):
        noise_seed = stable_seed(dataset_name, sample['image_id'], repeat)
        noisy, noise_mae = perturb_rgb(sample['image'][None], sigma=.005, seed=noise_seed)
        x = noisy.cuda()
        case = f"noise{repeat}_{sample['image_id']}"
        dump(out / 'status.json', dict(state='noise_probe', case=case, completed_calls=len(records), updated_at=time.time()))
        a = run('reference32', 32, x, int(sample['target']), 2000, 42, case)
        b = run('vector32', 32, x, int(sample['target']), 2000, 42, case)
        compare(a, b)
    dump(out / 'summary.json', dict(state='complete', records=records, comparisons=comparisons,
         failures=failures, numeric_pass=all(c['numeric_pass'] for c in comparisons),
         wall_seconds=time.time()-started, finished_at=time.time(), formal_started=False))
    dump(out / 'status.json', dict(state='complete', completed_calls=len(records), updated_at=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--devices', default='0,1,2,3,4,5')
    p.add_argument('--worker', choices=JOBS)
    args = p.parse_args()
    args.release = args.release.resolve(); args.output = args.output.resolve()
    sys.path.insert(0, str(args.release)); sys.path.insert(0, str(args.release/'scripts'))
    os.environ['TORCH_HOME'] = str(args.release/'.cache/torch')
    if args.worker:
        try: worker(args)
        except BaseException as error:
            dump(args.output / args.worker / 'failure.json', dict(error=repr(error), time=time.time()))
            raise
        return
    devices = [int(v) for v in args.devices.split(',')]
    assert len(set(devices)) == len(devices) and 1 <= len(devices) <= 6 and all(0 <= d < 8 for d in devices)
    assert not args.output.exists(), 'fresh output only; never overwrite an experiment'
    args.output.mkdir(parents=True)
    from run_dcu_formal_queue import preflight
    preflight(args.output)
    dump(args.output/'manifest.json', dict(scope=__doc__, devices=devices, jobs=JOBS,
         source_sha256=sha(__file__), prototype_sha256=sha(Path(__file__).with_name('lime_vectorized.py')),
         release_sha256=sha(args.release/'release-manifest.json'), formal_started=False, started_at=time.time()))
    active, pending, completed, failed = {}, list(JOBS), [], []
    try:
        while active or pending:
            for device in devices:
                if device in active or not pending: continue
                job = pending.pop(0)
                env = os.environ.copy(); env.pop('ROCR_VISIBLE_DEVICES', None)
                env.update(HIP_VISIBLE_DEVICES=str(device), CUDA_VISIBLE_DEVICES=str(device),
                           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
                log = (args.output/f'{job}.log').open('w')
                proc = subprocess.Popen([sys.executable, __file__, '--release', str(args.release),
                    '--output', str(args.output), '--worker', job], env=env, cwd=args.release,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[device] = (job, proc, log)
            for device, (job, proc, log) in list(active.items()):
                if proc.poll() is not None:
                    (completed if proc.returncode == 0 else failed).append(job)
                    log.close(); del active[device]
            dump(args.output/'status.json', dict(state='running' if active or pending else 'complete',
                running=[j for j, _, _ in active.values()], pending=pending, completed=completed,
                failed=failed, updated_at=time.time(), formal_started=False))
            if active or pending: time.sleep(5)
        if failed: raise RuntimeError(f'Failed workers: {failed}')
        (args.output/'postrun').mkdir()
        preflight(args.output/'postrun')
        dump(args.output/'COMPLETE.json', dict(status='probe_complete', numeric_pass=all(
            json.loads((args.output/j/'summary.json').read_text())['numeric_pass'] for j in JOBS), formal_started=False))
    finally:
        for job, proc, log in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            log.close()


if __name__ == '__main__': main()
