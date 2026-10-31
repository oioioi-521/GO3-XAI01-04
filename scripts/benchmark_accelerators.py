"""Isolated fixed-workload attribution benchmark (CUDA / HIP / CPU).

Does not write formal experiment tables. All workers warm up before a shared
start barrier; each owns a directory. FP32, image IDs, seeds, model and method
parameters are recorded so throughput comparisons can be checked afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def shard_indices(count, workers, rank):
    if not 0 < workers <= count or not 0 <= rank < workers:
        raise ValueError('require 0 <= rank < workers <= images')
    return list(range(rank, count, workers))


def parameters(args):
    if args.method == 'kernelshap':
        return dict(n_samples=args.samples or 2048, perturbations_per_eval=args.batch or 16,
                    feature_grid_size=7, seed=42)
    if args.method == 'rise':
        return dict(num_masks=args.samples or 4000, batch_size=args.batch or 32,
                    mask_size=7, mask_probability=.5, seed=42)
    if args.method == 'ig':
        return dict(n_steps=args.samples or 50, internal_batch_size=args.batch or 10,
                    approximation_method='gausslegendre')
    if args.method == 'occlusion':
        return dict(window_height=15, window_width=15, stride_height=8, stride_width=8,
                    perturbations_per_eval=args.batch or 16)
    layers = {'resnet50': 'layer4.2', 'densenet121': 'features.denseblock4',
              'vgg16': 'features.28'}
    return dict(target_layer=layers[args.model])


def worker(args):
    import importlib.metadata
    import numpy as np
    import torch
    from experiments.run_unit import _build_attributor, _black_baseline
    from experiments.metrics import faithfulness_morf_auc_raw
    from models import load_model
    from preprocessing.dataset import MetadataDataset

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('accelerator requested but unavailable; refusing CPU fallback')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('worker must see exactly one accelerator')
        torch.cuda.set_device(0)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    out = Path(args.output) / f'worker-{args.rank}'
    out.mkdir()
    dataset = MetadataDataset(args.dataset, split='debug', limit=args.images)
    if len(dataset) != args.images:
        raise ValueError('requested image count exceeds frozen debug split')
    indices = shard_indices(len(dataset), args.workers, args.rank)
    samples = [dataset[i] for i in indices]
    activation = 'sigmoid' if args.dataset == 'voc' else 'softmax'
    model = load_model(args.model, device, weights='none' if args.checkpoint else 'default',
                       num_classes=20 if args.dataset == 'voc' else 1000,
                       checkpoint=args.checkpoint, output_activation=activation)
    attr = _build_attributor(args.method, model, parameters(args))
    baseline = _black_baseline(device, torch.float32)
    warm = samples[0]['image'].unsqueeze(0).to(device)
    for _ in range(args.warmup):
        attr.attribute(warm, target=samples[0]['target'], baseline=baseline)
    sync()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0) if device.type == 'cuda' else None
    def package_version(package):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None
    env = dict(torch=torch.__version__, hip=torch.version.hip, cuda=torch.version.cuda,
               python=sys.version, device=str(device), name=props.name if props else 'cpu',
               memory_bytes=props.total_memory if props else None,
               arch=getattr(props, 'gcnArchName', None),
               visible_devices={k: os.environ.get(k) for k in
                                ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')},
               packages={p: package_version(p) for p in
                         ('torchvision', 'captum', 'numpy', 'scipy', 'scikit-learn')})
    write_json(out / 'ready.json', env)
    deadline = time.monotonic() + args.timeout
    while not (Path(args.output) / 'GO').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('start barrier expired')
        time.sleep(.02)
    rows = []
    started = time.perf_counter()
    for repetition in range(args.repeats):
        for sample in samples:
            image = sample['image'].unsqueeze(0).to(device)
            sync()
            t0 = time.perf_counter()
            heatmap = attr.attribute(image, target=sample['target'], baseline=baseline)
            sync()
            elapsed = time.perf_counter() - t0
            array = heatmap.detach().cpu().numpy().astype(np.float32)
            if array.shape != (224, 224) or not np.isfinite(array).all():
                raise ValueError('invalid attribution map')
            filename = f'{sample["image_id"]}-{repetition}.npy'
            np.save(out / filename, array, allow_pickle=False)
            row = dict(image_id=sample['image_id'], target=sample['target'], repeat=repetition,
                       attribution_ms=elapsed * 1000, map_file=f'worker-{args.rank}/{filename}',
                       map_sha256=hashlib.sha256((out / filename).read_bytes()).hexdigest())
            if args.morf:
                row['morf_auc'] = faithfulness_morf_auc_raw(
                    model, image, sample['target'], heatmap, baseline,
                    [i / 20 for i in range(21)], output_activation=activation)
            rows.append(row)
    sync()
    ended = time.perf_counter()
    write_json(out / 'result.json', dict(rank=args.rank, indices=indices, environment=env,
               started=started, ended=ended, rows=rows,
               peak_allocated_bytes=torch.cuda.max_memory_allocated() if props else 0))


def controller(args):
    import statistics
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output = str(output)
    devices = [s.strip() for s in args.devices.split(',')] if args.devices else []
    if args.device == 'cuda' and len(devices) != args.workers:
        raise ValueError('--devices must explicitly list one physical device per worker')
    if len(devices) != len(set(devices)):
        raise ValueError('duplicate devices')
    assigned = os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices and assigned and not set(devices).issubset(set(assigned.split(','))):
        raise ValueError('requested devices exceed the container assignment')
    shard_indices(args.images, args.workers, 0)
    write_json(output / 'manifest.json', dict(arguments=vars(args), attribution=parameters(args),
               metadata_sha256=hashlib.sha256((ROOT / 'data/metadata.csv').read_bytes()).hexdigest(),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                text=True).strip() if (ROOT / '.git').exists() else None))
    children, handles = [], []
    total_start = time.perf_counter()
    try:
        for rank in range(args.workers):
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads))
            if devices:
                env['HIP_VISIBLE_DEVICES'] = devices[rank]
                env['CUDA_VISIBLE_DEVICES'] = devices[rank]
            log = (output / f'worker-{rank}.log').open('w')
            handles.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                       '--rank', str(rank), '--output', str(output)]
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log))
        deadline = time.monotonic() + args.timeout
        while not all((output / f'worker-{r}/ready.json').exists() for r in range(args.workers)):
            if any(p.poll() is not None for p in children):
                raise RuntimeError('worker exited before barrier; see worker logs')
            if time.monotonic() > deadline:
                raise TimeoutError('warmup timed out')
            time.sleep(.1)
        (output / 'GO').touch()
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('worker failed; incomplete run must not be used')
            if time.monotonic() > deadline:
                raise TimeoutError('benchmark timed out')
            time.sleep(.1)
        if any(p.returncode for p in children):
            raise RuntimeError('worker failed')
        results = [json.loads((output / f'worker-{r}/result.json').read_text()) for r in range(args.workers)]
        indices = [i for result in results for i in result['indices']]
        if sorted(indices) != list(range(args.images)):
            raise ValueError('image coverage mismatch')
        rows = [row for result in results for row in result['rows']]
        keys = {(r['image_id'], r['repeat']) for r in rows}
        if len(rows) != args.images * args.repeats or len(keys) != len(rows):
            raise ValueError('duplicate or missing work')
        seconds = max(r['ended'] for r in results) - min(r['started'] for r in results)
        summary = dict(status='complete', workers=args.workers, images=args.images,
                       calls=len(rows), method=args.method, model=args.model, dataset=args.dataset,
                       parameters=parameters(args), steady_wall_seconds=seconds,
                       total_wall_seconds=time.perf_counter() - total_start,
                       calls_per_second=len(rows) / seconds,
                       mean_attribution_ms=statistics.mean(r['attribution_ms'] for r in rows),
                       median_attribution_ms=statistics.median(r['attribution_ms'] for r in rows),
                       max_device_memory_bytes=max(r['peak_allocated_bytes'] for r in results),
                       rows=rows)
        write_json(output / 'summary.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}, indent=2))
    except BaseException as error:
        write_json(output / 'failure.json', dict(status='failed', error=repr(error)))
        raise
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['ig', 'gradcam', 'occlusion', 'rise', 'kernelshap'], required=True)
    parser.add_argument('--model', choices=['resnet50', 'densenet121', 'vgg16'], default='resnet50')
    parser.add_argument('--dataset', choices=['imagenet', 'voc'], default='imagenet')
    parser.add_argument('--checkpoint')
    parser.add_argument('--images', type=int, default=14)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--devices', help='physical device IDs within the container device namespace')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--morf', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.images, args.workers, args.threads, args.repeats, args.timeout) <= 0 or args.warmup < 1:
        parser.error('positive counts and at least one warmup required')
    if any(v is not None and v <= 0 for v in (args.samples, args.batch)):
        parser.error('samples and batch must be positive')
    if args.dataset == 'voc' and not args.checkpoint:
        parser.error('VOC requires a real 20-class checkpoint')
    if args.method == 'rise' and args.dataset == 'voc':
        parser.error('this source RISE adapter uses softmax; use ImageNet for this benchmark')
    worker(args) if args.rank is not None else controller(args)


if __name__ == '__main__':
    main()
