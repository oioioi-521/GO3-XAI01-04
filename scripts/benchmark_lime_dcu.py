"""Independent LIME capacity pilot; not E's adapter or formal research output.

Uses official lime 0.2.0.1, explicit quickshift and true-target probabilities.
The full timer includes RGB conversion, segmentation, perturbation generation,
CPU/GPU transfers, inference, weighted regression and dense map construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def worker(args):
    import importlib.metadata
    import inspect
    import numpy as np
    import torch
    from lime import lime_image
    from skimage.segmentation import quickshift
    from models import load_model
    from models.factory import _BUILDERS
    from preprocessing.dataset import MetadataDataset, MEAN, STD
    from experiments.stability import perturb_rgb, stable_seed, stability_similarity
    from experiments.metrics import faithfulness_morf_auc_raw

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Each worker must see exactly one actual accelerator')
    torch.cuda.set_device(0)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    # Suppress only the progress display, leaving official sampling unchanged.
    lime_image.tqdm = lambda iterable: iterable
    device = torch.device('cuda:0')
    dataset = MetadataDataset(args.dataset, split='debug', limit=args.images)
    if len(dataset) != args.images:
        raise RuntimeError('insufficient frozen debug images')
    samples = [dataset[i] for i in range(args.rank, args.images, args.workers)]
    activation = 'sigmoid' if args.dataset == 'voc' else 'softmax'
    checkpoint = ROOT / 'models/checkpoints' / f'{args.model}_voc20.pt' if args.dataset == 'voc' else None
    if checkpoint:
        frozen = json.loads((ROOT / 'docs/VOC20_CHECKPOINT_MANIFEST.json').read_text())
        expected = next(r for r in frozen['checkpoints'] if r['model'] == args.model)
        if sha(checkpoint) != expected['sha256']:
            raise RuntimeError('VOC checkpoint hash mismatch')
    model = load_model(args.model, device, weights='none' if checkpoint else 'default',
                       num_classes=20 if checkpoint else 1000, checkpoint=str(checkpoint) if checkpoint else None,
                       output_activation=activation)
    model.eval()
    out = Path(args.output) / f'worker-{args.rank}'
    out.mkdir()
    mean = torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1)
    gpu_mean, gpu_std = mean.to(device), std.to(device)
    black = -gpu_mean / gpu_std
    rng_key = 'rng' if 'rng' in inspect.signature(quickshift).parameters else 'random_seed'

    def attribute(normalized, target):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        rgb = (normalized * std + mean).clamp(0, 1)[0].permute(1, 2, 0).numpy().copy()
        timings = {'segmentation_ms': 0., 'prediction_ms': 0., 'regression_ms': 0., 'prediction_samples': 0}

        def segment(image):
            t = time.perf_counter()
            result = quickshift(image, kernel_size=4, max_dist=200, ratio=.2,
                                convert2lab=True, **{rng_key: 42})
            timings['segmentation_ms'] += (time.perf_counter() - t) * 1000
            return result

        @torch.inference_mode()
        def predict(batch):
            torch.cuda.synchronize()
            t = time.perf_counter()
            tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(device, dtype=torch.float32)
            logits = model((tensor - gpu_mean) / gpu_std)
            if logits.shape != (len(batch), 20 if checkpoint else 1000):
                raise RuntimeError('model output shape mismatch')
            probabilities = logits.sigmoid() if activation == 'sigmoid' else logits.softmax(1)
            result = probabilities.cpu().numpy()
            if not np.isfinite(result).all():
                raise RuntimeError('nonfinite probabilities')
            timings['prediction_ms'] += (time.perf_counter() - t) * 1000
            timings['prediction_samples'] += len(batch)
            return result

        explainer = lime_image.LimeImageExplainer(kernel_width=.25, feature_selection='auto', random_state=42)
        original_fit = explainer.base.explain_instance_with_data

        def timed_fit(*pos, **kw):
            t = time.perf_counter()
            result = original_fit(*pos, **kw)
            timings['regression_ms'] += (time.perf_counter() - t) * 1000
            return result

        explainer.base.explain_instance_with_data = timed_fit
        explanation = explainer.explain_instance(
            rgb, predict, labels=(target,), top_labels=None, hide_color=0.,
            num_features=100000, num_samples=args.samples, batch_size=args.batch,
            segmentation_fn=segment, random_seed=42)
        if set(explanation.local_exp) != {target} or timings['prediction_samples'] != args.samples:
            raise RuntimeError('wrong target or sample count')
        segments = explanation.segments
        raw = np.zeros(segments.shape, dtype=np.float32)
        for segment_id, weight in explanation.local_exp[target]:
            raw[segments == segment_id] = weight
        lo, hi = float(raw.min()), float(raw.max())
        heatmap = (raw - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(raw)
        torch.cuda.synchronize()
        timings['attribution_ms'] = (time.perf_counter() - begin) * 1000
        timings['generation_and_other_ms'] = timings['attribution_ms'] - sum(timings[k] for k in ['segmentation_ms', 'prediction_ms', 'regression_ms'])
        timings.update(superpixels=int(len(np.unique(segments))), constant=hi-lo <= 1e-12,
                       local_r2=float(explanation.score), target=int(target))
        if not np.isfinite(heatmap).all() or not np.isfinite(raw).all() or heatmap.shape != (224, 224):
            raise RuntimeError('invalid map')
        return heatmap.astype(np.float32), timings

    warm_started = time.perf_counter()
    attribute(samples[0]['image'].unsqueeze(0), int(samples[0]['target']))
    warm_seconds = time.perf_counter() - warm_started
    torch.cuda.reset_peak_memory_stats()
    env = {'torch': torch.__version__, 'hip': torch.version.hip, 'python': sys.version,
           'device': torch.cuda.get_device_name(0), 'device_count': torch.cuda.device_count(),
           'visible_devices': {k: os.environ.get(k) for k in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']},
           'packages': {p: importlib.metadata.version(p) for p in ['lime', 'scikit-image', 'numpy', 'scipy', 'scikit-learn', 'torchvision']},
           'warmup_seconds': warm_seconds, 'checkpoint_sha256': sha(checkpoint) if checkpoint else sha(Path(torch.hub.get_dir()) / 'checkpoints' / _BUILDERS[args.model][1].url.rsplit('/', 1)[-1])}
    dump(out / 'ready.json', env)
    deadline = time.monotonic() + args.timeout
    while not (Path(args.output) / 'GO').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('start barrier timed out')
        time.sleep(.05)
    rows = []
    started = time.perf_counter()
    for cycle in range(args.cycles):
        for sample in samples:
            reference = None
            for repeat in range(-1, args.noise_repeats):
                normalized = sample['image'].unsqueeze(0)
                noise_seed = None
                if repeat >= 0:
                    noise_seed = stable_seed(args.dataset, sample['image_id'], repeat)
                    normalized, _ = perturb_rgb(normalized, sigma=.005, seed=noise_seed)
                heatmap, row = attribute(normalized, int(sample['target']))
                row.update(image_id=sample['image_id'], repeat=repeat, cycle=cycle, noise_seed=noise_seed)
                filename = f"{sample['image_id']}-{cycle}-{repeat}.npy"
                np.save(out / filename, heatmap, allow_pickle=False)
                row.update(map_file=f'worker-{args.rank}/{filename}', map_sha256=sha(out / filename))
                if repeat == -1:
                    reference = heatmap
                    if args.morf:
                        t = time.perf_counter()
                        row['morf_auc'] = faithfulness_morf_auc_raw(model, normalized.to(device), int(sample['target']),
                            torch.from_numpy(heatmap), black, [i/20 for i in range(21)], output_activation=activation)
                        row['morf_ms'] = (time.perf_counter() - t) * 1000
                else:
                    similarity = stability_similarity(heatmap, reference)
                    row.update(stability_score=similarity.score, stability_status=similarity.status)
                rows.append(row)
                print(json.dumps({'image': row['image_id'], 'repeat': repeat, 'ms': round(row['attribution_ms'], 2), 'segments': row['superpixels']}), flush=True)
    ended = time.perf_counter()
    dump(out / 'result.json', {'rank': args.rank, 'environment': env, 'started': started, 'ended': ended,
                              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'rows': rows})


def controller(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    args.output = str(out)
    devices = args.devices.split(',')
    if len(devices) != args.workers or len(set(devices)) != len(devices):
        raise ValueError('one unique device per worker is required')
    assigned = os.environ.get('HIP_VISIBLE_DEVICES') or os.environ.get('CUDA_VISIBLE_DEVICES')
    if assigned and not set(devices).issubset(assigned.split(',')):
        raise ValueError('devices exceed the container allocation')
    dump(out / 'manifest.json', {'args': vars(args), 'script_sha256': sha(__file__),
         'metadata_sha256': sha(ROOT/'data/metadata.csv'),
         'scope': 'Independent official-LIME capacity pilot; NOT E adapter or formal eval',
         'lime_contract': {'quickshift': {'kernel_size': 4, 'max_dist': 200, 'ratio': .2, 'seed': 42},
                           'kernel_width': .25, 'feature_selection': 'auto', 'num_features': 100000,
                           'hide_color': 0, 'target': 'metadata', 'map': 'signed coefficients, minmax normalization'}})
    processes, logs = [], []
    begin = time.perf_counter()
    try:
        for rank, device in enumerate(devices):
            env = os.environ.copy()
            env.pop('ROCR_VISIBLE_DEVICES', None)
            env.update(HIP_VISIBLE_DEVICES=device, CUDA_VISIBLE_DEVICES=device,
                       OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
            log = (out / f'worker-{rank}.log').open('w')
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--rank', str(rank)]
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        deadline = time.monotonic() + args.timeout
        while not all((out / f'worker-{r}/ready.json').exists() for r in range(args.workers)):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('worker exited before ready; inspect logs')
            if time.monotonic() > deadline:
                raise TimeoutError('worker startup timed out')
            time.sleep(.2)
        (out / 'GO').touch()
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('worker failed; inspect logs')
            if time.monotonic() > deadline:
                raise TimeoutError('run exceeded timeout')
            time.sleep(.2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('nonzero worker exit')
        total = time.perf_counter() - begin
        results = [json.loads((out / f'worker-{r}/result.json').read_text()) for r in range(args.workers)]
        rows = [row for result in results for row in result['rows']]
        expected = args.images * args.cycles * (1 + args.noise_repeats)
        keys = {(r['image_id'], r['cycle'], r['repeat']) for r in rows}
        if len(rows) != expected or len(keys) != expected or len({r['image_id'] for r in rows}) != args.images:
            raise RuntimeError('coverage or duplicate failure')
        steady = max(r['ended'] for r in results) - min(r['started'] for r in results)
        summary = {'status': 'complete', 'model': args.model, 'dataset': args.dataset,
                   'workers': args.workers, 'images': args.images, 'calls': len(rows),
                   'samples': args.samples, 'batch': args.batch, 'noise_repeats': args.noise_repeats,
                   'steady_wall_seconds': steady, 'total_wall_seconds': total, 'calls_per_second': len(rows)/steady,
                   'max_device_memory_bytes': max(r['peak_allocated_bytes'] for r in results),
                   'means': {k: statistics.mean(r[k] for r in rows) for k in ['attribution_ms','segmentation_ms','prediction_ms','regression_ms','generation_and_other_ms','superpixels']},
                   'mean_original_ms': statistics.mean(r['attribution_ms'] for r in rows if r['repeat'] == -1),
                   'mean_perturbed_ms': statistics.mean(r['attribution_ms'] for r in rows if r['repeat'] >= 0) if args.noise_repeats else None,
                   'rows': rows}
        dump(out / 'summary.json', summary)
        print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)
    except BaseException as error:
        dump(out / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['resnet50','densenet121','vgg16'], required=True)
    parser.add_argument('--dataset', choices=['imagenet','voc'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--images', type=int, default=8)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--noise-repeats', type=int, default=0)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--morf', action='store_true')
    parser.add_argument('--timeout', type=int, default=2400)
    parser.add_argument('--rank', type=int, default=-1)
    args = parser.parse_args()
    if min(args.images, args.workers, args.batch, args.samples, args.cycles, args.threads) < 1 or args.workers > args.images or args.noise_repeats not in (0,1,5):
        parser.error('invalid count')
    if args.rank >= 0:
        worker(args)
    else:
        controller(args)
