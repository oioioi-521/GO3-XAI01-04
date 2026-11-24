"""Deploy the five accepted methods as isolated, resumable one-device jobs.

All 30 debug gates (including resume verification) must pass before eval starts.
The existing base/stability runners retain their public outputs and provenance.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
METHODS = ('gradcam', 'ig', 'occlusion', 'rise', 'kernelshap')
MODELS = ('resnet50', 'densenet121', 'vgg16')
METRICS = {'efficiency_time_ms', 'faithfulness_morf_auc_raw',
           'stability_spearman', 'stability_valid_rate'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def jobs():
    return [f'{method}-{model}-{dataset}' for method in METHODS
            for model in MODELS for dataset in ('imagenet', 'voc')]


def prepare(root):
    import yaml
    for job in jobs():
        method, model, dataset = job.split('-')
        prefix = 'kernelshap_eval' if method == 'kernelshap' else method
        source = ROOT / f'configs/{prefix}_{model}_{dataset}.yaml'
        original = yaml.safe_load(source.read_text())
        assert original['dataset']['split'] == 'eval'
        assert original['metrics']['names'] == ['efficiency_time_ms', 'faithfulness_morf_auc_raw']
        assert original['metrics']['deletion_fractions'] == [i/20 for i in range(21)]
        activation = 'sigmoid' if dataset == 'voc' else 'softmax'
        assert original['model']['output_activation'] == activation
        if method == 'rise':
            assert original['attribution']['num_masks'] == 4000
            assert original['attribution']['output_activation'] == activation
        if method == 'occlusion':
            assert [original['attribution'][k] for k in ('window_height','window_width','stride_height','stride_width')] == [32,32,16,16]
        if method == 'gradcam' and model == 'densenet121':
            assert original['attribution']['target_layer'] == 'features.norm5'
        for phase in ('gate', 'formal'):
            out = root / phase / job
            relative = str(out.relative_to(ROOT))
            config = copy.deepcopy(original)
            config['dataset']['split'] = 'debug' if phase == 'gate' else 'eval'
            config['runtime'].update(device='cuda:0', max_images=1 if phase == 'gate' else None,
                                     warmup_runs=1, resume=True)
            config['output'] = {k:f'{relative}/{v}' for k,v in {
                'per_image_csv':'per_image.csv','units_csv':'units.csv','predictions_csv':'predictions.csv',
                'config_dir':'configs','state_dir':'run_state','run_log':'base_log.jsonl',
                'maps_dir':'maps','float_maps_dir':'maps_float'}.items()}
            config['output'].update(save_maps=True, save_float_maps=True)
            protocol = yaml.safe_load((ROOT/'configs/stability_protocol_v1.yaml').read_text())
            protocol['runtime']['device'] = 'cuda:0'
            protocol['output'] = {k:f'{relative}/{v}' for k,v in {
                'trace_csv':'stability_trace.csv','state_dir':'stability_state',
                'config_dir':'stability_configs','run_log':'stability_log.jsonl'}.items()}
            out.mkdir(parents=True, exist_ok=True)
            for name, content in [('unit.yaml',config), ('protocol.yaml',protocol)]:
                text = yaml.safe_dump(content,sort_keys=False)
                path = out/name
                if path.exists() and path.read_text() != text:
                    raise RuntimeError(f'Cannot change an existing run config: {path}')
                path.write_text(text)


def preflight(root):
    import torch
    from PIL import Image
    from preprocessing.dataset import get_metadata
    manifest = json.loads((ROOT/'release-manifest.json').read_text())
    for item in manifest['files']:
        assert sha(ROOT/item['path']) == item['sha256'], item['path']
    assert sha(ROOT/'data-file-manifest.json') == manifest['data_manifest_sha256']
    images = json.loads((ROOT/'data-file-manifest.json').read_text())
    assert len(images) == 1000
    for item in images:
        path = ROOT/item['path']
        assert sha(path) == item['sha256'], path
        with Image.open(path) as image:
            image.verify()
    metadata = get_metadata()
    assert {str(Path('data')/p.replace('\\','/')) for p in metadata.image_path} == {r['path'] for r in images}
    for dataset in ('imagenet','voc'):
        for split, count in [('eval',460),('debug',40)]:
            part = metadata[(metadata.dataset == dataset)&(metadata.split == split)]
            assert len(part) == count and part.image_id.nunique() == count
    for item in json.loads((ROOT/'docs/VOC20_CHECKPOINT_MANIFEST.json').read_text())['checkpoints']:
        assert sha(ROOT/f"models/checkpoints/{item['model']}_voc20.pt") == item['sha256']
    for item in manifest['imagenet_weights']:
        assert sha(Path(torch.hub.get_dir())/'checkpoints'/item['file']) == item['sha256']
    assert torch.cuda.is_available() and torch.cuda.device_count() == 8
    import shutil
    assert shutil.disk_usage(root).free > 20*1024**3
    dump(root/'preflight.json',dict(status='passed',time=time.time(),images=1000,
         devices=[torch.cuda.get_device_name(i) for i in range(8)],torch=torch.__version__,
         hip=torch.version.hip,release_sha256=sha(ROOT/'release-manifest.json')))


def validate(out, expected_count):
    import numpy as np
    import pandas as pd
    import yaml
    from experiments.map_provenance import verify_map_provenance
    from experiments.predictions import build_prediction_context
    from experiments.run_unit import _config_hash
    from experiments.stability import stable_seed
    from preprocessing.dataset import get_split
    config = yaml.safe_load((out/'unit.yaml').read_text())
    unit = dict(method=config['method'],model=config['model']['name'],dataset=config['dataset']['name'])
    split = config['dataset']['split']
    frozen = get_split(unit['dataset'],split).iloc[:expected_count]
    ids = set(frozen.image_id.astype(str))
    target = dict(zip(frozen.image_id.astype(str),frozen.class_id.astype(int)))
    frame = pd.read_csv(out/'per_image.csv',dtype={'image_id':str})
    assert len(frame) == expected_count*4 and set(frame.metric) == METRICS
    assert not frame.duplicated(['image_id','metric']).any()
    for key,value in unit.items(): assert set(frame[key]) == {value}
    assert np.isfinite(frame.value).all()
    for metric in METRICS: assert set(frame[frame.metric == metric].image_id) == ids
    predictions = pd.read_csv(out/'predictions.csv',dtype={'image_id':str})
    assert len(predictions) == expected_count and set(predictions.image_id) == ids
    assert predictions.image_id.nunique() == expected_count
    assert all(int(row.target_class_id) == target[row.image_id] for row in predictions.itertuples())
    checkpoint = config['model'].get('checkpoint')
    context = build_prediction_context(dataset=unit['dataset'],split=split,
        model_config=config['model'],checkpoint=ROOT/checkpoint if checkpoint else None)
    digest = _config_hash(config)
    map_root = out/'maps_float'/f"{unit['method']}_{unit['model']}_{unit['dataset']}"
    assert len(list(map_root.glob('*.npy'))) == expected_count
    for image_id in ids:
        path = map_root/f'{image_id}.npy'
        array = np.load(path,allow_pickle=False)
        assert array.shape == (224,224) and array.dtype == np.float32 and np.isfinite(array).all()
        assert array.min() >= 0 and array.max() <= 1
        assert verify_map_provenance(path,image_id=image_id,unit=unit,split=split,
            base_config_hash=digest,prediction_context=context)
    traces = list(out.glob('stability_trace_*.csv'))
    assert len(traces) == 1
    trace = pd.read_csv(traces[0],dtype={'image_id':str,'seed':str})
    assert len(trace) == expected_count*5
    assert set(zip(trace.image_id,trace['repeat'])) == {(i,r) for i in ids for r in range(5)}
    assert np.isfinite(trace.score).all() and np.isfinite(trace.attribution_time_ms).all()
    for row in trace.itertuples():
        assert int(row.seed) == stable_seed(unit['dataset'],row.image_id,int(row.repeat))
        assert row.target == target[row.image_id] and row.sigma == .005
    scores = trace.groupby('image_id').score.mean()
    reported = frame[frame.metric == 'stability_spearman'].set_index('image_id').value
    assert np.allclose(scores.sort_index(),reported.sort_index(),rtol=1e-7,atol=1e-9)
    units = pd.read_csv(out/'units.csv')
    assert len(units) == 4 and set(units.metric) == METRICS and (units.n == expected_count).all()
    for row in units.itertuples():
        assert np.isclose(row.mean,frame[frame.metric == row.metric].value.mean())
    return dict(status='verified',images=expected_count,metrics_rows=len(frame),trace_rows=len(trace),
                maps=expected_count,release_sha256=sha(ROOT/'release-manifest.json'),
                csv_sha256={p.name:sha(p) for p in [out/'per_image.csv',out/'units.csv',out/'predictions.csv',traces[0]]})


def worker(root, phase, job):
    import gc
    import torch
    from experiments import run_unit, run_stability
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    out = root/phase/job
    dump(out/'environment.json',dict(torch=torch.__version__,hip=torch.version.hip,
        device=torch.cuda.get_device_name(0),device_count=torch.cuda.device_count(),
        visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'),pid=os.getpid()))
    for stage, callback in [('base',lambda:run_unit.run(out/'unit.yaml')),
                            ('stability',lambda:run_stability.run(out/'unit.yaml',out/'protocol.yaml'))]:
        dump(out/'progress.json',dict(stage=stage,time=time.time(),pid=os.getpid()))
        callback()
        gc.collect()
        torch.cuda.empty_cache()
    expected = 1 if phase == 'gate' else 460
    result = validate(out,expected)
    # Exercise the real resume path at the gate and again after the full unit.
    for callback, log in [(lambda:run_unit.run(out/'unit.yaml'),'base_log.jsonl'),
                          (lambda:run_stability.run(out/'unit.yaml',out/'protocol.yaml'),'stability_log.jsonl')]:
        callback()
        record = json.loads((out/log).read_text().splitlines()[-1])
        assert (record['processed'],record['skipped']) == (0,expected), record
        gc.collect()
        torch.cuda.empty_cache()
    # Resume rebuilds summary CSVs, so validate and record their final hashes.
    result = validate(out,expected)
    result['resume_verified'] = True
    dump(out/'completion.json',result)
    dump(out/'progress.json',dict(stage='completed',time=time.time(),pid=os.getpid()))


def verified_done(out):
    path = out/'completion.json'
    if not path.exists(): return False
    result = json.loads(path.read_text())
    assert result['status'] == 'verified' and result['resume_verified']
    assert result['release_sha256'] == sha(ROOT/'release-manifest.json')
    for name,digest in result['csv_sha256'].items(): assert sha(out/name) == digest
    return True


def pool(root, phase):
    # Longest units first to reduce idle cards at the end of a full sweep.
    costs = json.loads((ROOT/'analysis/dcu_full_matrix_estimate_20261013.json').read_text())['cells']
    weights = {f"{c['method']}-{c['model']}-{c['dataset']}":c['full_minutes'] for c in costs}
    pending = sorted([j for j in jobs() if not verified_done(root/phase/j)],key=lambda j:-weights[j])
    active, failures = {}, []
    try:
        while pending or active:
            for device in range(8):
                if device in active or not pending: continue
                job = pending.pop(0)
                env = os.environ.copy()
                env.pop('ROCR_VISIBLE_DEVICES',None)
                env.update(HIP_VISIBLE_DEVICES=str(device),CUDA_VISIBLE_DEVICES=str(device),
                           OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                log = (root/phase/job/'worker.log').open('a')
                command = [sys.executable,str(Path(__file__).resolve()),'--root',str(root),
                           '--worker',job,'--phase',phase]
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[device] = dict(job=job,process=process,log=log,started=time.time())
            for device, item in list(active.items()):
                if time.time()-item['started'] > 8*3600 and item['process'].poll() is None:
                    os.killpg(item['process'].pid,signal.SIGTERM)
                    try: item['process'].wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(item['process'].pid,signal.SIGKILL)
                        item['process'].wait()
                    dump(root/phase/item['job']/'failure.json',dict(error='8-hour per-job timeout',time=time.time()))
                code = item['process'].poll()
                if code is None: continue
                item['log'].close()
                if code != 0 or not verified_done(root/phase/item['job']): failures.append(item['job'])
                del active[device]
            running = []
            for device,item in active.items():
                out = root/phase/item['job']
                progress = json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {'stage':'loading'}
                counts = collections.Counter()
                if (out/'per_image.csv').exists():
                    with (out/'per_image.csv').open() as f:
                        counts.update(row.get('metric') for row in csv.DictReader(f))
                running.append(dict(device=device,job=item['job'],pid=item['process'].pid,
                    stage=progress['stage'],base_images=counts['efficiency_time_ms'],
                    stability_images=counts['stability_spearman'],elapsed_seconds=time.time()-item['started']))
            dump(root/'status.json',dict(phase=phase,state='running',updated_at=time.time(),
                 pending=pending,running=running,failed=failures,
                 completed=[j for j in jobs() if (root/phase/j/'completion.json').exists()]))
            if pending or active: time.sleep(5)
        if failures: raise RuntimeError(f'{phase} jobs failed: {failures}')
    finally:
        for item in active.values():
            if item['process'].poll() is None: os.killpg(item['process'].pid,signal.SIGTERM)
            item['log'].close()


def merge(root):
    import pandas as pd
    frames,units,traces = [],[],[]
    for job in jobs():
        out = root/'formal'/job
        assert verified_done(out)
        frames.append(pd.read_csv(out/'per_image.csv',dtype={'image_id':str}))
        units.append(pd.read_csv(out/'units.csv'))
        traces.append(pd.read_csv(next(out.glob('stability_trace_*.csv')),dtype={'image_id':str,'seed':str}))
    merged = root/'merged'
    merged.mkdir(exist_ok=True)
    for name,parts,keys,count in [
        ('per_image.csv',frames,['method','model','dataset','image_id','metric'],30*460*4),
        ('units.csv',units,['method','model','dataset','metric'],30*4),
        ('stability_trace.csv',traces,['method','model','dataset','image_id','repeat'],30*460*5)]:
        frame = pd.concat(parts,ignore_index=True)
        assert len(frame) == count and not frame.duplicated(keys).any()
        frame.to_csv(merged/name,index=False)
    dump(root/'COMPLETE.json',dict(status='verified',units=30,images=13800,attributions=82800,
         time=time.time(),outputs={p.name:sha(p) for p in merged.glob('*.csv')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/dcu-full-five-20261013'))
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker',choices=jobs())
    parser.add_argument('--phase',choices=['gate','formal'],default='gate')
    args = parser.parse_args()
    root = (ROOT/args.root).resolve()
    if not root.is_relative_to(ROOT): raise ValueError('output root must stay inside this release')
    root.mkdir(parents=True,exist_ok=True)
    if args.worker:
        try: worker(root,args.phase,args.worker)
        except BaseException as error:
            dump(root/args.phase/args.worker/'failure.json',dict(error=repr(error),time=time.time()))
            raise
        return
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(root)
        if args.prepare: return
        try:
            preflight(root)
            pool(root,'gate')
            dump(root/'GATES_PASSED.json',dict(units=30,time=time.time()))
            pool(root,'formal')
            merge(root)
            dump(root/'status.json',dict(state='complete',phase='formal',completed=jobs(),updated_at=time.time()))
        except BaseException as error:
            dump(root/'controller_failure.json',dict(error=repr(error),time=time.time()))
            dump(root/'status.json',dict(state='failed',error=repr(error),updated_at=time.time()))
            raise


if __name__ == '__main__':
    main()
