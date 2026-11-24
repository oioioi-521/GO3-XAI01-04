"""Run six real LIME budget gates, then a separately reviewed formal matrix.

Budget mode deliberately ends at budget_ready. Formal mode requires an explicit
review artifact bound to all six gate summaries and the chosen tested budget.
The model adapter, 40-image budget experiment and formal runners are unchanged.
"""
from __future__ import annotations

import argparse
import collections
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_dcu_formal_queue as shared

shared.METHODS = ('lime',)
JOBS = shared.jobs()
dump, sha = shared.dump, shared.sha


def prepare(root, phase, review=None):
    import yaml
    if phase == 'formal':
        decision = json.loads(review.read_text())
        assert decision['status'] == 'accepted_for_formal'
        assert set(decision['units']) == set(JOBS)
        budget = decision['num_samples']
        for job in JOBS:
            assert decision['units'][job]['summary_sha256'] == sha(root/'budget'/job/'summary.json')
            plan = json.loads((root/'budget'/job/'plan.json').read_text())
            assert budget in plan['budgets']
            verify_budget(root, job)
    for job in JOBS:
        _, model, dataset = job.split('-')
        cfg = yaml.safe_load((ROOT/f'configs/lime_{model}_{dataset}.yaml').read_text())
        assert cfg['method'] == 'lime' and cfg['dataset']['split'] == 'eval'
        assert cfg['runtime']['budget_frozen'] is False
        assert cfg['attribution']['output_activation'] == ('sigmoid' if dataset == 'voc' else 'softmax')
        target_phase = 'gate' if phase == 'budget' else 'formal'
        out = root/target_phase/job
        rel = str(out.relative_to(ROOT))
        cfg['dataset']['split'] = 'debug' if phase == 'budget' else 'eval'
        cfg['runtime'].update(device='cuda:0', max_images=1 if phase == 'budget' else None,
                              warmup_runs=1, resume=True)
        if phase == 'formal':
            cfg['attribution']['num_samples'] = budget
            cfg['runtime'].update(budget_frozen=True, budget_review_sha256=sha(review))
            cfg['lime_contract']['budget_status'] = 'DCU reviewed; see frozen budget review SHA-256'
        cfg['output'] = {k:f'{rel}/{v}' for k,v in {
            'per_image_csv':'per_image.csv', 'units_csv':'units.csv',
            'predictions_csv':'predictions.csv', 'config_dir':'configs',
            'state_dir':'run_state', 'run_log':'base_log.jsonl',
            'maps_dir':'maps', 'float_maps_dir':'maps_float'}.items()}
        cfg['output'].update(save_maps=True, save_float_maps=True)
        protocol = yaml.safe_load((ROOT/'configs/stability_protocol_v1.yaml').read_text())
        protocol['runtime']['device'] = 'cuda:0'
        protocol['output'] = {k:f'{rel}/{v}' for k,v in {
            'trace_csv':'stability_trace.csv', 'state_dir':'stability_state',
            'config_dir':'stability_configs', 'run_log':'stability_log.jsonl'}.items()}
        out.mkdir(parents=True, exist_ok=True)
        for name, value in [('unit.yaml',cfg), ('protocol.yaml',protocol)]:
            path = out/name
            text = yaml.safe_dump(value, sort_keys=False)
            if path.exists() and path.read_text() != text:
                raise ValueError(f'Refusing to change existing config: {path}')
            path.write_text(text)
        (root/'budget'/job).mkdir(parents=True, exist_ok=True)


def verify_budget(root, job):
    import numpy as np
    from preprocessing.dataset import get_split
    out = root/'budget'/job
    _, model, dataset = job.split('-')
    plan = json.loads((out/'plan.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    assert summary['n_images'] == 40 and summary['config_hash'] == plan['config_hash']
    assert plan['missing_assets'] == [] and plan['budgets'] == [1000,2000]
    assert plan['seeds'] == [42,43,44]
    expected = get_split(dataset, 'debug')
    targets = dict(zip(expected.image_id.astype(str), expected.class_id.astype(int)))
    records = [json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
    assert len(records) == 40 and {r['image_id'] for r in records} == set(targets)
    assert set(map(str,plan['image_ids'])) == set(targets)
    for record in records:
        assert record['config_hash'] == plan['config_hash'] and record['target'] == targets[record['image_id']]
        assert len(record['runs']) == 6
        assert {(r['budget'],r['seed']) for r in record['runs']} == {(b,s) for b in (1000,2000) for s in (42,43,44)}
        for item in record['runs']:
            path = out/item['map_path']
            assert sha(path) == item['map_sha256']
            array = np.load(path, allow_pickle=False)
            assert array.shape == (224,224) and array.dtype == np.float32
            assert np.isfinite(array).all() and array.min() >= 0 and array.max() <= 1
            assert np.isfinite(item['time_ms']) and item['time_ms'] > 0
            assert item['num_samples'] == item['budget'] and item['segmentation_seed'] == 42
    return dict(status='verified', images=40, maps=240, summary_sha256=sha(out/'summary.json'),
                plan_sha256=sha(out/'plan.json'), release_sha256=sha(ROOT/'release-manifest.json'))


def worker(root, phase, job):
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    # Share the node without consuming the memory reserved by its vLLM service.
    torch.cuda.set_per_process_memory_fraction(0.10, 0)
    torch.set_num_threads(2)
    if phase == 'formal':
        shared.worker(root,'formal',job)
        return
    # Real 1000-sample smoke includes five stability repeats and both resume paths.
    shared.worker(root,'gate',job)
    with (root/'gate'/job/'per_image.csv').open() as handle:
        ms = next(float(r['value']) for r in csv.DictReader(handle) if r['metric']=='efficiency_time_ms')
    out = root/'budget'/job
    dump(out/'progress.json',dict(stage='budget',time=time.time(),pid=os.getpid(),smoke_time_ms=ms))
    # A child interpreter does not inherit the parent's allocator limit.
    launcher = ('import sys,runpy,torch; sys.argv=sys.argv[1:]; '
                'torch.cuda.set_per_process_memory_fraction(0.10,0); '
                'torch.set_num_threads(2); runpy.run_path(sys.argv[0],run_name="__main__")')
    command = [sys.executable,'-c',launcher,str(ROOT/'experiments/lime_budget_gate.py'),
               '--config',str(root/'gate'/job/'unit.yaml'), '--output',str(out),
               '--budgets','1000','2000','--seeds','42','43','44','--single-image-ms',str(ms)]
    subprocess.run(command+['--plan-only'],check=True,cwd=ROOT)
    subprocess.run(command,check=True,cwd=ROOT)
    result = verify_budget(root,job)
    dump(out/'completion.json',result)
    dump(out/'progress.json',dict(stage='budget_ready',time=time.time(),pid=os.getpid()))


def done(root, phase, job):
    out = root/phase/job
    if not (out/'completion.json').exists(): return False
    if phase == 'formal': return shared.verified_done(out)
    verify_budget(root,job)
    return True


def pool(root, phase, devices):
    pending = [j for j in JOBS if not done(root,phase,j)]
    active, failures = {}, []
    started = time.time()
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending: continue
                job = pending.pop(0)
                env = os.environ.copy()
                env.pop('ROCR_VISIBLE_DEVICES',None)
                env.update(HIP_VISIBLE_DEVICES=str(device),CUDA_VISIBLE_DEVICES=str(device),
                           OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                out = root/phase/job
                log = (out/'worker.log').open('a')
                cmd = [sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--phase',phase,'--worker',job]
                process = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[device] = dict(job=job,process=process,log=log,started=time.time())
            for device,item in list(active.items()):
                code = item['process'].poll()
                if code is None: continue
                item['log'].close()
                if code != 0 or not done(root,phase,item['job']): failures.append(item['job'])
                del active[device]
            running = []
            for device,item in active.items():
                job = item['job']; out = root/phase/job
                progress_path = out/'progress.json'
                if phase == 'budget' and not progress_path.exists(): progress_path = root/'gate'/job/'progress.json'
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'stage':'loading'}
                counts = collections.Counter()
                base = root/('gate' if phase == 'budget' else phase)/job/'per_image.csv'
                if base.exists():
                    with base.open() as handle: counts.update(r['metric'] for r in csv.DictReader(handle))
                running.append(dict(device=device,job=job,pid=item['process'].pid,stage=progress['stage'],
                    budget_images=len(list((out/'records').glob('*.json'))),base_images=counts['efficiency_time_ms'],
                    stability_images=counts['stability_spearman'],elapsed_seconds=time.time()-item['started']))
            dump(root/'status.json',dict(state='running',phase=phase,started_at=started,updated_at=time.time(),
                pending=pending,running=running,failed=failures,
                completed=[j for j in JOBS if (root/phase/j/'completion.json').exists()]))
            if pending or active: time.sleep(5)
        if failures: raise RuntimeError(f'{phase} jobs failed: {failures}')
    finally:
        for item in active.values():
            if item['process'].poll() is None:
                os.killpg(item['process'].pid,signal.SIGTERM)
                try: item['process'].wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(item['process'].pid,signal.SIGKILL); item['process'].wait()
            item['log'].close()


def merge(root):
    import pandas as pd
    merged = root/'merged'; merged.mkdir(exist_ok=True)
    for name,pattern,keys,count in [
        ('per_image.csv','per_image.csv',['method','model','dataset','image_id','metric'],11040),
        ('units.csv','units.csv',['method','model','dataset','metric'],24),
        ('stability_trace.csv','stability_trace_*.csv',['method','model','dataset','image_id','repeat'],13800)]:
        parts = []
        for job in JOBS:
            assert done(root,'formal',job)
            paths = list((root/'formal'/job).glob(pattern)); assert len(paths)==1
            parts.append(pd.read_csv(paths[0],dtype={'image_id':str,'seed':str}))
        frame = pd.concat(parts,ignore_index=True)
        assert len(frame)==count and not frame.duplicated(keys).any()
        frame.to_csv(merged/name,index=False)
    dump(root/'COMPLETE.json',dict(status='verified',units=6,images=2760,attributions=16560,
         time=time.time(),outputs={p.name:sha(p) for p in merged.glob('*.csv')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/dcu-lime-20261013'))
    parser.add_argument('--phase',choices=['budget','formal'],default='budget')
    parser.add_argument('--devices',default='0,1,2,3')
    parser.add_argument('--review',type=Path)
    parser.add_argument('--worker',choices=JOBS)
    args = parser.parse_args()
    root = (ROOT/args.root).resolve()
    assert root.is_relative_to(ROOT)
    root.mkdir(parents=True,exist_ok=True)
    if args.worker:
        try: worker(root,args.phase,args.worker)
        except BaseException as error:
            dump(root/args.phase/args.worker/'failure.json',dict(error=repr(error),time=time.time()))
            raise
        return
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started = time.time()
        try:
            if args.phase == 'formal' and args.review is None:
                raise ValueError('formal mode requires --review with accepted budget evidence')
            prepare(root,args.phase,(ROOT/args.review).resolve() if args.review else None)
            shared.preflight(root)
            devices = [int(v) for v in args.devices.split(',')]
            assert len(set(devices)) == len(devices) and devices and min(devices)>=0 and max(devices)<8
            dump(root/f'{args.phase}_start.json',dict(time=started,devices=devices,units=JOBS))
            pool(root,args.phase,devices)
            if args.phase == 'formal': merge(root)
            state = 'complete' if args.phase == 'formal' else 'budget_ready'
            dump(root/'status.json',dict(state=state,phase=args.phase,completed=JOBS,updated_at=time.time(),
                started_at=started,wall_seconds=time.time()-started))
        except BaseException as error:
            dump(root/f'{args.phase}_controller_failure.json',dict(error=repr(error),time=time.time()))
            dump(root/'status.json',dict(state='failed',phase=args.phase,error=repr(error),updated_at=time.time()))
            raise


if __name__ == '__main__': main()
