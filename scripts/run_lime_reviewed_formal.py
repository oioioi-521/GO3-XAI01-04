"""Run reviewed LIME smoke/formal units with immutable high-budget evidence.

Deploy outside the frozen release; set XAI_RELEASE_ROOT to that release.
The unchanged shared worker checks metrics, five noise repeats and real resume.
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

ROOT = Path(os.environ['XAI_RELEASE_ROOT']).resolve()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_dcu_lime_queue as original
import run_lime_high_budget_queue as high

shared = original.shared
JOBS = original.JOBS
dump, sha = shared.dump, shared.sha


def review_evidence(root, review):
    decision = json.loads(review.read_text())
    assert decision['status'] == 'accepted_for_formal' and decision['evidence_run'] == 'high_budget'
    assert set(decision['units']) == set(JOBS) and decision['num_samples'] in (2000,4000,8000)
    assert decision['release_sha256'] == sha(ROOT/'release-manifest.json')
    assert decision['high_budget_controller_sha256'] == sha(Path(high.__file__))
    audit = root/'high_budget_independent_audit.json'
    assert decision['independent_audit_sha256'] == sha(audit)
    audit_data = json.loads(audit.read_text())
    assert audit_data['status'] == 'integrity_passed_budget_review_pending'
    assert audit_data['maps'] == 2160 and audit_data['recomputed_pair_comparisons'] == 3600
    assert (root/'high_budget/controller.exit').read_text().strip() == '0'
    assert json.loads((root/'high_budget/status.json').read_text())['state'] == 'high_budget_ready'
    for job in JOBS:
        out = root/'high_budget'/job
        check = high.verify(ROOT,root/'high_budget',job)
        assert check == json.loads((out/'completion.json').read_text())
        for k in ('summary_sha256','plan_sha256'):
            assert decision['units'][job][k] == check[k] == audit_data['units'][job][k]
        assert decision['num_samples'] in json.loads((out/'plan.json').read_text())['budgets']
    return decision


def prepare(root, stage, review, decision):
    import yaml
    target = root/'formal_smoke' if stage == 'smoke' else root
    phase = 'gate' if stage == 'smoke' else 'formal'
    for job in JOBS:
        _,model,dataset = job.split('-')
        cfg = yaml.safe_load((ROOT/f'configs/lime_{model}_{dataset}.yaml').read_text())
        assert cfg['method'] == 'lime' and cfg['dataset']['split'] == 'eval' and cfg['runtime']['budget_frozen'] is False
        assert cfg['attribution']['output_activation'] == ('sigmoid' if dataset == 'voc' else 'softmax')
        out = target/phase/job; rel = str(out.relative_to(ROOT))
        cfg['dataset']['split'] = 'debug' if stage == 'smoke' else 'eval'
        cfg['attribution']['num_samples'] = decision['num_samples']
        cfg['runtime'].update(device='cuda:0',max_images=1 if stage == 'smoke' else None,warmup_runs=1,
                              resume=True,budget_frozen=True,budget_review_sha256=sha(review))
        cfg['lime_contract']['budget_status'] = 'Reviewed fixed budget; limitations and evidence bound by budget_review_sha256'
        cfg['output'] = {k:f'{rel}/{v}' for k,v in {
            'per_image_csv':'per_image.csv','units_csv':'units.csv','predictions_csv':'predictions.csv',
            'config_dir':'configs','state_dir':'run_state','run_log':'base_log.jsonl',
            'maps_dir':'maps','float_maps_dir':'maps_float'}.items()}
        cfg['output'].update(save_maps=True,save_float_maps=True)
        protocol = yaml.safe_load((ROOT/'configs/stability_protocol_v1.yaml').read_text())
        protocol['runtime']['device'] = 'cuda:0'
        protocol['output'] = {k:f'{rel}/{v}' for k,v in {'trace_csv':'stability_trace.csv',
            'state_dir':'stability_state','config_dir':'stability_configs','run_log':'stability_log.jsonl'}.items()}
        out.mkdir(parents=True,exist_ok=True)
        for name,value in [('unit.yaml',cfg),('protocol.yaml',protocol)]:
            path=out/name; text=yaml.safe_dump(value,sort_keys=False)
            if path.exists() and path.read_text()!=text:raise ValueError(f'Cannot change existing config: {path}')
            if not path.exists():path.write_text(text)
    return target,phase


def worker(target,phase,job):
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(.10,0);torch.set_num_threads(2)
    shared.worker(target,phase,job)


def pool(root,target,phase,stage,devices):
    pending=[j for j in JOBS if not shared.verified_done(target/phase/j)]
    active={};failed=[];started=time.time()
    status_path=root/('formal_smoke/status.json' if stage=='smoke' else 'status.json')
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending:continue
                job=pending.pop(0);env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None)
                env.update(HIP_VISIBLE_DEVICES=str(device),CUDA_VISIBLE_DEVICES=str(device),
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                log=(target/phase/job/'worker.log').open('a')
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--root',str(root),
                    '--stage',stage,'--worker',job],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[device]=dict(job=job,process=proc,log=log,started=time.time())
            for device,item in list(active.items()):
                if item['process'].poll() is None:continue
                item['log'].close()
                if item['process'].returncode!=0 or not shared.verified_done(target/phase/item['job']):failed.append(item['job'])
                del active[device]
            running=[]
            for device,item in active.items():
                out=target/phase/item['job'];p=out/'progress.json'
                progress=json.loads(p.read_text()) if p.exists() else {'stage':'loading'}
                counts=collections.Counter()
                if (out/'per_image.csv').exists():
                    with (out/'per_image.csv').open() as f:counts.update(r['metric'] for r in csv.DictReader(f))
                running.append(dict(device=device,job=item['job'],pid=item['process'].pid,stage=progress['stage'],
                    base_images=counts['efficiency_time_ms'],stability_images=counts['stability_spearman'],elapsed_seconds=time.time()-item['started']))
            dump(status_path,dict(state='running',phase='formal_smoke' if stage=='smoke' else 'formal',started_at=started,
                updated_at=time.time(),pending=pending,running=running,failed=failed,
                completed=[j for j in JOBS if (target/phase/j/'completion.json').exists()]))
            if pending or active:time.sleep(5)
        if failed:raise RuntimeError(f'{stage} jobs failed: {failed}')
    finally:
        for item in active.values():
            if item['process'].poll() is None:
                os.killpg(item['process'].pid,signal.SIGTERM)
                try:item['process'].wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(item['process'].pid,signal.SIGKILL);item['process'].wait()
            item['log'].close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','formal'],required=True);p.add_argument('--devices',default='0,1,2,3,4,5')
    p.add_argument('--review',type=Path);p.add_argument('--worker',choices=JOBS);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();root=(ROOT/args.root).resolve();assert root.is_relative_to(ROOT/'results')
    target=root/'formal_smoke' if args.stage=='smoke' else root;phase='gate' if args.stage=='smoke' else 'formal'
    if args.worker:
        try:worker(target,phase,args.worker)
        except BaseException as e:dump(target/phase/args.worker/'failure.json',dict(error=repr(e),time=time.time()));raise
        return
    assert args.review is not None;review=(ROOT/args.review).resolve();assert review.is_relative_to(root)
    with (root/'reviewed_controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.time()
        try:
            decision=review_evidence(root,review)
            devices=list(map(int,args.devices.split(',')));assert devices and len(devices)==len(set(devices)) and min(devices)>=0 and max(devices)<8
            preflight_dir=root/f'{args.stage}_preflight';preflight_dir.mkdir(parents=True,exist_ok=True)
            shared.preflight(preflight_dir)
            if args.stage=='formal':
                assert json.loads((root/'formal_smoke/status.json').read_text())['state']=='smoke_complete'
                assert (root/'formal_smoke_controller.exit').read_text().strip()=='0'
                for j in JOBS:assert shared.verified_done(root/'formal_smoke/gate'/j)
            target,phase=prepare(root,args.stage,review,decision)
            manifest=dict(stage=args.stage,num_samples=decision['num_samples'],devices=devices,max_workers=len(devices),units=JOBS,
                review_sha256=sha(review),controller_sha256=sha(Path(__file__)),release_sha256=sha(ROOT/'release-manifest.json'),
                config_sha256={j:sha(target/phase/j/'unit.yaml') for j in JOBS},protocol_sha256={j:sha(target/phase/j/'protocol.yaml') for j in JOBS})
            path=root/f'{args.stage}_controller_manifest.json'
            if path.exists():assert json.loads(path.read_text())==manifest
            else:dump(path,manifest)
            if args.prepare_only:print(json.dumps(manifest,indent=2));return
            dump(root/f'{args.stage}_start.json',dict(time=started,devices=devices,units=JOBS,num_samples=decision['num_samples']))
            pool(root,target,phase,args.stage,devices)
            if args.stage=='formal':original.merge(root)
            dump(target/'status.json',dict(state='smoke_complete' if args.stage=='smoke' else 'complete',phase=args.stage,
                completed=JOBS,started_at=started,updated_at=time.time(),wall_seconds=time.time()-started,num_samples=decision['num_samples']))
        except BaseException as e:
            dump(root/f'{args.stage}_reviewed_controller_failure.json',dict(error=repr(e),time=time.time()))
            dump(target/'status.json',dict(state='failed',phase=args.stage,error=repr(e),updated_at=time.time()));raise


if __name__=='__main__':main()
