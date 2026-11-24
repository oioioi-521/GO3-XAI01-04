"""Four-worker 2000/4000/8000 LIME debug comparison using the frozen adapter.

Deploy outside the existing release. Does not freeze or start formal eval.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

JOBS=[f'lime-{m}-{d}' for m in ('resnet50','densenet121','vgg16') for d in ('imagenet','voc')]
BUDGETS=[2000,4000,8000]
SEEDS=[42,43,44]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def original(root,job):return root/'results/dcu-lime-20261013'/job


def smoke_time(root,job):
    old=root/f'results/dcu-lime-20261013/budget/{job}'
    plan=json.loads((old/'plan.json').read_text())
    image_id=str(plan['image_ids'][0])
    records=[json.loads(p.read_text()) for p in (old/'records').glob('*.json')]
    rec=next(r for r in records if r['image_id']==image_id)
    item=next(r for r in rec['runs'] if r['budget']==2000 and r['seed']==42)
    assert sha(old/item['map_path'])==item['map_sha256']
    return item['time_ms']


def command(root,out,job):
    launcher=('import sys,runpy,torch; sys.argv=sys.argv[1:]; '
        'torch.cuda.set_per_process_memory_fraction(.10,0); torch.set_num_threads(2); '
        'runpy.run_path(sys.argv[0],run_name="__main__")')
    return [sys.executable,'-c',launcher,str(root/'experiments/lime_budget_gate.py'),
        '--config',str(root/f'results/dcu-lime-20261013/gate/{job}/unit.yaml'),
        '--output',str(out/job),'--budgets',*map(str,BUDGETS),'--seeds',*map(str,SEEDS),
        '--single-image-ms',str(smoke_time(root,job))]


def verify(root,out,job):
    import numpy as np
    from preprocessing.dataset import get_split
    unit=out/job
    plan=json.loads((unit/'plan.json').read_text());summary=json.loads((unit/'summary.json').read_text())
    assert plan['budgets']==BUDGETS and plan['seeds']==SEEDS and not plan['missing_assets']
    assert summary['n_images']==40 and summary['config_hash']==plan['config_hash']
    dataset=job.rsplit('-',1)[1];frozen=get_split(dataset,'debug')
    targets=dict(zip(frozen.image_id.astype(str),frozen.class_id.astype(int)))
    records=[json.loads(p.read_text()) for p in (unit/'records').glob('*.json')]
    assert len(records)==40 and {r['image_id'] for r in records}==set(targets)
    for r in records:
        assert r['target']==targets[r['image_id']] and r['config_hash']==plan['config_hash']
        assert len(r['runs'])==9 and {(x['budget'],x['seed']) for x in r['runs']}=={(b,s) for b in BUDGETS for s in SEEDS}
        for item in r['runs']:
            path=unit/item['map_path'];assert sha(path)==item['map_sha256']
            a=np.load(path,allow_pickle=False)
            assert a.shape==(224,224) and a.dtype==np.float32 and np.isfinite(a).all()
            assert a.min()>=0 and a.max()<=1 and item['num_samples']==item['budget']
            assert item['segmentation_seed']==42 and np.isfinite(item['weighted_r2']) and item['time_ms']>0
    return dict(status='verified_integrity_only',images=40,maps=360,summary_sha256=sha(unit/'summary.json'),
        plan_sha256=sha(unit/'plan.json'),release_sha256=sha(root/'release-manifest.json'),
        controller_sha256=sha(Path(__file__)),formal_budget_frozen=False)


def done(root,out,job):
    path=out/job/'completion.json'
    if not path.exists():return False
    assert json.loads(path.read_text())==verify(root,out,job)
    return True


def prepare(root,out):
    from experiments.lime_budget_gate import plan as make_plan
    from experiments.run_unit import _load_config
    plans={}
    for job in JOBS:
        cfg=_load_config(root/f'results/dcu-lime-20261013/gate/{job}/unit.yaml')
        _,_,report=make_plan(cfg,BUDGETS,SEEDS,out/job)
        ms=smoke_time(root,job)
        report['estimated_attribution_seconds']=ms/1000*41*3*sum(BUDGETS)/min(BUDGETS)
        report['estimate_caveat']='linear sampling extrapolation; excludes I/O and startup; validate measured gate wall time'
        assert report['missing_assets']==[] and report['expected_images']==40
        path=out/job/'plan.json'
        if path.exists():assert json.loads(path.read_text())['config_hash']==report['config_hash']
        dump(path,report)
        plans[job]=dict(config_hash=report['config_hash'],expected_images=40,attribution_calls=report['attribution_calls'],
            masked_forwards=report['masked_forwards'],estimated_attribution_seconds=report['estimated_attribution_seconds'])
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=JOBS);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();root=args.release.resolve();out=args.output.resolve()
    assert out.is_relative_to(root/'results/dcu-lime-20261013')
    sys.path.insert(0,str(root));sys.path.insert(0,str(root/'scripts'))
    os.environ['TORCH_HOME']=str(root/'.cache/torch')
    out.mkdir(parents=True,exist_ok=True)
    if args.worker:
        try:
            dump(out/args.worker/'progress.json',dict(state='running',started_at=time.time(),pid=os.getpid()))
            subprocess.run(command(root,out,args.worker),check=True,cwd=root)
            dump(out/args.worker/'completion.json',verify(root,out,args.worker))
        except BaseException as e:
            dump(out/args.worker/'failure.json',dict(error=repr(e),time=time.time()));raise
        return
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};started=time.time()
        try:
            from run_dcu_formal_queue import preflight
            preflight(out)
            plans=prepare(root,out)
            manifest=dict(plans=plans,controller_sha256=sha(Path(__file__)),release_sha256=sha(root/'release-manifest.json'),
                devices=[0,1,2,3],max_workers=4,budgets=BUDGETS,seeds=SEEDS,formal_budget_frozen=False)
            path=out/'controller_manifest.json'
            if path.exists():assert json.loads(path.read_text())==manifest
            dump(path,manifest)
            if args.prepare_only:
                print(json.dumps(manifest,indent=2));return
            pending=sorted([j for j in JOBS if not done(root,out,j)],key=lambda j:-plans[j]['estimated_attribution_seconds'])
            failed=[]
            while pending or active:
                for device in range(4):
                    if device in active or not pending:continue
                    job=pending.pop(0);env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None)
                    env.update(HIP_VISIBLE_DEVICES=str(device),CUDA_VISIBLE_DEVICES=str(device),
                        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                    log=(out/job/'worker.log').open('a')
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--release',str(root),
                        '--output',str(out),'--worker',job],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[device]=dict(job=job,process=proc,log=log,started=time.time())
                for device,item in list(active.items()):
                    if item['process'].poll() is None:continue
                    item['log'].close()
                    if item['process'].returncode!=0 or not done(root,out,item['job']):failed.append(item['job'])
                    del active[device]
                running=[dict(device=d,job=v['job'],pid=v['process'].pid,stage='high_budget',
                    budget_images=len(list((out/v['job']/'records').glob('*.json'))),elapsed_seconds=time.time()-v['started'])
                    for d,v in active.items()]
                dump(out/'status.json',dict(state='running',phase='high_budget',updated_at=time.time(),started_at=started,
                    running=running,pending=pending,failed=failed,completed=[j for j in JOBS if (out/j/'completion.json').exists()]))
                if pending or active:time.sleep(5)
            if failed:raise RuntimeError(f'High budget jobs failed: {failed}')
            dump(out/'status.json',dict(state='high_budget_ready',phase='high_budget',completed=JOBS,
                started_at=started,updated_at=time.time(),wall_seconds=time.time()-started,formal_budget_frozen=False))
        except BaseException as e:
            dump(out/'controller_failure.json',dict(error=repr(e),time=time.time()))
            dump(out/'status.json',dict(state='failed',phase='high_budget',updated_at=time.time(),error=repr(e)));raise
        finally:
            for item in active.values():
                if item['process'].poll() is None:
                    os.killpg(item['process'].pid,signal.SIGTERM)
                    try:item['process'].wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(item['process'].pid,signal.SIGKILL);item['process'].wait()
                item['log'].close()


if __name__=='__main__':main()
