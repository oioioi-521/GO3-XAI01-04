"""Immutable LIME 8000/batch32 overlay: 40-image equivalence, resume gates, eight-card eval.

The six 460-image units are partitioned into sixty disjoint 46-image jobs. No
frozen release file is edited; the config explicitly binds IDs and overlay hash.
"""
from __future__ import annotations
import argparse
import collections
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
JOBS = [f'lime-{m}-{d}' for m in ('vgg16', 'densenet121', 'resnet50') for d in ('imagenet', 'voc')]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)

def immutable(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Immutable manifest changed: {path}'
    else: dump(path, value)

def setup(release):
    os.environ['XAI_RELEASE_ROOT'] = str(release)
    os.environ['TORCH_HOME'] = str(release/'.cache/torch')
    sys.path[:0] = [str(HERE), str(release), str(release/'scripts')]

def overlay_integrity():
    manifest = json.loads((HERE/'overlay-manifest.json').read_text())
    assert manifest['num_samples'] == 8000 and manifest['batch_size'] == 32
    for name, digest in manifest['files'].items():
        assert '/' not in name and sha(HERE/name) == digest, name
    assert manifest['files']['lime_vectorized.py'] == 'c48e5b6b18298cd4ebf5736d4fda46c265011662f29c9a3207bb823f159ab314'
    return sha(HERE/'overlay-manifest.json')

def prepare(args, overlay_sha):
    import yaml
    from preprocessing.dataset import get_split
    from lime_formal_partition import partition_ids
    plan = {'regression': [], 'gate': [], 'formal': []}
    for phase, split, size in [('regression','debug',10), ('gate','debug',1), ('formal','eval',46)]:
        for job in JOBS:
            _, model, dataset = job.split('-')
            ids = get_split(dataset, split).image_id.astype(str).tolist()
            assert len(ids) == (460 if split == 'eval' else 40)
            # Two distant debug IDs per unit exercise explicit slicing, not first-N.
            if phase == 'gate': ids = [ids[0], ids[-1]]
            for index, part in enumerate(partition_ids(ids, size)):
                key = f'{job}__{index:02d}'; out = args.output/phase/key
                cfg = yaml.safe_load((args.release/f'configs/lime_{model}_{dataset}.yaml').read_text())
                assert cfg['method'] == 'lime' and cfg['attribution']['batch_size'] == 32
                cfg['dataset'].update(split=split, image_ids=part)
                cfg['attribution']['num_samples'] = 8000
                cfg['runtime'].update(device='cuda:0',max_images=None,warmup_runs=1,resume=True,
                    budget_frozen=True,budget_review_sha256=sha(HERE/'budget_review_8000.json'),
                    execution_overlay_sha256=overlay_sha,implementation='VectorizedLIME',partition=key)
                cfg['lime_contract']['budget_status'] = 'Reviewed fixed 8000 budget; exact batch32 vectorized execution'
                rel = str(out.relative_to(args.release))
                cfg['output'] = {k:f'{rel}/{v}' for k,v in {
                    'per_image_csv':'per_image.csv','units_csv':'units.csv','predictions_csv':'predictions.csv',
                    'config_dir':'configs','state_dir':'run_state','run_log':'base_log.jsonl',
                    'maps_dir':'maps','float_maps_dir':'maps_float'}.items()}
                cfg['output'].update(save_maps=True,save_float_maps=True)
                protocol = yaml.safe_load((args.release/'configs/stability_protocol_v1.yaml').read_text())
                protocol['runtime']['device'] = 'cuda:0'
                protocol['output'] = {k:f'{rel}/{v}' for k,v in {'trace_csv':'stability_trace.csv',
                    'state_dir':'stability_state','config_dir':'stability_configs','run_log':'stability_log.jsonl'}.items()}
                out.mkdir(parents=True, exist_ok=True)
                for name, data in [('unit.yaml',cfg), ('protocol.yaml',protocol)]:
                    p=out/name; text=yaml.safe_dump(data,sort_keys=False)
                    if p.exists(): assert p.read_text()==text
                    else: p.write_text(text)
                plan[phase].append(dict(key=key,unit=job,image_ids=part,config_sha256=sha(out/'unit.yaml'),
                    protocol_sha256=sha(out/'protocol.yaml')))
    immutable(args.output/'plan.json', plan)
    return plan

def regression(args, out, cfg):
    import numpy as np
    import torch
    from models import load_model
    from preprocessing.dataset import MetadataDataset
    from experiments.run_unit import _black_baseline
    from lime_vectorized import VectorizedLIME
    job = args.worker.split('__')[0]; _, model, dataset = job.split('-')
    old = args.release/'results/dcu-lime-20261013/high_budget'/job
    saved = {r['image_id']:r for r in (json.loads(p.read_text()) for p in (old/'records').glob('*.json'))}
    mc=cfg['model']; torch.manual_seed(42)
    network=load_model(name=model,device=torch.device('cuda:0'),weights=mc['weights'],
        num_classes=mc['num_classes'],output_activation=mc['output_activation'],
        checkpoint=str(args.release/mc['checkpoint']) if mc.get('checkpoint') else None)
    baseline=_black_baseline(torch.device('cuda:0'),torch.float32)
    data=MetadataDataset(dataset,split='debug'); rows=[]
    for i in range(len(data)):
        sample=data[i]; image_id=sample['image_id']; target=int(sample['target'])
        assert saved[image_id]['target']==target
        for seed in (42,43,44):
            method=VectorizedLIME(model=network,**{**cfg['attribution'],'seed':seed})
            torch.cuda.synchronize(); tick=time.perf_counter()
            a=method.attribute(sample['image'][None].cuda(),target,baseline).numpy()
            torch.cuda.synchronize(); seconds=time.perf_counter()-tick
            ref=next(r for r in saved[image_id]['runs'] if r['budget']==8000 and r['seed']==seed)
            path=old/ref['map_path']; assert sha(path)==ref['map_sha256']
            b=np.load(path,allow_pickle=False)
            assert a.shape==b.shape==(224,224) and a.dtype==b.dtype==np.float32 and np.isfinite(a).all()
            # Batch32 probe was bit-identical; any regression blocks formal automatically.
            assert np.array_equal(a,b), f'Non-exact optimized map: {job} {image_id} seed{seed}'
            assert method.last_diagnostics['weighted_r2']==ref['weighted_r2'], (image_id,seed,'R2 changed')
            p=out/f'{i:02d}_{seed}.npy'; np.save(p,a,allow_pickle=False)
            rows.append(dict(image_id=image_id,target=target,seed=seed,num_samples=8000,exact=True,
                weighted_r2=ref['weighted_r2'],map_path=p.name,map_sha256=sha(p),reference_sha256=ref['map_sha256'],seconds=seconds))
            dump(out/'records.json',rows)
            dump(out/'progress.json',dict(stage='regression',maps=len(rows),time=time.time()))
    dump(out/'completion.json',dict(status='verified_exact',images=len(data),maps=len(rows),
        records_sha256=sha(out/'records.json'),config_sha256=sha(out/'unit.yaml'),
        overlay_sha256=overlay_integrity(),map_sha256={r['map_path']:r['map_sha256'] for r in rows}))

def validate(out, count):
    import numpy as np
    import pandas as pd
    import yaml
    import run_dcu_formal_queue as shared
    from experiments.predictions import build_prediction_context
    cfg=yaml.safe_load((out/'unit.yaml').read_text())
    result=shared.validate(out,count)  # get_split is scoped to the explicit config IDs.
    predictions=pd.read_csv(out/'predictions.csv',dtype={'image_id':str})
    mc=cfg['model']; release=shared.ROOT
    context=build_prediction_context(dataset=cfg['dataset']['name'],split=cfg['dataset']['split'],
        model_config=mc,checkpoint=release/mc['checkpoint'] if mc.get('checkpoint') else None)
    assert set(predictions.config_hash)=={context['config_hash']}
    assert set(predictions.checkpoint_sha256)=={context['checkpoint_sha256']}
    assert np.isfinite(predictions.confidence).all()
    trace=pd.read_csv(next(out.glob('stability_trace_*.csv')),dtype={'image_id':str})
    frame=pd.read_csv(out/'per_image.csv',dtype={'image_id':str})
    valid=trace.assign(valid=(trace.status=='valid').astype(float)).groupby('image_id').valid.mean()
    reported=frame[frame.metric=='stability_valid_rate'].set_index('image_id').value
    assert np.allclose(valid.sort_index(),reported.sort_index(),rtol=0,atol=1e-12)
    result.update(config_sha256=sha(out/'unit.yaml'),protocol_sha256=sha(out/'protocol.yaml'),
        overlay_sha256=overlay_integrity(),map_sha256={str(p.relative_to(out)):sha(p) for p in out.glob('maps_float/**/*.npy*')})
    return result

def worker(args):
    import torch
    import yaml
    from experiments import run_unit, run_stability
    from lime_vectorized import VectorizedLIME
    from lime_formal_partition import execution_partition
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.10,0)
    out=args.output/args.phase/args.worker
    cfg=yaml.safe_load((out/'unit.yaml').read_text())
    assert cfg['runtime']['execution_overlay_sha256']==overlay_integrity()
    assert cfg['attribution']['num_samples']==8000 and cfg['attribution']['batch_size']==32
    dump(out/'environment.json',dict(pid=os.getpid(),torch=torch.__version__,hip=torch.version.hip,
        visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(0),
        memory_fraction=.10,threads=2,started_at=time.time()))
    with execution_partition(cfg,VectorizedLIME):
        if args.phase=='regression': regression(args,out,cfg); return
        callbacks=[('base',lambda:run_unit.run(out/'unit.yaml'),'base_log.jsonl'),
                   ('stability',lambda:run_stability.run(out/'unit.yaml',out/'protocol.yaml'),'stability_log.jsonl')]
        for stage,callback,log in callbacks:
            dump(out/'progress.json',dict(stage=stage,time=time.time())); callback()
            gc.collect(); torch.cuda.empty_cache()
        n=len(cfg['dataset']['image_ids']); validate(out,n)
        dump(out/'progress.json',dict(stage='resume_check',time=time.time()))
        for stage,callback,log in callbacks:
            callback(); record=json.loads((out/log).read_text().splitlines()[-1])
            assert (record['processed'],record['skipped'])==(0,n),record
            gc.collect(); torch.cuda.empty_cache()
        result=validate(out,n); result['resume_verified']=True
        dump(out/'completion.json',result)
        dump(out/'progress.json',dict(stage='completed',time=time.time()))

def completed(out, phase, overlay_sha):
    p=out/'completion.json'
    if not p.exists(): return False
    r=json.loads(p.read_text())
    assert r['overlay_sha256']==overlay_sha and r['config_sha256']==sha(out/'unit.yaml')
    if phase=='regression':
        assert r['status']=='verified_exact' and sha(out/'records.json')==r['records_sha256']
    else:
        assert r['status']=='verified' and r['resume_verified']
        assert r['protocol_sha256']==sha(out/'protocol.yaml')
        for name,digest in r['csv_sha256'].items(): assert sha(out/name)==digest
    for name,digest in r['map_sha256'].items(): assert sha(out/name)==digest
    return True

def pool(args,phase,jobs,overlay_sha):
    pending=[j['key'] for j in jobs if not completed(args.output/phase/j['key'],phase,overlay_sha)]
    done=[j['key'] for j in jobs if j['key'] not in pending]; sizes={j['key']:len(j['image_ids']) for j in jobs}
    active={};failed=[];started=time.time()
    immutable(args.output/f'{phase}_start.json',dict(time=started,devices=args.devices))
    try:
        while pending or active:
            for device in args.devices:
                if device in active or not pending or failed: continue
                key=pending.pop(0);out=args.output/phase/key
                env=os.environ.copy();env.pop('ROCR_VISIBLE_DEVICES',None)
                env.update(HIP_VISIBLE_DEVICES=str(device),CUDA_VISIBLE_DEVICES=str(device),
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                log=(out/'worker.log').open('a')
                process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--release',str(args.release),
                    '--output',str(args.output),'--phase',phase,'--worker',key],cwd=args.release,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[device]=dict(key=key,process=process,log=log,started=time.time())
            for device,item in list(active.items()):
                if item['process'].poll() is None:continue
                item['log'].close()
                if item['process'].returncode==0 and completed(args.output/phase/item['key'],phase,overlay_sha):done.append(item['key'])
                else:failed.append(item['key'])
                del active[device]
            running=[];base=sum(sizes[k] for k in done);stability=base;maps=base*3
            for device,item in active.items():
                out=args.output/phase/item['key'];p=out/'progress.json'
                prog=json.loads(p.read_text()) if p.exists() else {'stage':'loading'}
                import csv
                counts=collections.Counter()
                if (out/'per_image.csv').exists():
                    with (out/'per_image.csv').open() as f:counts.update(r['metric'] for r in csv.DictReader(f))
                b=counts['efficiency_time_ms'];s=counts['stability_spearman'];base+=b;stability+=s;maps+=prog.get('maps',0)
                running.append(dict(device=device,job=item['key'],pid=item['process'].pid,stage=prog['stage'],
                    base_images=b,stability_images=s,regression_maps=prog.get('maps',0),elapsed_seconds=time.time()-item['started']))
            dump(args.output/'status.json',dict(state='running',phase=phase,updated_at=time.time(),started_at=started,
                pending=pending,running=running,failed=failed,completed=done,total_jobs=len(jobs),
                base_images=base,stability_images=stability,regression_maps=maps,
                expected_images=sum(sizes.values()),formal_started=phase=='formal'))
            if failed and not active:raise RuntimeError(f'{phase} failed: {failed}; pending not launched')
            if pending or active:time.sleep(5)
        dump(args.output/f'{phase}_COMPLETE.json',dict(status='verified',jobs=len(jobs),time=time.time(),
            wall_seconds=time.time()-started,completions={j['key']:sha(args.output/phase/j['key']/'completion.json') for j in jobs}))
    finally:
        for item in active.values():
            if item['process'].poll() is None:
                os.killpg(item['process'].pid,signal.SIGTERM)
                try:item['process'].wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(item['process'].pid,signal.SIGKILL);item['process'].wait()
            item['log'].close()

def merge(args,plan,overlay_sha):
    import numpy as np
    import pandas as pd
    import yaml
    from preprocessing.dataset import get_split
    from experiments.run_unit import _config_hash
    from lime_formal_partition import execution_partition,summarize,aggregate_hash
    from lime_vectorized import VectorizedLIME
    frames=[];traces=[];predictions=[];summaries=[];map_index=[]
    for unit in JOBS:
        parts=[j for j in plan['formal'] if j['unit']==unit]
        expected=set(get_split(unit.rsplit('-',1)[1],'eval').image_id.astype(str));seen=[];hashes=[]
        uf=[];ut=[];up=[]
        for job in parts:
            out=args.output/'formal'/job['key'];assert completed(out,'formal',overlay_sha)
            cfg=yaml.safe_load((out/'unit.yaml').read_text());seen+=cfg['dataset']['image_ids'];hashes.append(_config_hash(cfg))
            with execution_partition(cfg,VectorizedLIME):validate(out,len(job['image_ids']))
            uf.append(pd.read_csv(out/'per_image.csv',dtype={'image_id':str}))
            ut.append(pd.read_csv(next(out.glob('stability_trace_*.csv')),dtype={'image_id':str,'seed':str}))
            up.append(pd.read_csv(out/'predictions.csv',dtype={'image_id':str}))
            for p in out.glob('maps_float/**/*.npy'):
                map_index.append(dict(unit=unit,image_id=p.stem,path=str(p.relative_to(args.output)),sha256=sha(p),
                    provenance_sha256=sha(Path(str(p)+'.provenance.json')),base_config_hash=_config_hash(cfg),shard=job['key']))
        assert len(seen)==len(set(seen))==460 and set(seen)==expected
        f=pd.concat(uf,ignore_index=True);t=pd.concat(ut,ignore_index=True);p=pd.concat(up,ignore_index=True)
        assert len(f)==1840 and not f.duplicated(['image_id','metric']).any()
        assert len(t)==2300 and not t.duplicated(['image_id','repeat']).any()
        assert len(p)==460 and not p.duplicated('image_id').any()
        summary=summarize(f,aggregate_hash(hashes));assert len(summary)==4 and (summary.n==460).all()
        out=args.output/'units'/unit;out.mkdir(parents=True,exist_ok=True)
        for name,table in [('per_image.csv',f),('stability_trace.csv',t),('predictions.csv',p),('units.csv',summary)]:table.to_csv(out/name,index=False)
        dump(out/'completion.json',dict(status='verified',resume_verified=True,images=460,shards=[j['key'] for j in parts],
            aggregate_config_hash=aggregate_hash(hashes),shard_config_hashes=hashes,
            outputs={p.name:sha(p) for p in out.glob('*.csv')}))
        frames.append(f);traces.append(t);predictions.append(p);summaries.append(summary)
    merged=args.output/'merged';merged.mkdir(exist_ok=True)
    for name,parts,keys,n in [('per_image.csv',frames,['model','dataset','image_id','metric'],11040),
        ('stability_trace.csv',traces,['model','dataset','image_id','repeat'],13800),
        ('predictions.csv',predictions,['model','dataset','image_id'],2760),
        ('units.csv',summaries,['model','dataset','metric'],24)]:
        table=pd.concat(parts,ignore_index=True);assert len(table)==n and not table.duplicated(keys).any()
        table.to_csv(merged/name,index=False)
    assert len(map_index)==2760;dump(merged/'maps_index.json',map_index)
    import run_dcu_formal_queue as shared
    (args.output/'postrun').mkdir(exist_ok=True);shared.preflight(args.output/'postrun')
    dump(args.output/'COMPLETE.json',dict(status='verified',units=6,shards=60,images=2760,attributions=16560,
        resume_verified=True,overlay_sha256=overlay_sha,time=time.time(),
        outputs={p.name:sha(p) for p in merged.iterdir() if p.is_file()},
        unit_completions={u:sha(args.output/'units'/u/'completion.json') for u in JOBS}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--devices',default='0,1,2,3,4,5,6,7');p.add_argument('--phase',choices=['regression','gate','formal'])
    p.add_argument('--worker');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();args.release=args.release.resolve();args.output=args.output.resolve()
    assert args.output.is_relative_to(args.release/'results')
    setup(args.release);overlay_sha=overlay_integrity()
    if args.worker:
        assert args.phase and '/' not in args.worker
        try:worker(args)
        except BaseException as e:dump(args.output/args.phase/args.worker/'failure.json',dict(error=repr(e),time=time.time()));raise
        return
    args.devices=list(map(int,args.devices.split(',')));assert args.devices==list(range(8))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # Read-only revalidation binds the already reviewed full high-budget evidence.
            from run_lime_reviewed_formal import review_evidence
            decision=review_evidence(args.release/'results/dcu-lime-20261013',HERE/'budget_review_8000.json')
            assert decision['num_samples']==8000
            audit=json.loads((HERE/'optimization_audit.json').read_text())
            assert audit['integrity']=='passed' and audit['pairs_verified']==192 and audit['maps_verified']==276
            assert audit['saved_high_budget_anchors_verified']==54
            for u in audit['units'].values():assert u['exact_pairs']['vector32']==14
            import run_dcu_formal_queue as shared
            shared.preflight(args.output)
            plan=prepare(args,overlay_sha)
            immutable(args.output/'manifest.json',dict(overlay_sha256=overlay_sha,release_sha256=sha(args.release/'release-manifest.json'),
                budget_review_sha256=sha(HERE/'budget_review_8000.json'),plan_sha256=sha(args.output/'plan.json'),
                num_samples=8000,batch_size=32,devices=args.devices,regression_maps=720,formal_shards=60,
                formal_images=2760,formal_attributions=16560,per_process_memory_fraction=.10,cpu_threads=2))
            if args.prepare_only:return
            assert not (args.output/'regression_start.json').exists(), 'Existing run must be reviewed, never automatically restarted'
            for phase in ('regression','gate','formal'):pool(args,phase,plan[phase],overlay_sha)
            merge(args,plan,overlay_sha)
            start=json.loads((args.output/'formal_start.json').read_text())['time']
            dump(args.output/'status.json',dict(state='complete',phase='formal',updated_at=time.time(),
                started_at=start,wall_seconds=time.time()-start,base_images=2760,stability_images=2760,
                formal_started=True,completed=[j['key'] for j in plan['formal']],failed=[],running=[],pending=[]))
        except BaseException as e:
            dump(args.output/'controller_failure.json',dict(error=repr(e),time=time.time()))
            dump(args.output/'status.json',dict(state='failed',error=repr(e),updated_at=time.time()));raise

if __name__=='__main__':main()
