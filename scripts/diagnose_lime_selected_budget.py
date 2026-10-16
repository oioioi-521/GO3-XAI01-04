"""Bounded diagnostics on selected frozen debug images, never formal eval.

Keep this script outside the immutable deployed release; point --release to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--model',choices=['densenet121','vgg16'],required=True)
    p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=args.release.resolve(); out=args.output.resolve()
    assert not out.exists(), 'use a fresh diagnostic directory'
    sys.path.insert(0,str(root))
    os.environ['TORCH_HOME']=str(root/'.cache/torch')
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from preprocessing.dataset import MetadataDataset
    from experiments.run_unit import _load_config,_build_attributor,_black_baseline
    from models import load_model
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(.10,0);torch.set_num_threads(2)
    config=_load_config(root/f'configs/lime_{args.model}_imagenet.yaml')
    dataset=MetadataDataset('imagenet',split='debug')
    ids=dataset.rows.image_id.astype(str).tolist()
    assert len(ids)==40 and len(args.ids)==len(set(args.ids)) and set(args.ids)<=set(ids)
    original=root/f'results/dcu-lime-20261013/budget/lime-{args.model}-imagenet'
    old_records={r['image_id']:r for r in (json.loads(p.read_text()) for p in (original/'records').glob('*.json'))}
    model=load_model(name=args.model,device=torch.device('cuda:0'),weights='default',num_classes=1000,output_activation='softmax')
    baseline=_black_baseline(torch.device('cuda:0'),torch.float32)
    manifest=dict(scope='selected debug diagnostics only; not a 40-image gate or formal result',
        model=args.model,dataset='imagenet',image_ids=args.ids,budgets=[4000,8000],seeds=[42,43,44],
        source_script_sha256=sha(Path(__file__)),release_sha256=sha(root/'release-manifest.json'),
        original_summary_sha256=sha(original/'summary.json'),torch=torch.__version__,hip=torch.version.hip,
        visible_device=os.environ.get('HIP_VISIBLE_DEVICES'),started_at=time.time(),config=config)
    dump(out/'manifest.json',manifest)
    first=dataset[ids.index(args.ids[0])]
    _build_attributor('lime',model,{**config['attribution'],'num_samples':4000,'seed':42}).attribute(first['image'][None].cuda(),first['target'],baseline)
    records=[]
    for image_id in args.ids:
        sample=dataset[ids.index(image_id)];x=sample['image'][None].cuda()
        assert int(sample['target'])==int(old_records[image_id]['target'])
        with torch.inference_mode():
            probabilities=model(x).softmax(1)[0]
            prediction=dict(target=sample['target'],target_probability=float(probabilities[sample['target']].cpu()),
                            predicted_class=int(probabilities.argmax().cpu()),predicted_probability=float(probabilities.max().cpu()))
        maps={};runs=[]
        for saved in old_records[image_id]['runs']:
            if saved['budget']==2000:
                path=original/saved['map_path'];assert sha(path)==saved['map_sha256']
                maps[2000,saved['seed']]=np.load(path,allow_pickle=False)
        for budget in (4000,8000):
            for seed in (42,43,44):
                dump(out/'status.json',dict(state='running',image_id=image_id,budget=budget,seed=seed,
                    completed_images=len(records),updated_at=time.time()))
                method=_build_attributor('lime',model,{**config['attribution'],'num_samples':budget,'seed':seed})
                torch.cuda.synchronize();start=time.perf_counter()
                result=method.attribute(x,sample['target'],baseline)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                arr=result.numpy();assert arr.shape==(224,224) and arr.dtype==np.float32 and np.isfinite(arr).all()
                assert arr.min()>=0 and arr.max()<=1
                path=out/f'{image_id}_{budget}_{seed}.npy'
                np.save(path,arr,allow_pickle=False);maps[budget,seed]=arr
                runs.append(dict(**method.last_diagnostics,time_ms=elapsed,map_path=path.name,map_sha256=sha(path)))
                print(image_id,budget,seed,round(elapsed/1000,2),flush=True)
        comparisons=[]
        for budget in (2000,4000,8000):
            for i,a in enumerate((42,43,44)):
                for b in (42,43,44)[i+1:]:
                    comparisons.append(dict(kind='seed',budget=budget,seed_a=a,seed_b=b,
                        spearman=float(spearmanr(maps[budget,a].ravel(),maps[budget,b].ravel()).statistic)))
        for b1,b2 in ((2000,4000),(4000,8000)):
            for seed in (42,43,44):
                comparisons.append(dict(kind='budget',budget_a=b1,budget_b=b2,seed=seed,
                    spearman=float(spearmanr(maps[b1,seed].ravel(),maps[b2,seed].ravel()).statistic)))
        record=dict(image_id=image_id,prediction=prediction,runs=runs,comparisons=comparisons)
        dump(out/f'{image_id}.json',record);records.append(record)
    dump(out/'summary.json',dict(status='diagnostic_complete',records=records,finished_at=time.time(),
        wall_seconds=time.time()-manifest['started_at'],formal_budget_frozen=False))
    dump(out/'status.json',dict(state='complete',completed_images=len(records),updated_at=time.time()))


if __name__=='__main__':main()
