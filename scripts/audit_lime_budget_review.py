"""Independently recompute the six DCU LIME budget comparisons from saved maps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a,.1)), minimum=float(a.min()), maximum=float(a.max()))


def comparison(a,b):
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    if np.ptp(a)<=1e-12 or np.ptp(b)<=1e-12:
        return dict(status='constant_map',spearman=None,rmse=rmse)
    return dict(status='valid',spearman=float(spearmanr(a.ravel(),b.ravel()).statistic),rmse=rmse)


def match(a,b):
    assert a['status']==b['status']
    for name in ('spearman','rmse'):
        if a[name] is None: assert b[name] is None
        else: assert np.isclose(a[name],b[name],rtol=1e-6,atol=1e-8),(name,a,b)


def audit(artifact):
    metadata = pd.read_csv(ROOT/'data/metadata.csv',dtype={'image_id':str})
    manifest = json.loads((artifact/'release-manifest.json').read_text())
    for item in manifest['files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    voc_weights=json.loads((ROOT/'docs/VOC20_CHECKPOINT_MANIFEST.json').read_text())['checkpoints']
    run = artifact/'run'
    assert (run/'budget_controller.exit').read_text().strip()=='0'
    status = json.loads((run/'status.json').read_text())
    assert status['state']=='budget_ready' and len(status['completed'])==6
    units = {}
    for model in ('resnet50','densenet121','vgg16'):
        for dataset in ('imagenet','voc'):
            job=f'lime-{model}-{dataset}'
            out=run/'budget'/job
            plan=json.loads((out/'plan.json').read_text())
            summary=json.loads((out/'summary.json').read_text())
            completion=json.loads((out/'completion.json').read_text())
            assert completion['status']=='verified' and completion['maps']==240
            assert completion['release_sha256']==sha(artifact/'release-manifest.json')
            assert completion['summary_sha256']==sha(out/'summary.json')
            assert completion['plan_sha256']==sha(out/'plan.json')
            assert summary['n_images']==40 and summary['config_hash']==plan['config_hash']
            assert plan['budgets']==[1000,2000] and plan['seeds']==[42,43,44] and not plan['missing_assets']
            cfg=plan['config']
            context=plan['prediction_context']
            weight_list=manifest['imagenet_weights'] if dataset=='imagenet' else voc_weights
            expected_weight=next(item['sha256'] for item in weight_list
                if item.get('model')==model or item.get('file','').startswith(model))
            assert context['checkpoint_sha256']==expected_weight
            assert cfg['dataset']=={'name':dataset,'split':'debug'}
            assert cfg['model']['name']==model and cfg['runtime']['budget_frozen'] is False
            assert cfg['attribution']['output_activation']==('sigmoid' if dataset=='voc' else 'softmax')
            frozen=metadata[(metadata.dataset==dataset)&(metadata.split=='debug')]
            targets=dict(zip(frozen.image_id,frozen.class_id.astype(int)))
            assert len(targets)==40 and set(map(str,plan['image_ids']))==set(targets)
            records=[json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
            assert len(records)==40 and {r['image_id'] for r in records}==set(targets)
            timing=pd.read_csv(out/'timings.csv',dtype={'image_id':str})
            assert len(timing)==240 and not timing.duplicated(['image_id','budget','seed']).any()
            pairs={1000:[],2000:[]}; variances={1000:[],2000:[]}; runs={1000:[],2000:[]}
            budget_pairs=[]; worst=[]; segment_counts=[]; constant_counts={1000:0,2000:0}
            for r in records:
                assert r['target']==targets[r['image_id']] and r['config_hash']==plan['config_hash']
                assert len(r['runs'])==6 and {(x['budget'],x['seed']) for x in r['runs']}=={(b,s) for b in (1000,2000) for s in (42,43,44)}
                maps={}
                for item in r['runs']:
                    path=out/item['map_path']; assert sha(path)==item['map_sha256']
                    a=np.load(path,allow_pickle=False)
                    assert a.shape==(224,224) and a.dtype==np.float32 and np.isfinite(a).all()
                    assert a.min()>=0 and a.max()<=1
                    assert item['num_samples']==item['budget'] and item['segmentation_seed']==42
                    assert np.isfinite(item['weighted_r2']) and item['time_ms']>0
                    selected=timing[(timing.image_id==r['image_id'])&(timing.budget==item['budget'])&(timing.seed==item['seed'])]
                    assert len(selected)==1 and np.isclose(selected.iloc[0].time_ms,item['time_ms'])
                    maps[item['budget'],item['seed']]=a
                    runs[item['budget']].append(item)
                    constant_counts[item['budget']]+=int(np.ptp(a)<=1e-12)
                assert len({item['n_superpixels'] for item in r['runs']})==1
                segment_counts.append(r['runs'][0]['n_superpixels'])
                assert len(r['budget_comparisons'])==3 and {x['seed'] for x in r['budget_comparisons']}=={42,43,44}
                for item in r['budget_comparisons']:
                    c=comparison(maps[1000,item['seed']],maps[2000,item['seed']]); match(c,item)
                    budget_pairs.append(dict(image_id=r['image_id'],seed=item['seed'],**c))
                for v in r['seed_variance']:
                    b=v['budget']; a=np.stack([maps[b,s] for s in (42,43,44)])
                    variance=float(a.var(0,ddof=1).mean())
                    assert np.isclose(variance,v['pixel_variance_mean'],rtol=1e-6,atol=1e-9)
                    variances[b].append(variance)
                    assert len(v['seed_pairs'])==3
                    for item in v['seed_pairs']:
                        c=comparison(maps[b,item['seed_a']],maps[b,item['seed_b']]); match(c,item)
                        pairs[b].append(dict(image_id=r['image_id'],seed_a=item['seed_a'],seed_b=item['seed_b'],**c))
            budgets={}
            for b in (1000,2000):
                valid=[x for x in pairs[b] if x['status']=='valid']
                assert len(valid)==len(pairs[b])==120
                budgets[str(b)]=dict(time_ms=stats([x['time_ms'] for x in runs[b]]),
                    weighted_r2=stats([x['weighted_r2'] for x in runs[b]]),
                    seed_spearman=stats([x['spearman'] for x in valid]),
                    seed_rmse=stats([x['rmse'] for x in valid]), pixel_variance=stats(variances[b]),
                    constant_maps=constant_counts[b], worst_seed_pairs=sorted(valid,key=lambda x:x['spearman'])[:8],
                    negative_seed_pairs=sum(x['spearman']<0 for x in valid),
                    seed_pairs_below_half=sum(x['spearman']<.5 for x in valid))
                assert np.isclose(budgets[str(b)]['time_ms']['mean'],summary['mean_time_ms'][str(b)])
                assert np.isclose(np.mean(variances[b]),summary['mean_pixel_variance'][str(b)])
            valid=[x for x in budget_pairs if x['status']=='valid']; assert len(valid)==120
            rho=stats([x['spearman'] for x in valid])
            assert np.isclose(rho['mean'],summary['similarity_by_budget']['1000']['mean_spearman'])
            assert np.isclose(rho['p10'],summary['similarity_by_budget']['1000']['p10_spearman'])
            units[job]=dict(summary_sha256=sha(out/'summary.json'),plan_sha256=sha(out/'plan.json'),
                images=40,maps=240,superpixels=stats(segment_counts),budgets=budgets,
                budget_spearman=rho,worst_budget_pairs=sorted(valid,key=lambda x:x['spearman'])[:8],
                variance_reduction_fraction=1-np.mean(variances[2000])/np.mean(variances[1000]),
                versions=plan['versions'],prediction_context=plan['prediction_context'])
            print(job,'seed2000',budgets['2000']['seed_spearman'],'worst1000to2000',rho['minimum'],flush=True)
    return dict(status='audit_passed_budget_decision_pending',units=units,
                release_sha256=sha(artifact/'release-manifest.json'),images=240,maps=1440,
                recomputed_pair_comparisons=2160,budget_wall_seconds=status['wall_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=audit(args.artifact)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
