"""Independent audit of the saved 2000/4000/8000 debug comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_lime_budget_review import comparison, match, sha, stats

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [2000, 4000, 8000]
SEEDS = [42, 43, 44]
JOBS = [f'lime-{m}-{d}' for m in ('resnet50','densenet121','vgg16') for d in ('imagenet','voc')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()[:12]


def audit(artifact, previous):
    release = json.loads((artifact/'release-manifest.json').read_text())
    release_sha = sha(artifact/'release-manifest.json')
    controller_sha = sha(artifact/'run_lime_high_budget_queue.py')
    assert controller_sha == '152c4a3fe5d6af03e17619dd17f7715c9234a338eb7964b51920815df9ef3844'
    for item in release['files']:
        assert sha(ROOT/item['path']) == item['sha256'], item['path']
    assert sha(artifact/'data-file-manifest.json') == release['data_manifest_sha256']
    data = json.loads((artifact/'data-file-manifest.json').read_text())
    metadata = pd.read_csv(ROOT/'data/metadata.csv', dtype={'image_id':str})
    assert len(data) == 1000
    assert {str(Path('data')/p.replace('\\','/')) for p in metadata.image_path} == {x['path'] for x in data}
    voc = json.loads((ROOT/'docs/VOC20_CHECKPOINT_MANIFEST.json').read_text())['checkpoints']
    run = artifact/'high_budget'
    assert (run/'controller.exit').read_text().strip() == '0'
    assert (run/'postrun_integrity/check.exit').read_text().strip() == '0'
    post = json.loads((run/'postrun_integrity/preflight.json').read_text())
    assert post['status'] == 'passed' and post['release_sha256'] == release_sha and post['images'] == 1000
    assert not list(run.rglob('*failure.json')) and not list(run.rglob('failures.jsonl'))
    state = json.loads((run/'status.json').read_text())
    assert state['state'] == 'high_budget_ready' and set(state['completed']) == set(JOBS)
    manifest = json.loads((run/'controller_manifest.json').read_text())
    assert manifest['controller_sha256'] == controller_sha and manifest['release_sha256'] == release_sha
    assert manifest['budgets'] == BUDGETS and manifest['seeds'] == SEEDS and manifest['devices'] == [0,1,2,3]
    units = {}; pair_count = 0; anchor_count = 0
    for job in JOBS:
        _, model, dataset = job.split('-')
        out = run/job
        plan = json.loads((out/'plan.json').read_text())
        summary = json.loads((out/'summary.json').read_text())
        completion = json.loads((out/'completion.json').read_text())
        assert completion['status'] == 'verified_integrity_only' and completion['images'] == 40 and completion['maps'] == 360
        assert completion['summary_sha256'] == sha(out/'summary.json') and completion['plan_sha256'] == sha(out/'plan.json')
        assert completion['release_sha256'] == release_sha and completion['controller_sha256'] == controller_sha
        assert completion['formal_budget_frozen'] is False
        assert plan['budgets'] == BUDGETS and plan['seeds'] == SEEDS and not plan['missing_assets']
        identity = {k:plan[k] for k in ('config','budgets','seeds','metadata_sha256','prediction_context','image_ids','versions')}
        assert digest(identity) == plan['config_hash'] == summary['config_hash'] == manifest['plans'][job]['config_hash']
        assert sha(ROOT/'data/metadata.csv') == plan['metadata_sha256']
        assert summary['n_images'] == 40
        cfg = plan['config']; context = plan['prediction_context']
        assert cfg['dataset'] == dict(name=dataset, split='debug') and cfg['model']['name'] == model
        assert cfg['method'] == 'lime' and cfg['runtime']['max_images'] == 40 and cfg['runtime']['budget_frozen'] is False
        weights = release['imagenet_weights'] if dataset == 'imagenet' else voc
        assert context['checkpoint_sha256'] == next(x['sha256'] for x in weights if x.get('model') == model or x.get('file','').startswith(model))
        assert cfg['attribution']['output_activation'] == ('sigmoid' if dataset == 'voc' else 'softmax')
        frozen = metadata[(metadata.dataset == dataset)&(metadata.split == 'debug')]
        targets = dict(zip(frozen.image_id, frozen.class_id.astype(int)))
        assert len(targets) == 40 and list(plan['image_ids']) == list(frozen.image_id)
        records = [json.loads(p.read_text()) for p in sorted((out/'records').glob('*.json'))]
        assert len(records) == 40 and {r['image_id'] for r in records} == set(targets)
        assert len(list(out.rglob('*.npy'))) == 360
        timing = pd.read_csv(out/'timings.csv', dtype={'image_id':str})
        assert len(timing) == 360 and not timing.duplicated(['image_id','budget','seed']).any()
        timing = timing.set_index(['image_id','budget','seed'])
        saved_comp = pd.read_csv(out/'comparisons.csv', dtype={'image_id':str})
        assert len(saved_comp) == 240 and not saved_comp.duplicated(['image_id','budget','seed']).any()
        saved_comp = saved_comp.set_index(['image_id','budget','seed'])
        saved_var = pd.read_csv(out/'variance.csv', dtype={'image_id':str})
        assert len(saved_var) == 120 and not saved_var.duplicated(['image_id','budget']).any()
        saved_var = saved_var.set_index(['image_id','budget'])
        old = previous/'run/budget'/job
        old_plan = json.loads((old/'plan.json').read_text())
        assert cfg == old_plan['config'] and context == old_plan['prediction_context'] and plan['versions'] == old_plan['versions']
        old_records = {r['image_id']:r for p in (old/'records').glob('*.json') for r in [json.loads(p.read_text())]}
        pairs = {b:[] for b in BUDGETS}; var = {b:[] for b in BUDGETS}; runs = {b:[] for b in BUDGETS}
        cross = {b:[] for b in BUDGETS[:-1]}; anchors = []; per_image = []; seen_paths = set()
        for record in records:
            image_id = record['image_id']
            assert record['target'] == targets[image_id] and record['config_hash'] == plan['config_hash']
            assert len(record['runs']) == 9 and {(x['budget'],x['seed']) for x in record['runs']} == {(b,s) for b in BUDGETS for s in SEEDS}
            maps = {}; image_runs = {b:[] for b in BUDGETS}
            for item in record['runs']:
                b,s = item['budget'],item['seed']; path = (out/item['map_path']).resolve()
                assert path.is_relative_to(out.resolve()) and path not in seen_paths
                seen_paths.add(path); assert sha(path) == item['map_sha256']
                a = np.load(path, allow_pickle=False)
                assert a.shape == (224,224) and a.dtype == np.float32 and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
                assert item['num_samples'] == b and item['segmentation_seed'] == 42 and item['time_ms'] > 0
                assert all(np.isfinite(item[k]) for k in ('time_ms','weighted_r2','coefficient_min','coefficient_max'))
                row = timing.loc[(image_id,b,s)]
                for key in ('time_ms','weighted_r2','n_superpixels','num_samples','segmentation_seed'):
                    assert np.isclose(row[key], item[key]), (job,image_id,key)
                assert row.map_sha256 == item['map_sha256'] and row.map_path == item['map_path']
                item = {**item,'constant_map':bool(np.ptp(a) <= 1e-12)}
                maps[b,s] = a; runs[b].append(item); image_runs[b].append(item)
                if b == 2000:
                    old_item = next(x for x in old_records[image_id]['runs'] if x['budget'] == b and x['seed'] == s)
                    old_path = old/old_item['map_path']; assert sha(old_path) == old_item['map_sha256']
                    c = comparison(a, np.load(old_path, allow_pickle=False))
                    anchors.append(dict(image_id=image_id,seed=s,sha_identical=item['map_sha256']==old_item['map_sha256'],**c)); anchor_count += 1
            assert len({x['n_superpixels'] for x in record['runs']}) == 1
            assert len(record['budget_comparisons']) == 6
            assert {(x['budget'],x['seed']) for x in record['budget_comparisons']} == {(b,s) for b in BUDGETS[:-1] for s in SEEDS}
            for item in record['budget_comparisons']:
                b,s = item['budget'],item['seed']; assert item['reference_budget'] == 8000
                c = comparison(maps[b,s], maps[8000,s]); match(c,item); match(c,saved_comp.loc[(image_id,b,s)])
                cross[b].append(dict(image_id=image_id,seed=s,**c)); pair_count += 1
            assert len(record['seed_variance']) == 3 and {x['budget'] for x in record['seed_variance']} == set(BUDGETS)
            image_summary = dict(image_id=image_id,target=record['target'],budgets={})
            for item in record['seed_variance']:
                b = item['budget']; value = float(np.stack([maps[b,s] for s in SEEDS]).var(0,ddof=1).mean())
                assert np.isclose(value,item['pixel_variance_mean'],rtol=1e-6,atol=1e-9)
                assert np.isclose(value,saved_var.loc[(image_id,b)].pixel_variance_mean,rtol=1e-6,atol=1e-9)
                var[b].append(value)
                assert len(item['seed_pairs']) == 3 and {(x['seed_a'],x['seed_b']) for x in item['seed_pairs']} == {(42,43),(42,44),(43,44)}
                values = []
                for pair in item['seed_pairs']:
                    c = comparison(maps[b,pair['seed_a']], maps[b,pair['seed_b']]); match(c,pair)
                    pairs[b].append(dict(image_id=image_id,seed_a=pair['seed_a'],seed_b=pair['seed_b'],**c)); pair_count += 1
                    if c['status'] == 'valid': values.append(c['spearman'])
                image_summary['budgets'][str(b)] = dict(seed_spearman_mean=float(np.mean(values)) if values else None,
                    weighted_r2_mean=float(np.mean([x['weighted_r2'] for x in image_runs[b]])),pixel_variance=value)
            per_image.append(image_summary)
        budget_stats = {}
        for b in BUDGETS:
            valid = [x for x in pairs[b] if x['status'] == 'valid']
            budget_stats[str(b)] = dict(time_ms=stats([x['time_ms'] for x in runs[b]]),weighted_r2=stats([x['weighted_r2'] for x in runs[b]]),
                seed_spearman=stats([x['spearman'] for x in valid]) if valid else None,valid_seed_pairs=len(valid),total_seed_pairs=len(pairs[b]),
                negative_seed_pairs=sum(x['spearman']<0 for x in valid),seed_pairs_below_half=sum(x['spearman']<.5 for x in valid),
                pixel_variance=stats(var[b]),constant_maps=sum(x['constant_map'] for x in runs[b]),worst_seed_pairs=sorted(valid,key=lambda x:x['spearman'])[:10])
            assert np.isclose(budget_stats[str(b)]['time_ms']['mean'],summary['mean_time_ms'][str(b)])
            assert np.isclose(np.mean(var[b]),summary['mean_pixel_variance'][str(b)])
        cross_stats = {}
        for b,values in cross.items():
            valid = [x for x in values if x['status'] == 'valid']; ss = summary['similarity_by_budget'][str(b)]
            rho = stats([x['spearman'] for x in valid]) if valid else None
            assert ss['reference_budget'] == 8000 and ss['total_pairs'] == len(values) and ss['valid_pairs'] == len(valid)
            if valid:
                assert np.isclose(rho['mean'],ss['mean_spearman']) and np.isclose(rho['p10'],ss['p10_spearman'])
            assert np.isclose(np.mean([x['rmse'] for x in values]),ss['mean_rmse'])
            cross_stats[str(b)] = dict(reference_budget=8000,spearman=rho,worst_pairs=sorted(valid,key=lambda x:x['spearman'])[:10])
        units[job] = dict(images=40,maps=360,summary_sha256=sha(out/'summary.json'),plan_sha256=sha(out/'plan.json'),budgets=budget_stats,
            cross_budget=cross_stats,anchor_2000=dict(maps=len(anchors),sha_identical=sum(x['sha_identical'] for x in anchors),
                spearman=stats([x['spearman'] for x in anchors if x['status']=='valid']),max_rmse=max(x['rmse'] for x in anchors)),
            variance_reduction_2000_to_8000=1-np.mean(var[8000])/np.mean(var[2000]),
            variance_reduction_4000_to_8000=1-np.mean(var[8000])/np.mean(var[4000]),
            measured_wall_seconds_excluding_warmup=summary['current_invocation_wall_seconds'],per_image=per_image)
        print(job, 'seed means', [round(budget_stats[str(b)]['seed_spearman']['mean'],5) for b in BUDGETS],
              '8000 min',round(budget_stats['8000']['seed_spearman']['minimum'],5),'anchor identical',units[job]['anchor_2000']['sha_identical'],flush=True)
    return dict(status='integrity_passed_budget_review_pending',units=units,images=240,maps=2160,recomputed_pair_comparisons=pair_count,
        recomputed_anchor_comparisons=anchor_count,release_sha256=release_sha,controller_sha256=controller_sha,
        started_at=state['started_at'],ended_at=state['updated_at'],wall_seconds=state['wall_seconds'],postrun_preflight=post)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=audit(args.artifact,args.previous)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
