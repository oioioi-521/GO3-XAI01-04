"""Independently audit the downloaded LIME 8000 formal archive without a GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = '78d1f2c799e073d6e8ae87981d8ec71c40f44a30456b9c578cc563e25049a5f6'
RELEASE = 'c1d82ec2b6f5a179db32fbb3042d07853f01bea88b7fb8ab4f10777169eb0526'
METRICS = {'efficiency_time_ms', 'faithfulness_morf_auc_raw', 'stability_spearman', 'stability_valid_rate'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()[:12]


def csv(path):
    return pd.read_csv(path, dtype={'image_id': str, 'seed': str})


def same_table(a, b, keys):
    pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)


def check_hashes(base, mapping):
    for name, expected in mapping.items():
        assert sha(base/name) == expected, str(base/name)


def check_array(path):
    a = np.load(path, allow_pickle=False)
    assert a.dtype == np.float32 and a.shape == (224, 224), str(path)
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, str(path)
    return a


def main(bundle):
    run, overlay, release = (bundle/x for x in ('run', 'overlay', 'release'))
    assert sha(overlay/'overlay-manifest.json') == OVERLAY
    assert sha(release/'release-manifest.json') == RELEASE
    om, rm, manifest, plan = read(overlay/'overlay-manifest.json'), read(release/'release-manifest.json'), read(run/'manifest.json'), read(run/'plan.json')
    check_hashes(overlay, om['files'])
    for item in rm['files']:
        assert sha(release/item['path']) == item['sha256'], item['path']
    assert sha(release/'data-file-manifest.json') == rm['data_manifest_sha256']
    assert manifest['overlay_sha256'] == OVERLAY and manifest['release_sha256'] == RELEASE
    assert manifest['plan_sha256'] == sha(run/'plan.json')
    assert manifest['budget_review_sha256'] == sha(overlay/'budget_review_8000.json')
    assert manifest['num_samples'] == 8000 and manifest['batch_size'] == 32 and manifest['devices'] == list(range(8))
    assert manifest['per_process_memory_fraction'] == .1 and manifest['cpu_threads'] == 2
    meta_path = release/'data/metadata.csv'
    assert sha(meta_path) == sha(ROOT/'data/metadata.csv')
    metadata = csv(meta_path)
    assert len(metadata) == 1000
    targets = {(r.dataset, r.image_id): int(r.class_id) for r in metadata.itertuples()}
    weights = {(p['file'].split('-')[0], 'imagenet'): p['sha256'] for p in rm['imagenet_weights']}
    weights.update({(p['model'], 'voc'): p['sha256'] for p in read(release/'docs/VOC20_CHECKPOINT_MANIFEST.json')['checkpoints']})
    assets = read(bundle/'server_asset_recheck.json')
    assert assets['status']=='verified' and assets['release_sha256']==RELEASE
    assert (assets['source_count'],assets['data_count'],assets['weight_count'])==(125,1000,6)
    for kind, expected in [('source',rm['files']),('data',read(release/'data-file-manifest.json'))]:
        actual=[r for r in assets['assets'] if r['kind']==kind]
        assert len(actual)==len(expected)
        assert {r['path']:r['sha256'] for r in actual}=={r['path']:r['sha256'] for r in expected}
    actual_weights=assets['weights']
    assert len(actual_weights)==6 and {r['sha256'] for r in actual_weights}==set(weights.values())
    review = read(overlay/'budget_review_8000.json')
    assert review['status'] == 'accepted_for_formal' and review['num_samples'] == 8000
    old = ROOT/'results/lime-high-budget-evidence-20261014/high_budget'
    for unit, binding in review['units'].items():
        assert sha(old/unit/'summary.json') == binding['summary_sha256']
        assert sha(old/unit/'plan.json') == binding['plan_sha256']
    assert sha(ROOT/'analysis/lime_dcu_high_budget_audit_20261014.json') == review['independent_audit_sha256']
    complete, status = read(run/'COMPLETE.json'), read(run/'status.json')
    assert (run/'controller.exit').read_text().strip() == '0'
    assert complete['status'] == 'verified' and complete['resume_verified'] and complete['overlay_sha256'] == OVERLAY
    assert [complete[k] for k in ('units', 'shards', 'images', 'attributions')] == [6, 60, 2760, 16560]
    assert status['state'] == 'complete' and status['base_images'] == status['stability_images'] == 2760
    assert not status['failed'] and not status['pending'] and not status['running']
    assert not list(run.rglob('failure.json')) and not (run/'controller_failure.json').exists()
    for p in [run/'preflight.json', run/'postrun/preflight.json']:
        pre = read(p)
        assert pre['status'] == 'passed' and pre['release_sha256'] == RELEASE and pre['images'] == 1000 and len(pre['devices']) == 8
    check_hashes(run/'merged', complete['outputs'])
    for unit, digest in complete['unit_completions'].items():
        assert sha(run/'units'/unit/'completion.json') == digest
    phase_counts = {'regression': 24, 'gate': 12, 'formal': 60}
    phase_times = {}
    for phase, count in phase_counts.items():
        marker = read(run/f'{phase}_COMPLETE.json')
        assert marker['status'] == 'verified' and marker['jobs'] == count and len(plan[phase]) == count
        assert set(marker['completions']) == {j['key'] for j in plan[phase]}
        for key, digest in marker['completions'].items():
            assert sha(run/phase/key/'completion.json') == digest
        phase_times[phase] = marker['wall_seconds']

    regression_count = 0
    per_unit_regression = Counter()
    regression_pairs = {u: set() for u in review['units']}
    for job in plan['regression']:
        out = run/'regression'/job['key']; result = read(out/'completion.json'); rows = read(out/'records.json')
        assert result['status'] == 'verified_exact' and result['maps'] == 30
        assert result['records_sha256'] == sha(out/'records.json') and result['config_sha256'] == sha(out/'unit.yaml') == job['config_sha256']
        assert result['overlay_sha256'] == OVERLAY
        check_hashes(out, result['map_sha256'])
        refs = {r['image_id']: r for r in (read(p) for p in (old/job['unit']/'records').glob('*.json'))}
        assert len(rows) == 30 and {(r['image_id'], r['seed']) for r in rows} == {(i, seed) for i in job['image_ids'] for seed in (42,43,44)}
        for row in rows:
            ref = next(x for x in refs[row['image_id']]['runs'] if x['budget'] == 8000 and x['seed'] == row['seed'])
            assert row['exact'] and row['num_samples'] == 8000 and row['target'] == refs[row['image_id']]['target']
            assert row['weighted_r2'] == ref['weighted_r2'] and row['reference_sha256'] == ref['map_sha256']
            assert sha(old/job['unit']/ref['map_path']) == ref['map_sha256']
            assert sha(out/row['map_path']) == row['map_sha256']
            assert np.array_equal(check_array(out/row['map_path']), check_array(old/job['unit']/ref['map_path']))
            regression_count += 1; per_unit_regression[job['unit']] += 1
            pair = (row['image_id'],row['seed'])
            assert pair not in regression_pairs[job['unit']]
            regression_pairs[job['unit']].add(pair)
    assert regression_count == 720 and set(per_unit_regression.values()) == {120}
    for unit, pairs in regression_pairs.items():
        dataset = unit.rsplit('-',1)[1]
        ids = metadata[(metadata.dataset==dataset)&(metadata.split=='debug')].image_id
        assert pairs == {(i,seed) for i in ids for seed in (42,43,44)}

    shard_tables, shard_configs, maps = {}, {}, {}
    for phase in ('gate', 'formal'):
        split = 'debug' if phase == 'gate' else 'eval'
        for job in plan[phase]:
            out = run/phase/job['key']; result = read(out/'completion.json')
            cfg, protocol = yaml.safe_load((out/'unit.yaml').read_text()), yaml.safe_load((out/'protocol.yaml').read_text())
            ids = job['image_ids']; n = len(ids); model, dataset = cfg['model']['name'], cfg['dataset']['name']
            assert len(ids) == len(set(ids)) == (1 if phase == 'gate' else 46)
            assert cfg['dataset']['image_ids'] == ids and cfg['dataset']['split'] == split
            assert result['status'] == 'verified' and result['resume_verified'] and result['images'] == n
            assert result['release_sha256'] == RELEASE and result['overlay_sha256'] == OVERLAY
            assert sha(out/'unit.yaml') == result['config_sha256'] == job['config_sha256']
            assert sha(out/'protocol.yaml') == result['protocol_sha256'] == job['protocol_sha256']
            assert cfg['attribution']['num_samples'] == 8000 and cfg['attribution']['batch_size'] == 32
            frozen_cfg = yaml.safe_load((release/f'configs/lime_{model}_{dataset}.yaml').read_text())
            expected_attr = {**frozen_cfg['attribution'], 'num_samples':8000}
            assert cfg['attribution'] == expected_attr and cfg['model'] == frozen_cfg['model']
            assert cfg['runtime']['execution_overlay_sha256'] == OVERLAY and cfg['runtime']['budget_review_sha256'] == manifest['budget_review_sha256']
            assert cfg['runtime']['implementation'] == 'VectorizedLIME' and cfg['runtime']['max_images'] is None
            assert protocol['repeats'] == 5 and protocol['sigma'] == .005 and protocol['target'] == 'metadata'
            assert cfg['metrics']['deletion_fractions'] == [i/20 for i in range(21)]
            check_hashes(out, result['csv_sha256']); check_hashes(out, result['map_sha256'])
            frame, pred, summary = csv(out/'per_image.csv'), csv(out/'predictions.csv'), csv(out/'units.csv')
            trace = csv(next(out.glob('stability_trace_*.csv')))
            assert len(frame) == 4*n and set(frame.metric) == METRICS and not frame.duplicated(['image_id','metric']).any()
            assert set(frame.image_id) == set(ids) and np.isfinite(frame.value).all()
            assert len(pred) == n and set(pred.image_id) == set(ids) and not pred.image_id.duplicated().any()
            assert np.isfinite(pred.confidence).all() and pred.confidence.between(0,1).all()
            assert set(pred.checkpoint_sha256) == {weights[(model,dataset)]}
            assert len(set(pred.config_hash)) == 1 and set(pred.split) == {split}
            mc = cfg['model']; task = mc.get('task_type','multilabel' if dataset=='voc' else 'multiclass')
            activation = mc.get('output_activation','sigmoid' if task=='multilabel' else 'softmax')
            context = {'dataset':dataset,'split':split,'model':model,'weights':mc.get('weights','default').lower(),
                       'checkpoint_sha256':weights[(model,dataset)],'task_type':task,'output_activation':activation}
            identity = {'prediction_schema':1,'dataset':dataset,'split':split,'metadata_sha256':sha(meta_path),
                        'model':{'name':model,'weights':context['weights'],'num_classes':mc.get('num_classes',1000),
                                 'checkpoint_sha256':context['checkpoint_sha256'],'task_type':task,'output_activation':activation},
                        'preprocessing':{'resize':[224,224],'mean':[.485,.456,.406],'std':[.229,.224,.225]}}
            context['config_hash'] = canonical(identity)
            assert set(pred.config_hash) == {context['config_hash']}
            stability_hash = canonical({'base_config':cfg,'base_config_hash':canonical(cfg),'prediction_context':context,'stability':protocol,'max_images':None})
            assert set(trace.config_hash) == {stability_hash} and set(trace.protocol_version) == {protocol['protocol_version']}
            for row in pred.itertuples():
                assert row.target_class_id == targets[(dataset,row.image_id)]
                assert row.correct == int(row.target_class_id == row.predicted_class_id)
            assert len(trace) == 5*n and not trace.duplicated(['image_id','repeat']).any()
            assert set(zip(trace.image_id, trace['repeat'])) == {(i,r) for i in ids for r in range(5)}
            assert np.isfinite(trace[['score','attribution_time_ms','sigma','perturbation_mae_pixel','target_score_abs_delta']]).all().all()
            assert (trace.attribution_time_ms > 0).all() and trace.score.between(-1,1).all()
            for row in trace.itertuples():
                seed = int.from_bytes(hashlib.sha256(f'{dataset}\0{row.image_id}\0{row.repeat}'.encode()).digest()[:8],'big') % (2**63-1)
                assert int(row.seed) == seed and row.sigma == .005 and row.target == targets[(dataset,row.image_id)]
                if row.status == 'valid':
                    assert np.isfinite(row.spearman) and row.score == row.spearman
                else:
                    assert row.status in {'both_constant','reference_constant','candidate_constant','undefined_spearman'} and row.score == 0 and pd.isna(row.spearman)
            score = trace.groupby('image_id').score.mean().sort_index()
            valid = trace.assign(valid=(trace.status=='valid').astype(float)).groupby('image_id').valid.mean().sort_index()
            assert np.allclose(score,frame[frame.metric=='stability_spearman'].set_index('image_id').value.sort_index(),rtol=1e-10,atol=1e-12)
            assert np.allclose(valid,frame[frame.metric=='stability_valid_rate'].set_index('image_id').value.sort_index(),rtol=1e-10,atol=1e-12)
            recalc = frame.groupby(['method','model','dataset','metric']).value.agg(mean='mean',std='std',n='count').reset_index().fillna({'std':0})
            same_table(recalc, summary[recalc.columns], ['method','model','dataset','metric'])
            for log in ('base_log.jsonl','stability_log.jsonl'):
                last = json.loads((out/log).read_text().splitlines()[-1])
                assert last['processed'] == 0 and last['skipped'] == n
            paths = list(out.glob('maps_float/**/*.npy')); assert len(paths) == n
            digest = canonical(cfg)
            for p in paths:
                check_array(p); prov = read(Path(str(p)+'.provenance.json'))
                assert prov['image_id'] == p.stem and p.stem in ids and prov['split'] == split
                assert prov['base_config_hash'] == digest and prov['checkpoint_sha256'] == weights[(model,dataset)]
                assert prov['prediction_context_hash'] == pred.config_hash.iloc[0]
                assert prov['map_sha256'] == sha(p) and prov['binding'] == 'base_runner'
                assert (prov['method'],prov['model'],prov['dataset']) == ('lime',model,dataset)
                if phase == 'formal': maps[(job['unit'],p.stem)] = (p, digest, job['key'])
            pngs=list(out.glob('maps/**/*.png')); assert len(pngs)==n
            for p in pngs:
                with Image.open(p) as im:
                    assert im.size == (224,224)
                    im.verify()
            if phase == 'formal':
                shard_tables[job['key']] = {'per_image.csv':frame,'stability_trace.csv':trace,'predictions.csv':pred}
                shard_configs[job['key']] = digest

    index = read(run/'merged/maps_index.json')
    assert len(index) == len(maps) == 2760 and len({(r['unit'],r['image_id']) for r in index}) == 2760
    for item in index:
        p,digest,key=maps[(item['unit'],item['image_id'])]
        assert run/item['path'] == p and item['shard']==key and item['base_config_hash']==digest
        assert item['sha256']==sha(p) and item['provenance_sha256']==sha(Path(str(p)+'.provenance.json'))
    unit_tables, unit_reports = {}, {}
    for unit in complete['unit_completions']:
        parts=[j for j in plan['formal'] if j['unit']==unit]; keys=[j['key'] for j in parts]
        _,model,dataset=unit.split('-'); ids=[i for j in parts for i in j['image_ids']]
        expected=metadata[(metadata.dataset==dataset)&(metadata.split=='eval')].image_id.tolist()
        assert ids==expected and len(set(ids))==460 and len(parts)==10
        out=run/'units'/unit; result=read(out/'completion.json')
        assert result['status']=='verified' and result['resume_verified'] and result['images']==460 and result['shards']==keys
        check_hashes(out,result['outputs'])
        configs=[shard_configs[k] for k in keys]
        assert result['shard_config_hashes']==configs
        aggregate=hashlib.sha256(json.dumps(sorted(configs),separators=(',',':')).encode()).hexdigest()[:12]
        assert result['aggregate_config_hash']==aggregate
        tables={}
        for name,sort in [('per_image.csv',['image_id','metric']),('stability_trace.csv',['image_id','repeat']),('predictions.csv',['image_id'])]:
            tables[name]=csv(out/name); same_table(pd.concat([shard_tables[k][name] for k in keys],ignore_index=True),tables[name],sort)
        frame=tables['per_image.csv']; summary=csv(out/'units.csv'); tables['units.csv']=summary
        recalc=frame.groupby(['method','model','dataset','metric']).value.agg(mean='mean',std='std',n='count').reset_index()
        same_table(recalc,summary[recalc.columns],['method','model','dataset','metric'])
        assert set(summary.config_hash)=={aggregate} and (summary.n==460).all()
        trace=tables['stability_trace.csv']; pred=tables['predictions.csv']
        times=frame.loc[frame.metric=='efficiency_time_ms','value']
        unit_reports[unit]={'metrics':summary.drop(columns='config_hash').to_dict('records'),'base_p50_ms':float(times.median()),'base_p95_ms':float(times.quantile(.95)),
                           'noise_mean_ms':float(trace.attribution_time_ms.mean()),'base_attribution_seconds':float(times.sum()/1000),'noise_attribution_seconds':float(trace.attribution_time_ms.sum()/1000),
                           'prediction_correct':int(pred.correct.sum()),'trace_status_counts':dict(Counter(trace.status)),
                           'stability_p10':float(frame.loc[frame.metric=='stability_spearman','value'].quantile(.1))}
        unit_tables[unit]=tables
    row_counts={}
    for name,keys,n in [('per_image.csv',['model','dataset','image_id','metric'],11040),('stability_trace.csv',['model','dataset','image_id','repeat'],13800),('predictions.csv',['model','dataset','image_id'],2760),('units.csv',['model','dataset','metric'],24)]:
        table=csv(run/'merged'/name); assert len(table)==n and not table.duplicated(keys).any()
        same_table(pd.concat([t[name] for t in unit_tables.values()],ignore_index=True),table,keys); row_counts[name]=n
    report={'status':'verified','formal_images':2760,'formal_attributions':16560,'formal_shards':60,'logical_units':6,'regression_exact_maps':720,'resume_gate_images':12,'base_maps_verified':2760,
            'merged_rows':row_counts,'phase_wall_seconds':phase_times,'formal_with_merge_seconds':status['wall_seconds'],'formal_started_at':status['started_at'],'complete_time':complete['time'],
            'source_files_verified':len(rm['files']),'overlay_sha256':OVERLAY,'release_sha256':RELEASE,'units':unit_reports,
            'server_asset_recheck_sha256':sha(bundle/'server_asset_recheck.json'),
            'limits':['Original images and weight bytes verified on server; archive contains their manifests, not duplicate large assets.','Noise maps are not saved by the frozen protocol; audit verifies every trace, seeds, targets, per-image means, validity, and runner/resume evidence without rerunning inference.']}
    (bundle/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('bundle',type=Path)
    main(parser.parse_args().bundle.resolve())
