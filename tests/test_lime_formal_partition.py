"""Partition correctness is essential: a first-N bug would duplicate formal images."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from scripts.lime_formal_partition import partition_ids, select_rows, execution_partition, summarize, aggregate_hash
from scripts.lime_vectorized import VectorizedLIME


def test_exact_disjoint_coverage_and_weighted_summary():
    ids=[str(i) for i in range(460)]
    parts=partition_ids(ids,46)
    assert len(parts)==10 and [x for p in parts for x in p]==ids
    assert len(set(x for p in parts for x in p))==460
    frame=pd.DataFrame({'image_id':ids,'class_id':range(460)})
    assert select_rows(frame,parts[7]).class_id.tolist()==list(range(322,368))
    with pytest.raises(ValueError):select_rows(frame,['2','2'])
    with pytest.raises(ValueError):select_rows(frame,['missing'])
    with pytest.raises(ValueError):partition_ids(['a','a'],1)
    values=pd.DataFrame([dict(method='lime',model='tiny',dataset='synthetic',metric='x',value=v) for v in [1,2,30]])
    result=summarize(values,aggregate_hash(['a','b']))
    assert result.iloc[0]['mean']==11 and result.iloc[0]['std']==np.std([1,2,30],ddof=1)
    assert aggregate_hash(['a','b'])==aggregate_hash(['b','a'])


def test_real_runner_subset_and_stability_resume(tmp_path,monkeypatch):
    from preprocessing import dataset
    from experiments import run_unit,run_stability
    meta=pd.DataFrame([dict(image_id=f'fixture-{i}',class_id=1,class_name='one',image_path=f'{i}.png') for i in range(4)])
    monkeypatch.setattr(dataset,'get_split',lambda name,split:meta.copy())
    image=torch.linspace(-2,2,3*24*32).reshape(3,24,32)
    monkeypatch.setattr(dataset,'load_image',lambda path:image.clone())
    class Tiny(torch.nn.Module):
        def forward(self,x):
            score=x[:,0,:,:16].mean((1,2))-x[:,1,:,16:].mean((1,2))
            return torch.stack((score,-score),dim=1)
    for module in (run_unit,run_stability):monkeypatch.setattr(module,'load_model',lambda **kw:Tiny().eval())
    cfg=dict(method='lime',dataset=dict(name='synthetic',split='debug',image_ids=['fixture-3','fixture-1']),
        model=dict(name='tiny',weights='none',num_classes=2,task_type='multiclass',output_activation='softmax',checkpoint=None),
        attribution=dict(num_samples=32,batch_size=7,max_dist=10,seed=42),
        metrics=dict(names=['efficiency_time_ms','faithfulness_morf_auc_raw'],deletion_fractions=[0.,.5,1.]),
        runtime=dict(device='cpu',seed=42,max_images=None,warmup_runs=1,resume=True,budget_frozen=True),
        output={k:str(tmp_path/v) for k,v in dict(per_image_csv='per_image.csv',units_csv='units.csv',
            predictions_csv='predictions.csv',config_dir='configs',state_dir='run_state',run_log='base_log.jsonl',
            maps_dir='maps',float_maps_dir='maps_float').items()})
    cfg['output'].update(save_maps=True,save_float_maps=True)
    config=tmp_path/'unit.yaml';config.write_text(yaml.safe_dump(cfg))
    protocol=yaml.safe_load(Path('configs/stability_protocol_v1.yaml').read_text());protocol['runtime']['device']='cpu'
    protocol['output']={k:str(tmp_path/('stability_'+Path(v).name)) for k,v in protocol['output'].items()}
    pp=tmp_path/'protocol.yaml';pp.write_text(yaml.safe_dump(protocol))
    original=run_unit.SUPPORTED_METHODS['lime']
    with execution_partition(cfg,VectorizedLIME):
        assert dataset.MetadataDataset('synthetic').rows.image_id.tolist()==['fixture-3','fixture-1']
        run_unit.run(config);run_stability.run(config,pp)
        frame=pd.read_csv(tmp_path/'per_image.csv')
        assert len(frame)==8 and set(frame.image_id)=={'fixture-3','fixture-1'}
        before={p.name:p.read_bytes() for p in (tmp_path/'maps_float/lime_tiny_synthetic').glob('*.npy')}
        run_unit.run(config);run_stability.run(config,pp)
        for path in [tmp_path/'base_log.jsonl',Path(protocol['output']['run_log'])]:
            log=json.loads(path.read_text().splitlines()[-1]);assert (log['processed'],log['skipped'])==(0,2)
        after={p.name:p.read_bytes() for p in (tmp_path/'maps_float/lime_tiny_synthetic').glob('*.npy')}
        assert before==after
        trace=pd.read_csv(next(tmp_path.glob('stability_stability_trace_*.csv')))
        assert len(trace)==10 and not trace.duplicated(['image_id','repeat']).any()
        assert set(pd.read_csv(tmp_path/'predictions.csv').image_id)=={'fixture-3','fixture-1'}
    assert dataset.get_split('synthetic','debug').image_id.tolist()==meta.image_id.tolist()
    assert run_unit.SUPPORTED_METHODS['lime'] is original
