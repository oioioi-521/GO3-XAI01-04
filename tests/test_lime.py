import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.attribution import LIME
from experiments import run_unit


class TinyModel(torch.nn.Module):
    def forward(self, x):
        score = (x[:, 0, :, :x.shape[-1] // 2].mean((1, 2))
                 - x[:, 0, :, x.shape[-1] // 2:].mean((1, 2)))
        return torch.stack((score, -score), dim=1)


def image():
    rgb = torch.zeros(3, 24, 32)
    rgb[0, :, :16] = .9
    rgb[1, :, 16:] = .5
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (rgb - mean) / std


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('activation', ['softmax', 'sigmoid'])
def test_contract_determinism_and_local_rng(batched, activation):
    x = image()
    if batched:
        x = x[None]
    method = LIME(TinyModel().eval(), num_samples=64, batch_size=7,
                  output_activation=activation, max_dist=10)
    np.random.seed(91)
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    first = method.attribute(x, 1)
    method.attribute(x, 0)  # target/order must not consume another call's RNG
    second = method.attribute(x, 1)
    assert torch.equal(first, second)
    assert first.shape == (24, 32) and first.dtype == torch.float32
    assert first.device.type == 'cpu'
    assert torch.isfinite(first).all() and 0 <= first.min() <= first.max() <= 1
    assert torch.equal(torch_state, torch.get_rng_state())
    after = np.random.get_state()
    assert np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    assert method.last_diagnostics['n_superpixels'] > 1


def test_target_and_spatial_baseline(monkeypatch):
    import skimage.segmentation
    monkeypatch.setattr(skimage.segmentation, 'quickshift',
                        lambda rgb, **kw: np.broadcast_to(np.arange(32) // 16, (24, 32)))
    model = TinyModel().eval()
    method = LIME(model, num_samples=256)
    black = torch.tensor([-m/s for m,s in zip([.485,.456,.406],[.229,.224,.225])])[:,None,None]
    spatial_baseline = black.expand_as(image()).clone()
    positive = method.attribute(image(), 0, spatial_baseline)
    negative = method.attribute(image(), 1, spatial_baseline)
    assert positive[:, :16].mean() > positive[:, 16:].mean()
    assert negative[:, :16].mean() < negative[:, 16:].mean()
    with pytest.raises(ValueError, match='baseline'):
        method.attribute(image(), 0, torch.zeros(2, 3))


def test_constant_model_returns_zero_map():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros((len(x), 2))
    out = LIME(Constant().eval(), num_samples=16).attribute(image(), 0)
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize('kw', [dict(num_samples=1), dict(num_samples=True),
    dict(seed=-1), dict(segmentation_seed=2**32), dict(kernel_width=float('nan')),
    dict(ridge_alpha=0), dict(output_activation='logits'), dict(std=[1, 0, 1]),
    dict(segmentation='slic'), dict(batch_size=0)])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError):
        LIME(TinyModel().eval(), **kw)


@pytest.mark.parametrize('target', [-1, True, .2, 2])
def test_invalid_target(target):
    with pytest.raises(ValueError):
        LIME(TinyModel().eval(), num_samples=8).attribute(image(), target)


def test_nonfinite_inputs_and_logits():
    method = LIME(TinyModel().eval(), num_samples=8)
    bad = image(); bad[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        method.attribute(bad, 0)
    class Bad(torch.nn.Module):
        def forward(self, x):
            return x.new_full((len(x), 2), float('nan'))
    with pytest.raises(ValueError, match='non-finite logits'):
        LIME(Bad().eval(), num_samples=8).attribute(image(), 0)
    with pytest.raises(ValueError, match='model.eval'):
        LIME(TinyModel(), num_samples=8).attribute(image(), 0)


class SyntheticDataset:
    def __init__(self, dataset, split='debug', limit=None):
        assert (dataset, split) == ('synthetic', 'debug')
        self.samples = [dict(image=image(), image_id='synthetic-lime-0001',
                             target=1, class_name='synthetic-target', image_path='synthetic.png')]
    def __len__(self):
        return 1
    def __iter__(self):
        return iter(self.samples)


def smoke_config(output):
    cfg = yaml.safe_load(Path('configs/lime_smoke_imagenet.yaml').read_text())
    cfg['dataset'] = dict(name='synthetic', split='debug')
    cfg['model'] = dict(name='tiny', weights='none', num_classes=2,
                        checkpoint=None, output_activation='softmax')
    cfg['attribution']['num_samples'] = 64
    cfg['attribution']['max_dist'] = 10
    cfg['runtime']['device'] = 'cpu'
    cfg['output'] = {k: str(output / Path(v).relative_to('results/lime_smoke'))
                     if isinstance(v, str) else v for k, v in cfg['output'].items()}
    return cfg


def test_runner_smoke_resume_float_maps_and_snapshot(tmp_path, monkeypatch):
    cfg = smoke_config(tmp_path)
    config_path = tmp_path / 'smoke.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(run_unit, 'load_model', lambda **kw: TinyModel().eval())
    monkeypatch.setattr(run_unit, 'MetadataDataset', SyntheticDataset)
    run_unit.run(config_path)
    map_path = tmp_path / 'maps_float/lime_tiny_synthetic/synthetic-lime-0001.npy'
    before = map_path.read_bytes()
    run_unit.run(config_path)
    assert map_path.read_bytes() == before
    assert np.load(map_path).dtype == np.float32
    assert map_path.with_suffix('.npy.provenance.json').is_file()
    rows = run_unit.pd.read_csv(tmp_path / 'per_image.csv')
    assert len(rows) == 2 and set(rows.metric) == {'efficiency_time_ms', 'faithfulness_morf_auc_raw'}
    logs = [json.loads(s) for s in (tmp_path / 'run_log.jsonl').read_text().splitlines()]
    assert [(r['processed'], r['skipped']) for r in logs] == [(1, 0), (0, 1)]
    snapshot = next((tmp_path / 'configs').glob('*.yaml'))
    assert yaml.safe_load(snapshot.read_text()) == cfg
    run_unit.run(config_path, force=True)
    assert map_path.read_bytes() == before
    assert len(run_unit.pd.read_csv(tmp_path / 'per_image.csv')) == 2


def test_six_candidate_configs_and_formal_gate(tmp_path):
    for model in ('vgg16', 'resnet50', 'densenet121'):
        for dataset in ('imagenet', 'voc'):
            cfg = run_unit._load_config(Path(f'configs/lime_{model}_{dataset}.yaml'))
            assert cfg['dataset']['split'] == 'eval'
            assert cfg['runtime']['max_images'] is None
            assert cfg['runtime']['warmup_runs'] == 1
            assert cfg['runtime']['resume'] and cfg['output']['save_float_maps']
            assert cfg['metrics']['deletion_fractions'] == [i / 20 for i in range(21)]
            assert cfg['attribution']['output_activation'] == ('sigmoid' if dataset == 'voc' else 'softmax')
            with pytest.raises(ValueError, match='formal budget is pending'):
                run_unit.run(Path(f'configs/lime_{model}_{dataset}.yaml'))


def test_lime_stability_reuses_reference_and_preserves_morf(tmp_path, monkeypatch):
    from experiments import run_stability
    cfg = smoke_config(tmp_path)
    config_path = tmp_path / 'base.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    for module in (run_unit, run_stability):
        monkeypatch.setattr(module, 'load_model', lambda **kw: TinyModel().eval())
        monkeypatch.setattr(module, 'MetadataDataset', SyntheticDataset)
    run_unit.run(config_path)
    original = run_unit.pd.read_csv(tmp_path / 'per_image.csv')
    original_units = run_unit.pd.read_csv(tmp_path / 'units.csv')
    ref = tmp_path / 'maps_float/lime_tiny_synthetic/synthetic-lime-0001.npy'
    reference_bytes = ref.read_bytes()
    protocol = yaml.safe_load(Path('configs/stability_protocol_v1.yaml').read_text())
    protocol['runtime']['device'] = 'cpu'
    protocol['output'] = {k: str(tmp_path / ('stability_' + Path(v).name))
                           for k,v in protocol['output'].items()}
    protocol_path = tmp_path / 'protocol.yaml'
    protocol_path.write_text(yaml.safe_dump(protocol))
    def forbid_morf(*a, **kw):
        raise AssertionError('stability must never recompute MoRF')
    monkeypatch.setattr(run_unit, 'faithfulness_morf_auc_raw', forbid_morf)
    run_stability.run(config_path, protocol_path)
    run_stability.run(config_path, protocol_path)
    frame = run_unit.pd.read_csv(tmp_path / 'per_image.csv')
    selected = frame[frame.metric.isin(original.metric)].reset_index(drop=True)
    run_unit.pd.testing.assert_frame_equal(selected, original)
    units = run_unit.pd.read_csv(tmp_path / 'units.csv')
    selected_units = units[units.metric.isin(original_units.metric)].reset_index(drop=True)
    run_unit.pd.testing.assert_frame_equal(selected_units, original_units)
    assert ref.read_bytes() == reference_bytes
    traces = list(tmp_path.glob('stability_stability_trace_*debug*.csv'))
    assert len(traces) == 1
    trace = run_unit.pd.read_csv(traces[0])
    assert len(trace) == 5 and set(trace.target) == {1}


def test_gate_preflight_and_degenerate_diagnostics(tmp_path):
    from experiments.lime_budget_gate import plan, compare_maps
    cfg = run_unit._load_config(Path('configs/lime_resnet50_imagenet.yaml'))
    effective, dataset, report = plan(cfg, [1000, 2000], [42, 43, 44], tmp_path)
    assert len(dataset) == 40 and effective['dataset']['split'] == 'debug'
    assert report['attribution_calls'] == 246
    assert report['masked_forwards'] == 369000
    assert len(report['image_ids']) == len(set(report['image_ids'])) == 40
    assert compare_maps(np.ones((2,2)), np.ones((2,2)))['status'] == 'constant_map'
    comparison = compare_maps(np.arange(4), np.arange(4))
    assert comparison['spearman'] == 1 and comparison['rmse'] == 0
    with pytest.raises(ValueError, match='distinct budgets'):
        plan(cfg, [1000, 1000], [42,43,44], tmp_path)


def test_gate_writes_and_resumes_records_without_duplicate_samples(tmp_path, monkeypatch):
    from experiments import lime_budget_gate as gate
    cfg = smoke_config(tmp_path)
    config_path = tmp_path / 'fixture.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    class FixtureDataset:
        rows = run_unit.pd.DataFrame({'image_id': [f'fixture-{i}' for i in range(40)]})
        def __getitem__(self, index):
            return dict(image=image(), target=1, image_id=f'fixture-{index}')
    report = dict(config_hash='fixture-only', budgets=[8,16], seeds=[42,43,44],
                  missing_assets=[], output=str(tmp_path/'gate'))
    monkeypatch.setattr(gate, 'plan', lambda *a: (cfg, FixtureDataset(), copy.deepcopy(report)))
    monkeypatch.setattr(gate, 'load_model', lambda **kw: TinyModel().eval())
    calls = []
    class FixtureAttributor:
        last_diagnostics = {'seed': 42, 'num_samples': 8, 'n_superpixels': 2, 'weighted_r2': .5}
        def attribute(self, x, target, baseline):
            calls.append(target)
            return torch.arange(24*32, dtype=torch.float32).reshape(24,32)/(24*32)
    monkeypatch.setattr(gate, '_build_attributor', lambda *a: FixtureAttributor())
    monkeypatch.setattr('sys.argv', ['lime_budget_gate', '--config', str(config_path),
        '--output', str(tmp_path/'gate'), '--single-image-ms', '1'])
    gate.main()
    assert len(calls) == 246  # six discarded warmups, 40 * 6 timed maps
    gate.main()
    assert len(calls) == 252  # only warmups, no repeated timed samples
    assert len(list((tmp_path/'gate/records').glob('*.json'))) == 40
    timings = run_unit.pd.read_csv(tmp_path/'gate/timings.csv')
    assert len(timings) == 240
    summary = json.loads((tmp_path/'gate/summary.json').read_text())
    assert summary['n_images'] == 40 and 'pending' in summary['budget_status']
    saved_map = next((tmp_path/'gate/maps_float').glob('*.npy'))
    saved_map.write_bytes(b'changed')
    with pytest.raises(ValueError, match='map missing or changed'):
        gate.main()
