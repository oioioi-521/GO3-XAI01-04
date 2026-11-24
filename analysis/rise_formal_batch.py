"""Run frozen RISE units sequentially, resume immediately, and validate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.predictions import build_prediction_context  # noqa: E402
from experiments.run_unit import _config_hash, _effective_config, _load_config  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(config_path: Path) -> dict:
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    config = _effective_config(_load_config(config_path), None)
    digest = _config_hash(config)
    dataset, model = config['dataset']['name'], config['model']['name']
    require(config['dataset']['split'] == 'eval', 'formal split must be eval')
    meta = pd.read_csv(ROOT / 'data/metadata.csv', dtype={'image_id': str})
    expected = meta[(meta.dataset == dataset) & (meta.split == 'eval')]
    ids = set(expected.image_id)
    require(len(ids) == len(expected) == 460, 'frozen eval count mismatch')
    output = config['output']
    per = pd.read_csv(ROOT / output['per_image_csv'], dtype={'image_id': str})
    rows = per[(per.dataset == dataset) & (per.model == model) & (per.method == 'rise')]
    metrics = {'efficiency_time_ms', 'faithfulness_morf_auc_raw'}
    require(len(rows) == 920 and set(rows.image_id) == ids, 'metric IDs/count mismatch')
    require(set(rows.metric) == metrics, 'metric set mismatch')
    require(not rows.duplicated(['image_id', 'metric']).any(), 'duplicate metrics')
    require(np.isfinite(rows.value).all(), 'nonfinite metrics')
    require(rows.loc[rows.metric == 'faithfulness_morf_auc_raw', 'value'].between(0, 1).all(), 'AUC range')
    require((rows.loc[rows.metric == 'efficiency_time_ms', 'value'] > 0).all(), 'nonpositive time')
    summaries = pd.read_csv(ROOT / output['units_csv'])
    unit = summaries[(summaries.dataset == dataset) & (summaries.model == model) & (summaries.method == 'rise')]
    require(len(unit) == 2 and set(unit.metric) == metrics and set(unit.config_hash) == {digest}, 'unit identity')
    stats = rows.groupby('metric').value.agg(['mean', 'std', 'count'])
    for row in unit.itertuples(index=False):
        values = stats.loc[row.metric]
        require(int(row.n) == int(values['count']) == 460, 'summary count')
        require(np.allclose([row.mean, row.std], [values['mean'], values['std']], rtol=1e-10, atol=1e-12), 'summary recomputation')
    checkpoint = config['model'].get('checkpoint')
    context = build_prediction_context(dataset=dataset, split='eval', model_config=config['model'], checkpoint=ROOT / checkpoint if checkpoint else None)
    predictions = pd.read_csv(ROOT / output.get('predictions_csv', 'results/predictions.csv'), dtype={'image_id': str})
    predictions = predictions[(predictions.dataset == dataset) & (predictions.model == model) & (predictions.split == 'eval')]
    require(len(predictions) == 460 and set(predictions.image_id) == ids and not predictions.image_id.duplicated().any(), 'prediction IDs/count')
    require(set(predictions.config_hash) == {context['config_hash']}, 'prediction context')
    require(set(predictions.checkpoint_sha256) == {context['checkpoint_sha256']}, 'prediction checkpoint')
    truth = expected.set_index('image_id').class_id.astype(int)
    require(all(int(row.target_class_id) == int(truth.loc[row.image_id]) for row in predictions.itertuples()), 'prediction targets')
    require(np.isfinite(predictions.confidence).all() and predictions.confidence.between(0, 1).all(), 'prediction confidence')
    require(((predictions.target_class_id == predictions.predicted_class_id).astype(int) == predictions.correct).all(), 'correctness flag')
    artifacts = {}
    for key, directory, suffix in [('npy', 'float_maps_dir', '.npy'), ('png', 'maps_dir', '.png')]:
        folder = ROOT / output[directory] / f'rise_{model}_{dataset}'
        paths = sorted((p for p in folder.glob('*' + suffix) if p.stem in ids), key=lambda p: p.name)
        require(len(paths) == 460, 'map count')
        entries = []
        for path in paths:
            data = path.read_bytes()
            if suffix == '.npy':
                array = np.load(path, allow_pickle=False)
                require(array.dtype == np.float32 and array.shape == (224, 224), 'map dtype/shape')
                require(np.isfinite(array).all() and 0 <= float(array.min()) <= float(array.max()) <= 1, 'map values')
                require(float(array.max()) > float(array.min()), 'constant RISE map')
            else:
                with Image.open(path) as im:
                    require(im.format == 'PNG' and im.size == (224, 224), 'PNG format/shape')
                    im.load()
            entries.append({'path': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        artifacts[key] = {'count': len(paths), 'bytes': sum(e['bytes'] for e in entries), 'aggregate_sha256': hashlib.sha256(canonical).hexdigest()}
    grouped = rows[rows.metric == 'faithfulness_morf_auc_raw'].merge(predictions[['image_id', 'correct']], on='image_id', validate='one_to_one')
    return {'status': 'passed', 'dataset': dataset, 'model': model, 'config_hash': digest, 'metric_rows': len(rows), 'predictions': len(predictions), 'metrics': stats.to_dict(orient='index'), 'faithfulness_groups': {str(k): {'mean': float(g.value.mean()), 'std': float(g.value.std()), 'n': len(g)} for k, g in grouped.groupby('correct')}, 'artifacts': artifacts}


def execute(config: Path, log_path: Path) -> None:
    # Exclusive creation preserves evidence from previous attempts.
    with log_path.open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'experiments/run_unit.py'), '--config', str(config)], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.strip() and ('RISE ' not in line or '%|' not in line):
                print(line.rstrip(), flush=True)
        code = process.wait()
        require(code == 0, f'runner exit={code}; log={log_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['voc', 'imagenet'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output.is_relative_to(ROOT / 'results'), 'batch logs must stay under project results/')
    output.mkdir(parents=True, exist_ok=True)
    for model in ['resnet50', 'densenet121', 'vgg16']:
        config = ROOT / f'configs/rise_{model}_{args.dataset}.yaml'
        print(f'START {model}/{args.dataset}', flush=True)
        execute(config, output / f'{model}_formal.log')
        execute(config, output / f'{model}_resume.log')
        receipt = json.loads((ROOT / 'results/run_log.jsonl').read_text().splitlines()[-1])
        require(receipt['model'] == model and receipt['dataset'] == args.dataset and receipt['split'] == 'eval' and receipt['processed'] == 0 and receipt['skipped'] == 460, 'resume receipt mismatch')
        report = validate(config)
        report['resume'] = receipt
        with (output / f'{model}_validation.json').open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f'VALIDATED {model}/{args.dataset}: 460 eval, zero-recompute resume', flush=True)
    print('ALL FORMAL UNITS PASSED', flush=True)


if __name__ == '__main__':
    main()
