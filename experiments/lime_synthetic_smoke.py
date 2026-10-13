"""One synthetic image + deterministic tiny model; NOT a dataset/model gate.

Exercises the actual base runner, 21-point raw MoRF, NPY/provenance and resume.
No frozen dataset is modified, and no pretrained-weight result is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import run_unit


class TinyModel(torch.nn.Module):
    def forward(self, x):
        score = (x[:, 0, :, :x.shape[-1]//2].mean((1,2))
                 - x[:, 0, :, x.shape[-1]//2:].mean((1,2)))
        return torch.stack((score, -score), dim=1)


def synthetic_image():
    rgb = torch.zeros(3, 24, 32)
    rgb[0, :, :16] = .9
    rgb[1, :, 16:] = .5
    mean = torch.tensor([.485,.456,.406])[:,None,None]
    std = torch.tensor([.229,.224,.225])[:,None,None]
    return (rgb-mean)/std, rgb


class SyntheticDataset:
    def __init__(self, dataset, split='debug', limit=None):
        if (dataset, split) != ('synthetic', 'debug'):
            raise ValueError('synthetic smoke only')
        self.samples = [dict(image=synthetic_image()[0], image_id='synthetic-lime-0001',
            target=1, class_name='synthetic-ground-truth', image_path='synthetic.png')]
    def __len__(self):
        return len(self.samples)
    def __iter__(self):
        return iter(self.samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/lime_synthetic_smoke'))
    args = parser.parse_args()
    output = run_unit._resolve(str(args.output))
    output.mkdir(parents=True, exist_ok=True)
    config = run_unit._load_config(ROOT/'configs/lime_smoke_imagenet.yaml')
    config['dataset'] = dict(name='synthetic', split='debug')
    config['model'] = dict(name='tiny', weights='none', num_classes=2,
        checkpoint=None, output_activation='softmax')
    config['attribution']['num_samples'] = 64
    config['attribution']['max_dist'] = 10.0
    config['runtime']['device'] = 'cpu'
    config['output'] = {k: str(output/Path(v).relative_to('results/lime_smoke'))
        if isinstance(v,str) else v for k,v in config['output'].items()}
    config_path = output/'synthetic_config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    versions = {name: importlib.metadata.version(name) for name in
        ('torch','torchvision','lime','scikit-image','scikit-learn','numpy','scipy')}
    (output/'pre_run_plan.json').write_text(json.dumps(dict(
        scope='synthetic plumbing only; not frozen debug/eval', config=config,
        expected_duration='short CPU check; measured runtime saved after execution',
        output=str(output), versions=versions), indent=2))
    torch.set_num_threads(1)
    with patch.object(run_unit, 'load_model', lambda **kw: TinyModel().eval()), \
         patch.object(run_unit, 'MetadataDataset', SyntheticDataset):
        run_unit.run(config_path)
        run_unit.run(config_path)
    path = output/'maps_float/lime_tiny_synthetic/synthetic-lime-0001.npy'
    attribution = np.load(path, allow_pickle=False)
    source = synthetic_image()[1].permute(1,2,0).numpy()
    original = (source*255).astype(np.uint8)
    heat = np.stack((attribution, np.zeros_like(attribution), 1-attribution), axis=-1)
    overlay = ((source*.55+heat*.45)*255).astype(np.uint8)
    canvas = Image.new('RGB', (768,244), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8,4), 'SYNTHETIC ONLY / tiny model / target=1 / 64 samples', fill='black')
    for i,(name,arr) in enumerate((('RGB input', original), ('LIME map', (heat*255).astype(np.uint8)),('Overlay',overlay))):
        draw.text((i*256+8,24),name,fill='black')
        canvas.paste(Image.fromarray(arr).resize((256,192)),(i*256,48))
    canvas.save(output/'synthetic_example.png')
    result = dict(scope='synthetic plumbing only', map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        shape=list(attribution.shape), dtype=str(attribution.dtype), finite=bool(np.isfinite(attribution).all()),
        config_hash=run_unit._config_hash(config), versions=versions)
    (output/'smoke_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
