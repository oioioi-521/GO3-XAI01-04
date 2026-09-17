"""Ensure importing the shared attribution package does not require Captum for RISE."""

import subprocess
import sys
from pathlib import Path


def test_rise_runs_when_captum_imports_are_blocked():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockCaptum(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'captum' or fullname.startswith('captum.'):
            raise ImportError('Captum deliberately unavailable in this regression test')
        return None

sys.meta_path.insert(0, BlockCaptum())
import torch
from experiments.attribution import RISE

class TinyClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs.mean(dim=(1, 2, 3))
        return torch.stack((-score, score), dim=1)

result = RISE(TinyClassifier(), num_masks=4, mask_size=2, batch_size=2).attribute(
    torch.ones(1, 3, 4, 4), target=1
)
assert result.shape == (4, 4)
print('rise-without-captum=ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "rise-without-captum=ok" in completed.stdout
