"""Paired reference checks for an experimental acceleration, not a new method."""
import numpy as np
import pytest
import torch

from experiments.attribution.lime import LIME
from scripts.lime_vectorized import VectorizedLIME


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        score = x[:, 0, :, :7].mean((1, 2)) - x[:, 1, :, 7:].mean((1, 2))
        return torch.stack((score, -score, score / 3), dim=1)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
@pytest.mark.parametrize("seed", [42, 43])
@pytest.mark.parametrize("baseline_kind", ["default", "scalar", "spatial"])
def test_reference_inputs_maps_r2_and_rng(activation, seed, baseline_kind):
    # Non-square images, a partial final batch, out-of-range RGB and a spatial
    # baseline exercise places where an apparently faster path changes LIME.
    image = torch.linspace(-4, 4, 3 * 16 * 20).reshape(1, 3, 16, 20)
    baseline = {"default": None, "scalar": torch.ones(1, 3, 1, 1) * -.3,
                "spatial": image.flip(-1) * .3}[baseline_kind]
    old_model, fast_model = RecordingModel().eval(), RecordingModel().eval()
    kwargs = dict(num_samples=67, batch_size=13, seed=seed, max_dist=10,
                  output_activation=activation)
    old, fast = LIME(old_model, **kwargs), VectorizedLIME(fast_model, **kwargs)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
    a, b = old.attribute(image, 2, baseline), fast.attribute(image, 2, baseline)
    assert torch.equal(torch.cat(old_model.inputs), torch.cat(fast_model.inputs))
    assert torch.equal(a, b)
    assert old.last_diagnostics['weighted_r2'] == fast.last_diagnostics['weighted_r2']
    after = np.random.get_state()
    assert np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    assert torch.equal(b, fast.attribute(image, 2, baseline))


def test_sampling_matrix_is_upstream_exact(monkeypatch):
    from lime.lime_image import LimeImageExplainer
    captured = {}
    original = LimeImageExplainer.data_labels
    def record(self, *args, **kwargs):
        data, labels = original(self, *args, **kwargs)
        captured.update(masks=data.copy(), predictions=labels.copy())
        return data, labels
    monkeypatch.setattr(LimeImageExplainer, "data_labels", record)
    model = RecordingModel().eval()
    image = torch.linspace(-2, 2, 3 * 16 * 20).reshape(1, 3, 16, 20)
    kw = dict(num_samples=67, batch_size=13, seed=44, max_dist=10)
    old, fast = LIME(model, **kw), VectorizedLIME(model, capture_neighborhood=True, **kw)
    old.attribute(image, 1)
    fast.attribute(image, 1)
    assert np.array_equal(captured['masks'], fast.last_neighborhood['masks'])
    assert np.array_equal(captured['predictions'], fast.last_neighborhood['predictions'])


def test_constant_and_invalid_logits():
    class Constant(torch.nn.Module):
        def __init__(self, value):
            super().__init__(); self.value = value
        def forward(self, x):
            return x.new_full((len(x), 2), self.value)
    x = torch.zeros(3, 16, 20)
    assert not VectorizedLIME(Constant(0.).eval(), num_samples=8).attribute(x, 1).any()
    with pytest.raises(ValueError, match="non-finite logits"):
        VectorizedLIME(Constant(float('nan')).eval(), num_samples=8).attribute(x, 1)
    with pytest.raises(ValueError, match="target incompatible"):
        VectorizedLIME(Constant(0.).eval(), num_samples=8).attribute(x, 2)
