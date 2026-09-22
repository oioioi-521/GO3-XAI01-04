import torch

from experiments.attribution import RISE


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs.mean(dim=(1, 2, 3))
        return torch.stack((-score, score), dim=1)


def test_rise_shape_range_and_determinism():
    model = TinyClassifier().eval()
    image = torch.linspace(0, 1, 3 * 12 * 10).reshape(1, 3, 12, 10)
    rise = RISE(model, num_masks=32, mask_size=4, batch_size=8, seed=7)

    first = rise.attribute(image, target=1)
    second = rise.attribute(image, target=1)

    assert first.shape == (12, 10)
    assert torch.equal(first, second)
    assert 0.0 <= first.min() <= first.max() <= 1.0


def test_rise_rejects_invalid_target():
    rise = RISE(TinyClassifier(), num_masks=4, mask_size=2, batch_size=2)
    image = torch.ones(1, 3, 4, 4)

    try:
        rise.attribute(image, target=2)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("invalid target should fail")
