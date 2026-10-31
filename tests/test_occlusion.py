import math

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("captum")

from captum.attr import Occlusion as CaptumOcclusion

from experiments.attribution.occlusion import Occlusion, normalized_black_baseline
from experiments.metrics import faithfulness_morf_auc


class SpatialClassifier(torch.nn.Module):
    """A deterministic two-logit image classifier with spatially varying input."""

    def forward(self, inputs):
        score = inputs[:, 0].sum(dim=(1, 2)) - inputs[:, 1].sum(dim=(1, 2))
        return torch.stack((score, -score), dim=1)


class RecordingClassifier(SpatialClassifier):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, inputs):
        self.seen.append(inputs.detach().clone())
        return super().forward(inputs)


class InvalidOutputClassifier(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, :1]


class ConstantClassifier(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros((inputs.shape[0], 2), device=inputs.device)


def _signed_image():
    image = torch.zeros(1, 3, 4, 4)
    image[:, 0, :, :2] = 1.0
    image[:, 1, :, 2:] = 1.0
    return image


def test_occlusion_shape_range_reproducibility_and_morf_compatibility():
    model = SpatialClassifier().eval()
    image = _signed_image()
    baseline = torch.zeros(1, 3, 1, 1)
    occlusion = Occlusion(
        model,
        window_height=2,
        window_width=2,
        stride_height=2,
        stride_width=2,
        perturbations_per_eval=2,
    )

    first = occlusion.attribute(image, target=0, baseline=baseline)
    second = occlusion.attribute(image, target=0, baseline=baseline)

    assert first.shape == (4, 4)
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert 0.0 <= first.min() <= first.max() <= 1.0

    score = faithfulness_morf_auc(
        model=model,
        image=image,
        target=0,
        attribution=first,
        baseline=baseline,
        fractions=[0.0, 0.5, 1.0],
    )
    assert score >= 0.0


def test_constant_attribution_is_a_finite_zero_heatmap_for_morf():
    model = ConstantClassifier().eval()
    image = _signed_image()
    occlusion = Occlusion(model, 2, 2, 2, 2, perturbations_per_eval=2)

    heatmap = occlusion.attribute(image, target=0, baseline=torch.zeros(1, 3, 1, 1))

    assert heatmap.shape == (4, 4)
    assert torch.equal(heatmap, torch.zeros_like(heatmap))
    score = faithfulness_morf_auc(
        model=model,
        image=image,
        target=0,
        attribution=heatmap,
        baseline=torch.zeros(1, 3, 1, 1),
        fractions=[0.0, 0.5, 1.0],
    )
    assert math.isfinite(score)


def test_occlusion_preserves_signed_channel_mean_before_normalization():
    model = SpatialClassifier().eval()
    image = _signed_image()
    baseline = torch.zeros(1, 3, 1, 1)
    occlusion = Occlusion(model, 2, 2, 2, 2, perturbations_per_eval=1)

    actual = occlusion.attribute(image, target=0, baseline=baseline)
    raw = CaptumOcclusion(model).attribute(
        image,
        sliding_window_shapes=(3, 2, 2),
        strides=(3, 2, 2),
        baselines=baseline,
        target=0,
        perturbations_per_eval=1,
    )
    signed = raw.mean(dim=1).squeeze(0)
    assert signed.min() < 0 < signed.max()
    expected = (signed - signed.min()) / (signed.max() - signed.min())

    assert torch.allclose(actual, expected.cpu())
    assert not torch.allclose(actual, raw.abs().mean(dim=1).squeeze(0).cpu())
    # A normalized zero is the raw map's minimum (negative here), not a raw
    # zero contribution. MoRF ranks pixels descending, and this monotonic
    # transform leaves that ranking intact.
    minimum_index = actual.argmin()
    assert signed.flatten()[minimum_index] < 0
    ordered = signed.flatten().argsort(descending=True)
    assert torch.all(actual.flatten()[ordered[:-1]] >= actual.flatten()[ordered[1:]])


def test_default_baseline_is_normalized_rgb_black_and_reaches_captum():
    model = RecordingClassifier().eval()
    image = torch.ones(1, 3, 4, 4)
    expected = normalized_black_baseline(image)
    occlusion = Occlusion(model, 4, 4, 4, 4)

    occlusion.attribute(image, target=0)

    assert torch.allclose(expected.flatten(), torch.tensor([
        -0.485 / 0.229,
        -0.456 / 0.224,
        -0.406 / 0.225,
    ]))
    assert any(torch.equal(batch, expected.expand_as(batch)) for batch in model.seen)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_height": 0},
        {"window_width": 0},
        {"stride_height": 0},
        {"stride_width": 0},
        {"perturbations_per_eval": 0},
    ],
)
def test_invalid_constructor_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Occlusion(SpatialClassifier(), **kwargs)


def test_invalid_target_geometry_and_model_output_are_rejected():
    image = _signed_image()
    occlusion = Occlusion(SpatialClassifier(), 2, 2, 2, 2)
    with pytest.raises(ValueError, match="incompatible"):
        occlusion.attribute(image, target=2)
    with pytest.raises(ValueError, match="window"):
        Occlusion(SpatialClassifier(), 5, 2, 2, 2).attribute(image, target=0)
    with pytest.raises(ValueError, match="stride"):
        Occlusion(SpatialClassifier(), 2, 2, 3, 2).attribute(image, target=0)
    with pytest.raises(ValueError, match="logits"):
        Occlusion(InvalidOutputClassifier(), 2, 2, 2, 2).attribute(image, target=0)
