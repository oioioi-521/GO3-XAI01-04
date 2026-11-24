import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("captum")

from experiments.attribution import GradCAM, IntegratedGradients
from experiments.attribution.grad_cam import resolve_target_layer


class SpatialClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs[:, 0].sum(dim=(1, 2)) - inputs[:, 1].sum(dim=(1, 2))
        return torch.stack((score, -score), dim=1)


class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, kernel_size=1, bias=False),
            torch.nn.ReLU(),
        )
        with torch.no_grad():
            self.features[0].weight.zero_()
            self.features[0].weight[0, 0, 0, 0] = 1.0

    def forward(self, inputs):
        activation = self.features(inputs)
        score = activation.sum(dim=(1, 2, 3))
        return torch.stack((score, -score), dim=1)


def _signed_image():
    image = torch.zeros(1, 3, 6, 6)
    image[:, 0, :, :3] = 1.0
    image[:, 1, :, 3:] = 1.0
    return image


def test_integrated_gradients_shape_signed_order_and_determinism():
    method = IntegratedGradients(
        SpatialClassifier().eval(),
        n_steps=8,
        internal_batch_size=4,
        approximation_method="riemann_middle",
    )
    image = _signed_image()
    baseline = torch.zeros(1, 3, 1, 1)

    first = method.attribute(image, target=0, baseline=baseline)
    second = method.attribute(image, target=0, baseline=baseline)

    assert first.shape == (6, 6)
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert 0.0 <= first.min() <= first.max() <= 1.0
    assert torch.equal(first[:, :3], torch.ones(6, 3))
    assert torch.equal(first[:, 3:], torch.zeros(6, 3))


def test_gradcam_resolves_layer_upsamples_and_uses_positive_evidence():
    model = TinyCNN().eval()
    assert resolve_target_layer(model, "features.0") is model.features[0]
    method = GradCAM(model, target_layer="features.0")

    heatmap = method.attribute(
        _signed_image(), target=0, baseline=torch.zeros(1, 3, 1, 1)
    )

    assert heatmap.shape == (6, 6)
    assert heatmap.device.type == "cpu"
    assert torch.isfinite(heatmap).all()
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0
    assert torch.equal(heatmap[:, :3], torch.ones(6, 3))
    assert torch.equal(heatmap[:, 3:], torch.zeros(6, 3))


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (IntegratedGradients, {"n_steps": 1}),
        (IntegratedGradients, {"internal_batch_size": 0}),
        (IntegratedGradients, {"approximation_method": "unknown"}),
        (GradCAM, {"target_layer": "features.99"}),
    ],
)
def test_gradient_method_invalid_configuration_is_rejected(constructor, kwargs):
    model = TinyCNN().eval() if constructor is GradCAM else SpatialClassifier().eval()
    with pytest.raises(ValueError):
        constructor(model, **kwargs)


@pytest.mark.parametrize(
    "method",
    [
        IntegratedGradients(SpatialClassifier().eval(), n_steps=4),
        GradCAM(TinyCNN().eval(), target_layer="features.0"),
    ],
)
def test_gradient_methods_reject_invalid_target_and_image(method):
    with pytest.raises(ValueError, match="target"):
        method.attribute(_signed_image(), target=2, baseline=torch.zeros(1, 3, 1, 1))
    with pytest.raises(ValueError, match="three-channel"):
        method.attribute(torch.ones(1, 1, 4, 4), target=0)
