"""Unit tests for the shared torchvision model factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn

from models import factory


class DummyVGG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(512, 1000))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DummyResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2048, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DummyDenseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1024, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def test_load_model_supported_weights_passes_correct_weights_to_builder(monkeypatch):
    calls: list[Dict[str, Any]] = []

    def fake_builder(weights=None):
        calls.append({"weights": weights})
        return DummyResNet()

    monkeypatch.setitem(factory._BUILDERS, "resnet50", (fake_builder, "default_weights_sentinel"))

    # Test "default" (case-insensitive)
    model = factory.load_model("resnet50", device=torch.device("cpu"), weights="default")
    assert calls[-1]["weights"] == "default_weights_sentinel"
    assert model.training is False

    model_upper = factory.load_model("resnet50", device=torch.device("cpu"), weights="DEFAULT")
    assert calls[-1]["weights"] == "default_weights_sentinel"
    assert model_upper.training is False

    # Test "none" (case-insensitive) and None
    model_none = factory.load_model("resnet50", device=torch.device("cpu"), weights="none")
    assert calls[-1]["weights"] is None
    assert model_none.training is False

    model_none_upper = factory.load_model("resnet50", device=torch.device("cpu"), weights="NONE")
    assert calls[-1]["weights"] is None
    assert model_none_upper.training is False

    model_python_none = factory.load_model("resnet50", device=torch.device("cpu"), weights=None)
    assert calls[-1]["weights"] is None
    assert model_python_none.training is False


def test_load_model_rejects_unsupported_weights_without_calling_builder(monkeypatch):
    called = False

    def fake_builder(weights=None):
        nonlocal called
        called = True
        return DummyResNet()

    monkeypatch.setitem(factory._BUILDERS, "resnet50", (fake_builder, "sentinel"))

    # Explicit strings that are not "default" or "none" must raise ValueError
    invalid_cases = ["IMAGENET1K_V2", "random", "VGG16_Weights.DEFAULT", "", "invalid"]
    for invalid in invalid_cases:
        called = False
        with pytest.raises(ValueError, match="unsupported weights"):
            factory.load_model("resnet50", device=torch.device("cpu"), weights=invalid)
        assert called is False, f"builder was unexpectedly invoked for weights={invalid!r}"


def test_load_model_rejects_unsupported_model():
    with pytest.raises(ValueError, match="unsupported model 'alexnet'"):
        factory.load_model("alexnet", device=torch.device("cpu"))


@pytest.mark.parametrize(
    ("name", "dummy_cls", "attr_name"),
    [
        ("vgg16", DummyVGG, "classifier"),
        ("resnet50", DummyResNet, "fc"),
        ("densenet121", DummyDenseNet, "classifier"),
    ],
)
def test_replace_classifier_for_supported_models(monkeypatch, name, dummy_cls, attr_name):
    monkeypatch.setitem(factory._BUILDERS, name, (lambda weights=None: dummy_cls(), "sentinel"))

    # num_classes == 1000 should retain original head
    model_1000 = factory.load_model(name, device=torch.device("cpu"), weights="none", num_classes=1000)
    layer_1000 = model_1000.classifier[-1] if name == "vgg16" else getattr(model_1000, attr_name)
    assert layer_1000.out_features == 1000

    # num_classes == 20 should replace head
    model_20 = factory.load_model(name, device=torch.device("cpu"), weights="none", num_classes=20)
    layer_20 = model_20.classifier[-1] if name == "vgg16" else getattr(model_20, attr_name)
    assert layer_20.out_features == 20
    assert layer_20.in_features == layer_1000.in_features


def test_checkpoint_not_found(monkeypatch, tmp_path):
    monkeypatch.setitem(factory._BUILDERS, "resnet50", (lambda weights=None: DummyResNet(), "sentinel"))
    missing_path = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        factory.load_model("resnet50", device=torch.device("cpu"), weights="none", checkpoint=str(missing_path))


def test_checkpoint_invalid_payload(monkeypatch, tmp_path):
    monkeypatch.setitem(factory._BUILDERS, "resnet50", (lambda weights=None: DummyResNet(), "sentinel"))
    invalid_path = tmp_path / "invalid.pt"
    torch.save(["not", "a", "dict"], invalid_path)
    with pytest.raises(ValueError, match="checkpoint must be a state_dict"):
        factory.load_model("resnet50", device=torch.device("cpu"), weights="none", checkpoint=str(invalid_path))


def test_checkpoint_loading_strips_module_prefix(monkeypatch, tmp_path):
    monkeypatch.setitem(factory._BUILDERS, "resnet50", (lambda weights=None: DummyResNet(), "sentinel"))
    ckpt_path = tmp_path / "valid_checkpoint.pt"

    weight = torch.randn(20, 2048)
    bias = torch.randn(20)
    payload = {
        "state_dict": {
            "module.fc.weight": weight,
            "module.fc.bias": bias,
        }
    }
    torch.save(payload, ckpt_path)

    model = factory.load_model(
        "resnet50",
        device=torch.device("cpu"),
        weights="none",
        num_classes=20,
        checkpoint=str(ckpt_path),
    )
    assert torch.equal(model.fc.weight, weight)
    assert torch.equal(model.fc.bias, bias)


def test_predict_contract():
    class SimpleClassifier(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Deterministic output for 3 classes: [1.0, 5.0, 2.0]
            batch = x.shape[0]
            return torch.tensor([[1.0, 5.0, 2.0]], dtype=torch.float32).repeat(batch, 1)

    model = SimpleClassifier().eval()

    # Test 4D image (1, 3, 4, 4)
    image_4d = torch.zeros(1, 3, 4, 4)
    pred_idx, conf, logits = factory.predict(model, image_4d)
    assert isinstance(pred_idx, int)
    assert pred_idx == 1
    assert isinstance(conf, float)
    assert conf > 0.9  # softmax([1, 5, 2]) for class 1 is ~0.9465
    assert logits.shape == (1, 3)

    # Test 3D image (3, 4, 4) should be automatically unsqueezed
    image_3d = torch.zeros(3, 4, 4)
    pred_idx_3d, conf_3d, logits_3d = factory.predict(model, image_3d)
    assert pred_idx_3d == 1
    assert conf_3d == pytest.approx(conf)
    assert logits_3d.shape == (1, 3)


def test_predict_sigmoid_uses_independent_class_confidence():
    class Scores(nn.Module):
        def forward(self, x):
            return torch.tensor([[1.0, 5.0, 2.0]]).repeat(x.shape[0], 1)
    predicted, confidence, _ = factory.predict(Scores(), torch.zeros(3, 4, 4), output_activation="sigmoid")
    assert predicted == 1
    assert confidence == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item())
