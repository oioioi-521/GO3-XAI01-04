"""Small torchvision model factory shared by all experiment methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torchvision import models


_BUILDERS = {
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT),
}


def _replace_classifier(model: torch.nn.Module, name: str, num_classes: int) -> None:
    if name == "vgg16":
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)


def load_model(
    name: str,
    device: torch.device,
    weights: str = "default",
    num_classes: int = 1000,
    checkpoint: str | None = None,
) -> torch.nn.Module:
    """Load a supported classifier and put it in evaluation mode.

    VOC configs intentionally require a 20-class checkpoint. ImageNet-1k
    weights cannot be interpreted as VOC ground-truth classes.
    """
    name = name.lower()
    if name not in _BUILDERS:
        raise ValueError(f"unsupported model {name!r}; choose from {sorted(_BUILDERS)}")
    builder, default_weights = _BUILDERS[name]
    normalized_weights = "none" if weights is None else str(weights).lower()
    if normalized_weights == "default":
        selected_weights = default_weights
    elif normalized_weights == "none":
        selected_weights = None
    else:
        raise ValueError(
            f"unsupported weights {weights!r}; choose from ['default', 'none']"
        )
    model = builder(weights=selected_weights)
    if num_classes != 1000:
        _replace_classifier(model, name, num_classes)

    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"checkpoint not found: {checkpoint_path}. "
                "VOC experiments require the 20-class checkpoint supplied by member B."
            )
        payload: Any = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if not isinstance(payload, dict):
            raise ValueError("checkpoint must be a state_dict or contain a state_dict key")
        clean_state: Dict[str, torch.Tensor] = {
            key.removeprefix("module."): value for key, value in payload.items()
        }
        model.load_state_dict(clean_state, strict=True)

    return model.to(device).eval()


@torch.inference_mode()
def predict(model: torch.nn.Module, image: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
    """Return ``(predicted class, confidence, logits)`` for one image."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    logits = model(image)
    probabilities = logits.softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    return int(predicted.item()), float(confidence.item()), logits
