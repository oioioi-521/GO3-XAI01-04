from __future__ import annotations

import torch

from experiments.attribution import kernelshap as kernelshap_module
from experiments.attribution.kernelshap import KernelSHAP


class TinyModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = inputs.mean(dim=(1, 2, 3))
        return torch.stack([pooled, -pooled], dim=1)


class FakeCaptumKernelShap:
    latest_kwargs = None

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def attribute(self, **kwargs):
        FakeCaptumKernelShap.latest_kwargs = kwargs
        image = kwargs["inputs"]
        height, width = image.shape[-2:]
        gradient = torch.arange(
            height * width, dtype=image.dtype, device=image.device
        ).view(1, 1, height, width)
        return gradient.expand_as(image)


class FakeSignedCaptumKernelShap(FakeCaptumKernelShap):
    def attribute(self, **kwargs):
        image = kwargs["inputs"]
        height, width = image.shape[-2:]
        signed = torch.linspace(-1, 1, height * width).view(1, 1, height, width)
        return signed.expand_as(image)


def test_feature_mask_contains_grid_regions() -> None:
    method = KernelSHAP(TinyModel(), feature_grid_size=2)
    mask = method._feature_mask(4, 6, torch.device("cpu"))
    assert mask.shape == (1, 1, 4, 6)
    assert set(mask.unique().tolist()) == {0, 1, 2, 3}


def test_attribute_matches_pipeline_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        kernelshap_module,
        "_load_captum_kernel_shap",
        lambda: FakeCaptumKernelShap,
    )
    method = KernelSHAP(
        TinyModel(), n_samples=8, perturbations_per_eval=4, feature_grid_size=2
    )
    image = torch.ones(1, 3, 4, 6)
    saliency = method.attribute(image, target=0, baseline=torch.zeros(1, 3, 1, 1))
    assert saliency.shape == (4, 6)
    assert float(saliency.min()) == 0.0
    assert float(saliency.max()) == 1.0
    assert FakeCaptumKernelShap.latest_kwargs["n_samples"] == 8
    assert FakeCaptumKernelShap.latest_kwargs["feature_mask"].shape == (1, 1, 4, 6)


def test_signed_ranking_is_preserved_for_morf(monkeypatch) -> None:
    monkeypatch.setattr(
        kernelshap_module, "_load_captum_kernel_shap",
        lambda: FakeSignedCaptumKernelShap,
    )
    method = KernelSHAP(TinyModel(), n_samples=8, feature_grid_size=2)
    saliency = method.attribute(torch.ones(1, 3, 4, 4), target=0)
    assert float(saliency[0, 0]) == 0.0
    assert float(saliency[-1, -1]) == 1.0
    assert float(saliency[0, 0]) < float(saliency[2, 2])
