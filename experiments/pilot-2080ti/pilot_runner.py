#!/usr/bin/env python3
"""Single-GPU XAI cost calibration runner.

This file is intentionally self-contained and does not import any course or
repository implementation.  It uses torchvision's official ImageNet weights
when they can be downloaded, and falls back to explicitly marked random
weights only when a download or model load fails.

The default calibration is the six-method by three-model matrix requested for
XAI01-04: Grad-CAM, KernelSHAP, Integrated Gradients, RISE, input Ablation,
and Captum LIME on one deterministic synthetic 224x224 image.  Every timed
replicate is appended to JSONL and summaries are rewritten after each row so a
deadline, timeout, OOM, or process interruption is recoverable.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep all caches and artifacts in the independent pilot directory.  This is
# set before importing torchvision so its download helper follows this path.
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / "weights_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "mpl_cache"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    VGG16_Weights,
    densenet121,
    resnet50,
    vgg16,
)

from captum.attr import FeatureAblation, IntegratedGradients, KernelShap, LayerAttribution, LayerGradCam, Lime
from captum._utils.models.linear_model import SkLearnLasso

try:
    import psutil
except Exception:  # pragma: no cover - optional fallback is recorded
    psutil = None

try:
    import pynvml
except Exception:  # pragma: no cover - optional fallback is recorded
    pynvml = None


METHOD_ORDER = ["gradcam", "kernelshap", "ig", "rise", "ablation", "lime"]
MODEL_ORDER = ["resnet50", "vgg16", "densenet121"]
DEFAULT_INTERNAL_BATCH = 16
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TaskTimeout(RuntimeError):
    """Raised by the per-task wall-clock alarm."""


class SkipTask(RuntimeError):
    """Internal signal used when a global deadline has elapsed."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return jsonable(value.detach().cpu().item())
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(jsonable(value), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def safe_mean(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


def safe_min(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.min(xs)) if xs else None


def safe_max(values: Iterable[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.max(xs)) if xs else None


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_normalize(raw: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=raw.dtype, device=raw.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=raw.dtype, device=raw.device).view(1, 3, 1, 1)
    return (raw - mean) / std


def make_synthetic_image(seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized input and the raw [0, 1] image used to make it."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # A deterministic natural-image-like texture avoids a totally flat input
    # while retaining the explicit synthetic/timing-only scope.
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, IMAGE_SIZE),
        torch.linspace(0.0, 1.0, IMAGE_SIZE),
        indexing="ij",
    )
    phase = torch.rand((3,), generator=gen)
    raw = torch.empty((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    raw[0, 0] = 0.45 + 0.20 * torch.sin(2.0 * math.pi * (xx * (1.2 + phase[0]) + yy * 0.35))
    raw[0, 1] = 0.46 + 0.18 * torch.sin(2.0 * math.pi * (yy * (1.0 + phase[1]) + xx * 0.25))
    raw[0, 2] = 0.42 + 0.16 * torch.sin(2.0 * math.pi * ((xx + yy) * (0.75 + phase[2])))
    raw += 0.06 * torch.rand(raw.shape, generator=gen)
    raw = raw.clamp(0.0, 1.0)
    return image_normalize(raw).to(device), raw.to(device)


def black_baseline(device: torch.device) -> torch.Tensor:
    # “Black image” means raw RGB zero followed by ImageNet normalization;
    # this must not be confused with a zero standardized tensor.
    raw = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
    return image_normalize(raw)


def make_grid_feature_mask(grid: int, device: torch.device) -> torch.Tensor:
    """Return one feature id per non-overlapping grid region.

    The final row/column absorb the remainder (224 is divisible by 7 and 8),
    so the default 7x7 mask has exactly 49 groups and no superpixel step.
    """
    rows = torch.arange(IMAGE_SIZE, device=device) * grid // IMAGE_SIZE
    cols = torch.arange(IMAGE_SIZE, device=device) * grid // IMAGE_SIZE
    mask = rows[:, None] * grid + cols[None, :]
    return mask.to(torch.long).view(1, 1, IMAGE_SIZE, IMAGE_SIZE)


class CountingModel(nn.Module):
    """Wrapper that records model forward calls and actual examples."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.forward_calls = 0
        self.forward_samples = 0
        self.backward_hook_calls = 0
        self._backward_hook = self.register_full_backward_hook(self._count_backward)

    def _count_backward(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.backward_hook_calls += 1

    def reset_counts(self) -> None:
        self.forward_calls = 0
        self.forward_samples = 0
        self.backward_hook_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.forward_samples += int(x.shape[0])
        return self.inner(x)


@dataclasses.dataclass
class ModelBundle:
    name: str
    model: CountingModel
    target_layer: nn.Module
    target_layer_name: str
    weights_source: str
    weights_status: str
    weights_path: Optional[str]
    weights_sha256: Optional[str]
    load_error: Optional[str]
    load_wall_sec: Optional[float]
    target_class: int
    target_score: float


MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "builder": resnet50,
        "weights": ResNet50_Weights.DEFAULT,
        "target_layer": "layer4.2.conv3",
    },
    "vgg16": {
        "builder": vgg16,
        "weights": VGG16_Weights.DEFAULT,
        "target_layer": "features.28",
    },
    "densenet121": {
        "builder": densenet121,
        "weights": DenseNet121_Weights.DEFAULT,
        "target_layer": "features.denseblock4.denselayer16.conv2",
    },
}


def resolve_module(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    for part in name.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def locate_weight_file(weight_url: str) -> Optional[Path]:
    filename = Path(weight_url.split("?")[0]).name
    candidates = [
        ROOT / "weights_cache" / "checkpoints" / filename,
        Path(torch.hub.get_dir()) / "checkpoints" / filename,
        ROOT / "weights_cache" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_model(name: str, device: torch.device, x: torch.Tensor) -> ModelBundle:
    spec = MODEL_SPECS[name]
    builder = spec["builder"]
    weights = spec["weights"]
    source = str(weights.url)
    load_error: Optional[str] = None
    weights_status = "official_pretrained"
    load_start = time.perf_counter()
    try:
        inner = builder(weights=weights)
    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        weights_status = "timing_only_random_fallback"
        inner = builder(weights=None)
    inner.eval().to(device)
    model = CountingModel(inner).eval().to(device)
    target_layer = resolve_module(inner, spec["target_layer"])
    model.reset_counts()
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(dim=1)
        target_class = int(logits.argmax(dim=1).item())
    target_score = float(probs[0, target_class].item())
    load_wall_sec = time.perf_counter() - load_start
    model.reset_counts()
    weight_path = locate_weight_file(source) if weights_status == "official_pretrained" else None
    return ModelBundle(
        name=name,
        model=model,
        target_layer=target_layer,
        target_layer_name=spec["target_layer"],
        weights_source=source,
        weights_status=weights_status,
        weights_path=str(weight_path) if weight_path else None,
        weights_sha256=sha256_file(weight_path) if weight_path else None,
        load_error=load_error,
        load_wall_sec=load_wall_sec,
        target_class=target_class,
        target_score=target_score,
    )


def model_signature(bundle: ModelBundle) -> Dict[str, Any]:
    return {
        "model": bundle.name,
        "weights_source": bundle.weights_source,
        "weights_status": bundle.weights_status,
        "weights_path": bundle.weights_path,
        "weights_sha256": bundle.weights_sha256,
        "load_error": bundle.load_error,
        "model_load_wall_sec": bundle.load_wall_sec,
        "target_layer": bundle.target_layer_name,
        "target_class": bundle.target_class,
        "target_score": bundle.target_score,
    }


class TelemetrySampler:
    """Low-rate process/GPU sampler used only around each timed replicate."""

    def __init__(self, device_index: int = 0, interval_sec: float = 0.10):
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_ok = False
        self._handle: Any = None
        self._nvml_error: Optional[str] = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._nvml_ok = True
            except Exception as exc:
                self._nvml_error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 3))
        self._sample_once()

    def close(self) -> None:
        if self._nvml_ok and pynvml is not None:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        self._nvml_ok = False

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_sec)

    def _sample_once(self) -> None:
        row: Dict[str, Any] = {"t": time.perf_counter()}
        if psutil is not None:
            try:
                row["cpu_rss_bytes"] = int(psutil.Process().memory_info().rss)
            except Exception:
                row["cpu_rss_bytes"] = None
        else:
            row["cpu_rss_bytes"] = None
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                row["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
                row["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
        if self._nvml_ok and pynvml is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                row.update(
                    {
                        "gpu_mem_used_bytes": int(mem.used),
                        "gpu_mem_total_bytes": int(mem.total),
                        "gpu_util_percent": int(util.gpu),
                        "gpu_mem_util_percent": int(util.memory),
                        "gpu_temperature_c": int(pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)),
                        "gpu_power_w": float(pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0,
                    }
                )
            except Exception as exc:
                row["gpu_telemetry_error"] = f"{type(exc).__name__}: {exc}"
        self.samples.append(row)

    def summary(self) -> Dict[str, Any]:
        def vals(key: str) -> List[float]:
            return [float(s[key]) for s in self.samples if s.get(key) is not None]

        return {
            "sample_count": len(self.samples),
            "sampling_interval_sec": self.interval_sec,
            "cpu_rss_peak_mib": (safe_max(vals("cpu_rss_bytes")) or 0.0) / 2**20 if vals("cpu_rss_bytes") else None,
            "cuda_allocated_peak_mib": (safe_max(vals("cuda_allocated_bytes")) or 0.0) / 2**20 if vals("cuda_allocated_bytes") else None,
            "cuda_reserved_peak_mib": (safe_max(vals("cuda_reserved_bytes")) or 0.0) / 2**20 if vals("cuda_reserved_bytes") else None,
            "gpu_mem_used_peak_mib": (safe_max(vals("gpu_mem_used_bytes")) or 0.0) / 2**20 if vals("gpu_mem_used_bytes") else None,
            "gpu_mem_total_mib": (safe_max(vals("gpu_mem_total_bytes")) or 0.0) / 2**20 if vals("gpu_mem_total_bytes") else None,
            "gpu_util_mean_percent": safe_mean(vals("gpu_util_percent")),
            "gpu_util_peak_percent": safe_max(vals("gpu_util_percent")),
            "gpu_temperature_peak_c": safe_max(vals("gpu_temperature_c")),
            "gpu_power_mean_w": safe_mean(vals("gpu_power_w")),
            "gpu_power_peak_w": safe_max(vals("gpu_power_w")),
            "nvml_available": self._nvml_ok,
            "nvml_error": self._nvml_error,
        }


def device_snapshot(device_index: int = 0) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "python": sys.version,
        "python_executable": sys.executable,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
    }
    if psutil is not None:
        with contextlib.suppress(Exception):
            result["ram_total_mib"] = psutil.virtual_memory().total / 2**20
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_index)
        result.update(
            {
                "gpu_name": props.name,
                "gpu_total_mib": props.total_memory / 2**20,
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_index": device_index,
            }
        )
    return result


def feature_mask_grid(grid: int, device: torch.device) -> torch.Tensor:
    return make_grid_feature_mask(grid, device)


def to_heatmap(attr: torch.Tensor) -> torch.Tensor:
    """Convert an attribution tensor to a signed [1, 1, H, W] heatmap."""
    if not isinstance(attr, torch.Tensor):
        raise TypeError(f"attribution must be a Tensor, got {type(attr)}")
    if attr.ndim == 4 and attr.shape[1] == 1:
        heat = attr
    elif attr.ndim == 4:
        heat = attr.sum(dim=1, keepdim=True)
    elif attr.ndim == 3:
        heat = attr.sum(dim=1, keepdim=True) if attr.shape[1] == 3 else attr.unsqueeze(1)
    elif attr.ndim == 2:
        heat = attr.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"unsupported attribution shape {tuple(attr.shape)}")
    if heat.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        heat = F.interpolate(heat, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return heat


def attr_stats(heat: torch.Tensor) -> Dict[str, Any]:
    finite = torch.isfinite(heat)
    finite_fraction = float(finite.float().mean().item())
    finite_heat = torch.where(finite, heat, torch.zeros_like(heat))
    abs_heat = finite_heat.abs()
    max_abs = float(abs_heat.max().item()) if abs_heat.numel() else 0.0
    return {
        "attribution_shape": list(heat.shape),
        "attribution_finite_fraction": finite_fraction,
        "attribution_invalid": bool(finite_fraction < 1.0 or max_abs == 0.0),
        "attribution_abs_max": max_abs,
        "attribution_signed_sum": float(finite_heat.sum().item()),
        "attribution_abs_sum": float(abs_heat.sum().item()),
        "attribution_positive_fraction": float((finite_heat > 0).float().mean().item()),
    }


def run_gradcam(bundle: ModelBundle, x: torch.Tensor, _baseline: torch.Tensor, _cfg: Dict[str, Any]) -> torch.Tensor:
    method = LayerGradCam(bundle.model, bundle.target_layer)
    attr = method.attribute(x, target=bundle.target_class, relu_attributions=False)
    return to_heatmap(attr)


def run_ig(bundle: ModelBundle, x: torch.Tensor, baseline: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    steps = int(cfg["n_steps"])
    internal_batch = int(cfg["internal_batch_size"])
    method = IntegratedGradients(bundle.model)
    attr = method.attribute(
        x,
        baselines=baseline,
        target=bundle.target_class,
        n_steps=steps,
        internal_batch_size=min(internal_batch, steps),
        method="gausslegendre",
    )
    return to_heatmap(attr)


def run_kernelshap(bundle: ModelBundle, x: torch.Tensor, baseline: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    method = KernelShap(bundle.model)
    attr = method.attribute(
        x,
        baselines=baseline,
        target=bundle.target_class,
        feature_mask=feature_mask_grid(int(cfg["grid"]), x.device),
        n_samples=int(cfg["n_samples"]),
        perturbations_per_eval=int(cfg["internal_batch_size"]),
        return_input_shape=True,
        show_progress=False,
    )
    return to_heatmap(attr)


def run_lime(bundle: ModelBundle, x: torch.Tensor, baseline: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    method = Lime(bundle.model, interpretable_model=SkLearnLasso(alpha=0.01))
    attr = method.attribute(
        x,
        baselines=baseline,
        target=bundle.target_class,
        feature_mask=feature_mask_grid(int(cfg["grid"]), x.device),
        n_samples=int(cfg["n_samples"]),
        perturbations_per_eval=int(cfg["internal_batch_size"]),
        return_input_shape=True,
        show_progress=False,
    )
    return to_heatmap(attr)


def run_ablation(bundle: ModelBundle, x: torch.Tensor, baseline: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    method = FeatureAblation(bundle.model)
    attr = method.attribute(
        x,
        baselines=baseline,
        target=bundle.target_class,
        feature_mask=feature_mask_grid(int(cfg["grid"]), x.device),
        perturbations_per_eval=int(cfg["internal_batch_size"]),
        show_progress=False,
    )
    return to_heatmap(attr)


def run_rise(bundle: ModelBundle, x: torch.Tensor, baseline: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """RISE with random Bernoulli 7x7 masks and batched model calls."""
    n_masks = int(cfg["n_masks"])
    grid = int(cfg["grid"])
    keep_prob = float(cfg["keep_prob"])
    batch_size = int(cfg["internal_batch_size"])
    gen = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
    result = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=x.device, dtype=x.dtype)
    seen = 0
    while seen < n_masks:
        batch = min(batch_size, n_masks - seen)
        low = torch.rand((batch, 1, grid, grid), generator=gen, dtype=x.dtype)
        # Interpolation kernels do not support Bool tensors on CUDA; retain a
        # float mask whose values are exactly 0/1 before upsampling.
        low = (low < keep_prob).to(device=x.device, dtype=x.dtype)
        masks = F.interpolate(low, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        masked = baseline + masks * (x - baseline)
        logits = bundle.model(masked)
        scores = logits.softmax(dim=1)[:, bundle.target_class].view(batch, 1, 1, 1)
        result += (scores * masks).sum(dim=0, keepdim=True)
        seen += batch
    result /= max(1.0, n_masks * keep_prob)
    return result


METHOD_FUNCS: Dict[str, Callable[[ModelBundle, torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor]] = {
    "gradcam": run_gradcam,
    "ig": run_ig,
    "kernelshap": run_kernelshap,
    "rise": run_rise,
    "ablation": run_ablation,
    "lime": run_lime,
}


def method_config(method: str, internal_batch_size: int, seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "method": method,
        "dtype": "float32",
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "internal_batch_size": internal_batch_size,
        "outer_batch_size": 1,
        "baseline": "raw_black_then_imagenet_normalize",
        "channel_aggregation": "sum_signed_channels_then_heatmap",
        "attribution_resize": "bilinear_to_224_if_needed",
        "output_rule": "signed_heatmap_saved; absolute_heatmap_stats_only",
        "score_function": "softmax_probability" if method == "rise" else "logit",
        "seed": seed,
    }
    if method == "ig":
        config.update({"n_steps": 32, "captum_method": "gausslegendre"})
    elif method == "rise":
        config.update({"n_masks": 512, "grid": 7, "keep_prob": 0.5, "mask_interpolation": "bilinear"})
    elif method in {"kernelshap", "lime"}:
        config.update({"n_samples": 512, "grid": 7})
        if method == "kernelshap":
            config.update({"variant": "Captum KernelShap"})
        else:
            config.update({"variant": "Captum Lime", "surrogate": "SkLearnLasso(alpha=0.01)"})
    elif method == "ablation":
        config.update({"grid": 7, "regions": 49, "variant": "Captum FeatureAblation input-region elimination"})
    elif method == "gradcam":
        config.update({"variant": "Captum LayerGradCam", "relu_attributions": False})
    if overrides:
        config.update(overrides)
    return config


def timed_call(
    fn: Callable[[], torch.Tensor],
    sampler: TelemetrySampler,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    sampler.start()
    start_wall = time.perf_counter()
    start_event = end_event = None
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    result: Optional[torch.Tensor] = None
    try:
        result = fn()
        if device.type == "cuda":
            end_event.record()  # type: ignore[union-attr]
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - start_wall
    finally:
        sampler.stop()
    cuda_sec: Optional[float] = None
    if start_event is not None and end_event is not None:
        with contextlib.suppress(Exception):
            cuda_sec = float(start_event.elapsed_time(end_event) / 1000.0)
    mem: Dict[str, Any] = {}
    if device.type == "cuda":
        mem = {
            "cuda_max_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "cuda_max_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
        }
    mem.update(sampler.summary())
    mem.update({"attribution_wall_sec": wall, "attribution_cuda_event_sec": cuda_sec, "timing_synchronized": device.type == "cuda"})
    if result is None:
        raise RuntimeError("timed function returned no result")
    return result, mem


def install_alarm(seconds: float) -> Any:
    if seconds <= 0:
        return None
    previous = signal.getsignal(signal.SIGALRM)

    def handler(_signum: int, _frame: Any) -> None:
        raise TaskTimeout(f"task exceeded {seconds:.1f}s timeout")

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous


def clear_alarm(previous: Any) -> None:
    with contextlib.suppress(Exception):
        signal.setitimer(signal.ITIMER_REAL, 0)
    if previous is not None:
        signal.signal(signal.SIGALRM, previous)


def one_replicate(
    bundle: ModelBundle,
    x: torch.Tensor,
    baseline: torch.Tensor,
    method: str,
    config: Dict[str, Any],
    replicate: int,
    task_timeout_sec: float,
    device: torch.device,
) -> Dict[str, Any]:
    seed = int(config["seed"]) + replicate * 1009
    set_all_seeds(seed)
    config = dict(config)
    config["seed"] = seed
    bundle.model.reset_counts()
    sampler = TelemetrySampler(device_index=device.index or 0)
    previous_alarm = install_alarm(task_timeout_sec)
    warmup_error: Optional[str] = None
    warmup_start = time.perf_counter()
    warmup_cuda_start = warmup_cuda_end = None
    try:
        # One explicit warmup is required before each measured replicate.
        try:
            if device.type == "cuda":
                warmup_cuda_start = torch.cuda.Event(enable_timing=True)
                warmup_cuda_end = torch.cuda.Event(enable_timing=True)
                warmup_cuda_start.record()
            with torch.enable_grad() if method in {"gradcam", "ig"} else torch.no_grad():
                _ = METHOD_FUNCS[method](bundle, x, baseline, config)
            if device.type == "cuda":
                warmup_cuda_end.record()
                torch.cuda.synchronize(device)
            warmup_wall_sec = time.perf_counter() - warmup_start
            warmup_cuda_event_sec = None
            if warmup_cuda_start is not None and warmup_cuda_end is not None:
                warmup_cuda_event_sec = float(warmup_cuda_start.elapsed_time(warmup_cuda_end) / 1000.0)
        except Exception as exc:
            warmup_error = f"{type(exc).__name__}: {exc}"
            raise
        bundle.model.reset_counts()
        set_all_seeds(seed)
        start_forward = bundle.model.forward_calls
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if method in {"gradcam", "ig"}:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()
        with context:
            heat, timing = timed_call(
                lambda: METHOD_FUNCS[method](bundle, x, baseline, config),
                sampler,
                device,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stats = attr_stats(heat)
        timing.update(
            {
                "forward_calls": int(bundle.model.forward_calls - start_forward),
                "forward_samples": int(bundle.model.forward_samples),
                "backward_hook_calls": int(bundle.model.backward_hook_calls),
                "processed_perturbation_samples": int(bundle.model.forward_samples),
            }
        )
        artifact_dir = ROOT / "artifacts" / "calibration_attributions"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = f"{bundle.name}__{method}__rep{replicate}__{stable_hash(config)}.pt"
        artifact_path = artifact_dir / artifact_name
        io_start = time.perf_counter()
        torch.save(
            {
                "attribution_signed_heatmap": heat.detach().cpu().float(),
                "model": model_signature(bundle),
                "method_config": config,
                "scope": "synthetic_calibration_timing_only",
            },
            artifact_path,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        io_sec = time.perf_counter() - io_start
        return {
            "status": "ok",
            "failure_stage": None,
            "failure_reason": None,
            "warmup_error": warmup_error,
            "replicate": replicate,
            "seed": seed,
            "method_config": config,
            "model": model_signature(bundle),
            "quality_scope": "synthetic_calibration_timing_only",
            "warmup_wall_sec": warmup_wall_sec,
            "warmup_cuda_event_sec": warmup_cuda_event_sec,
            "attribution_cache_path": str(artifact_path),
            "io_sec": io_sec,
            **timing,
            **stats,
        }
    except TaskTimeout as exc:
        return {
            "status": "timeout",
            "failure_stage": "warmup_or_attribution",
            "failure_reason": str(exc),
            "warmup_error": warmup_error,
            "replicate": replicate,
            "seed": seed,
            "method_config": config,
            "model": model_signature(bundle),
            "quality_scope": "synthetic_calibration_timing_only",
            "warmup_wall_sec": None,
            "warmup_cuda_event_sec": None,
            "attribution_cache_path": None,
            "io_sec": None,
            "attribution_wall_sec": None,
            "attribution_cuda_event_sec": None,
            "timing_synchronized": device.type == "cuda",
            "forward_calls": int(bundle.model.forward_calls),
            "forward_samples": int(bundle.model.forward_samples),
            "backward_hook_calls": int(bundle.model.backward_hook_calls),
            "processed_perturbation_samples": int(bundle.model.forward_samples),
        }
    except torch.cuda.OutOfMemoryError as exc:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        return {
            "status": "oom",
            "failure_stage": "warmup_or_attribution",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "warmup_error": warmup_error,
            "replicate": replicate,
            "seed": seed,
            "method_config": config,
            "model": model_signature(bundle),
            "quality_scope": "synthetic_calibration_timing_only",
            "warmup_wall_sec": None,
            "warmup_cuda_event_sec": None,
            "attribution_cache_path": None,
            "io_sec": None,
            "attribution_wall_sec": None,
            "attribution_cuda_event_sec": None,
            "timing_synchronized": device.type == "cuda",
            "forward_calls": int(bundle.model.forward_calls),
            "forward_samples": int(bundle.model.forward_samples),
            "backward_hook_calls": int(bundle.model.backward_hook_calls),
            "processed_perturbation_samples": int(bundle.model.forward_samples),
        }
    except Exception as exc:
        return {
            "status": "error",
            "failure_stage": "warmup_or_attribution",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "warmup_error": warmup_error,
            "replicate": replicate,
            "seed": seed,
            "method_config": config,
            "model": model_signature(bundle),
            "quality_scope": "synthetic_calibration_timing_only",
            "warmup_wall_sec": None,
            "warmup_cuda_event_sec": None,
            "attribution_cache_path": None,
            "io_sec": None,
            "attribution_wall_sec": None,
            "attribution_cuda_event_sec": None,
            "timing_synchronized": device.type == "cuda",
            "forward_calls": int(bundle.model.forward_calls),
            "forward_samples": int(bundle.model.forward_samples),
            "backward_hook_calls": int(bundle.model.backward_hook_calls),
            "processed_perturbation_samples": int(bundle.model.forward_samples),
        }
    finally:
        clear_alarm(previous_alarm)
        sampler.close()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
            except json.JSONDecodeError:
                continue
    return rows


def record_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return (str(row.get("model", {}).get("model")), str(row.get("method_config", {}).get("method")), int(row.get("replicate", -1)))


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("record_type") != "calibration_replicate":
            continue
        key = (str(row.get("model", {}).get("model")), str(row.get("method_config", {}).get("method")))
        groups.setdefault(key, []).append(row)
    result: List[Dict[str, Any]] = []
    for (model, method), rs in sorted(groups.items(), key=lambda kv: (MODEL_ORDER.index(kv[0][0]) if kv[0][0] in MODEL_ORDER else 99, METHOD_ORDER.index(kv[0][1]) if kv[0][1] in METHOD_ORDER else 99)):
        oks = [r for r in rs if r.get("status") == "ok"]
        result.append(
            {
                "model": model,
                "method": method,
                "replicates_seen": len(rs),
                "replicates_ok": len(oks),
                "statuses": {s: sum(1 for r in rs if r.get("status") == s) for s in sorted({str(r.get("status")) for r in rs})},
                "attribution_wall_sec_mean": safe_mean(r.get("attribution_wall_sec") for r in oks),
                "attribution_wall_sec_min": safe_min(r.get("attribution_wall_sec") for r in oks),
                "attribution_wall_sec_max": safe_max(r.get("attribution_wall_sec") for r in oks),
                "attribution_cuda_event_sec_mean": safe_mean(r.get("attribution_cuda_event_sec") for r in oks),
                "forward_calls_mean": safe_mean(r.get("forward_calls") for r in oks),
                "forward_samples_mean": safe_mean(r.get("forward_samples") for r in oks),
                "cuda_allocated_peak_mib_max": safe_max(r.get("cuda_max_memory_allocated_mib") for r in oks),
                "cuda_reserved_peak_mib_max": safe_max(r.get("cuda_max_memory_reserved_mib") for r in oks),
                "gpu_mem_used_peak_mib_max": safe_max(r.get("gpu_mem_used_peak_mib") for r in oks),
                "cpu_rss_peak_mib_max": safe_max(r.get("cpu_rss_peak_mib") for r in oks),
                "gpu_util_mean_percent_mean": safe_mean(r.get("gpu_util_mean_percent") for r in oks),
                "io_sec_mean": safe_mean(r.get("io_sec") for r in oks),
                "quality_scope": "synthetic_calibration_timing_only",
            }
        )
    return result


def render_markdown(snapshot: Dict[str, Any], rows: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# XAI01-04 试跑：合成图计时校准",
        "",
        f"生成时间（UTC）：`{snapshot.get('generated_utc')}`",
        "",
        "合成图校准仅用于计时，不是 ImageNet、VOC 或 CHNCXR 的质量结果。",
        "",
        "## 环境",
        "",
        f"- 设备：`{snapshot.get('environment', {}).get('device')}`；GPU：`{snapshot.get('environment', {}).get('gpu_name')}`",
        f"- PyTorch / torchvision / CUDA runtime：`{snapshot.get('environment', {}).get('torch_version')}` / `{snapshot.get('environment', {}).get('torchvision_version')}` / `{snapshot.get('environment', {}).get('torch_cuda_runtime')}`",
        f"- 官方权重下载目录：`{ROOT / 'weights_cache'}`",
        f"- 计时副本记录总数：`{len(rows)}`",
        "",
        "## 方法 × 模型计时汇总",
        "",
        "| 模型 | 方法 | 成功/总计 | 墙钟均值(s) | CUDA event 均值(s) | 前向样本数均值 | CUDA 保留峰值(MiB) | GPU 显存峰值(MiB) | CPU RSS 峰值(MiB) | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        def fmt(v: Any) -> str:
            return "null" if v is None else f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)

        lines.append(
            f"| {row['model']} | {row['method']} | {row['replicates_ok']}/{row['replicates_seen']} | {fmt(row['attribution_wall_sec_mean'])} | {fmt(row['attribution_cuda_event_sec_mean'])} | {fmt(row['forward_samples_mean'])} | {fmt(row['cuda_reserved_peak_mib_max'])} | {fmt(row['gpu_mem_used_peak_mib_max'])} | {fmt(row['cpu_rss_peak_mib_max'])} | {row['statuses']} |"
        )
    lines += [
        "",
        "## 复现命令",
        "",
        "```bash",
        f"cd {ROOT}",
        ".venv/bin/python pilot_runner.py --calibration --resume --global-deadline-sec 900 --task-timeout-sec 180",
        "```",
        "",
        "原始记录追加写入 `calibration_records.jsonl`；聚合 JSON 为 `calibration_summary.json`。",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibration", action="store_true", help="run the 6x3 synthetic calibration matrix")
    p.add_argument("--smoke", action="store_true", help="run a tiny ResNet50 smoke test for all methods")
    p.add_argument("--resume", action="store_true", help="skip successful replicate keys already in JSONL")
    p.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    p.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--internal-batch-size", type=int, default=DEFAULT_INTERNAL_BATCH)
    p.add_argument("--global-deadline-sec", type=float, default=900.0)
    p.add_argument("--task-timeout-sec", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--image-seed", type=int, default=20260911)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--records", type=Path, default=ROOT / "calibration_records.jsonl")
    p.add_argument("--summary", type=Path, default=ROOT / "calibration_summary.json")
    p.add_argument("--report", type=Path, default=ROOT / "calibration_report.md")
    return p


def run(args: argparse.Namespace) -> int:
    start_wall = time.perf_counter()
    if args.device == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            print("CUDA is unavailable in this process; rerun with require_escalated GPU access or pass --allow-cpu.", file=sys.stderr)
            return 2
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
    set_all_seeds(args.seed)
    x, raw = make_synthetic_image(args.image_seed, device)
    baseline = black_baseline(device)
    records = read_jsonl(args.records)
    done_keys = {record_key(r) for r in records if r.get("record_type") == "calibration_replicate" and r.get("status") == "ok"}
    snapshot: Dict[str, Any] = {
        "run_id": f"calibration-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{stable_hash({'seed': args.seed, 'image_seed': args.image_seed})}",
        "generated_utc": utc_now(),
        "scope": "synthetic_calibration_timing_only",
        "command": " ".join([str(x) for x in sys.argv]),
        "arguments": vars(args),
        "environment": device_snapshot(device.index or 0),
        "seed": args.seed,
        "image_seed": args.image_seed,
        "raw_image_sha256": hashlib.sha256(raw.detach().cpu().numpy().tobytes()).hexdigest(),
        "input_preprocessing": {
            "raw_range": "[0,1]",
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "resize": "synthetic generated directly at 224x224",
            "baseline": "raw black RGB zeros then ImageNet normalization",
        },
        "matrix": {"models": args.models, "methods": args.methods, "replicates": args.replicates},
        "deadline_sec": args.global_deadline_sec,
        "task_timeout_sec": args.task_timeout_sec,
    }
    write_json(ROOT / "environment_snapshot.json", snapshot)
    bundles: Dict[str, ModelBundle] = {}
    deadline = start_wall + float(args.global_deadline_sec)
    combos = [(m, meth) for m in args.models for meth in args.methods]
    # Reorder to put ResNet50/Grad-CAM and KernelSHAP first as requested.
    combos.sort(key=lambda mm: (MODEL_ORDER.index(mm[0]), METHOD_ORDER.index(mm[1])))
    if args.smoke:
        combos = [("resnet50", method) for method in METHOD_ORDER]
        args.replicates = 1
        args.task_timeout_sec = min(float(args.task_timeout_sec), 120.0)
    if not args.calibration and not args.smoke:
        print("choose --calibration or --smoke", file=sys.stderr)
        return 2
    try:
        for model_name, method in combos:
            if time.perf_counter() >= deadline:
                row = {
                    "record_type": "calibration_replicate",
                    "status": "skipped",
                    "failure_stage": "global_deadline",
                    "failure_reason": "global_deadline_exceeded",
                    "model": {"model": model_name},
                    "method_config": method_config(method, args.internal_batch_size, args.seed),
                    "replicate": None,
                    "quality_scope": "synthetic_calibration_timing_only",
                }
                append_jsonl(args.records, row)
                records.append(row)
                continue
            if model_name not in bundles:
                if time.perf_counter() >= deadline:
                    continue
                print(f"[{utc_now()}] loading {model_name}", flush=True)
                bundles[model_name] = build_model(model_name, device, x)
                # Refresh environment snapshot once CUDA and a real model are active.
                snapshot["models_loaded"] = {name: model_signature(bundle) for name, bundle in bundles.items()}
                write_json(ROOT / "environment_snapshot.json", snapshot)
            bundle = bundles[model_name]
            for replicate in range(args.replicates):
                smoke_overrides: Dict[str, Any] = {}
                if args.smoke:
                    if method == "ig":
                        smoke_overrides["n_steps"] = 4
                    elif method == "rise":
                        smoke_overrides["n_masks"] = 16
                    elif method in {"kernelshap", "lime"}:
                        smoke_overrides["n_samples"] = 16
                config = method_config(
                    method,
                    args.internal_batch_size,
                    args.seed + 10000 * (MODEL_ORDER.index(model_name) + 1) + METHOD_ORDER.index(method),
                    overrides=smoke_overrides,
                )
                key = (model_name, method, replicate)
                if args.resume and key in done_keys:
                    print(f"[{utc_now()}] resume skip {model_name}/{method}/rep{replicate}", flush=True)
                    continue
                if time.perf_counter() >= deadline:
                    row = {
                        "record_type": "calibration_replicate",
                        "status": "skipped",
                        "failure_stage": "global_deadline",
                        "failure_reason": "global_deadline_exceeded",
                        "model": model_signature(bundle),
                        "method_config": config,
                        "replicate": replicate,
                        "quality_scope": "synthetic_calibration_timing_only",
                    }
                    append_jsonl(args.records, {**row, "record_type": "calibration_replicate"})
                    records.append(row)
                    continue
                print(f"[{utc_now()}] run {model_name}/{method}/rep{replicate} config={stable_hash(config)}", flush=True)
                row = one_replicate(bundle, x, baseline, method, config, replicate, float(args.task_timeout_sec), device)
                row.update(
                    {
                        "record_type": "calibration_replicate",
                        "run_id": snapshot["run_id"],
                        "utc": utc_now(),
                        "config_hash": stable_hash(config),
                        "elapsed_wall_sec": time.perf_counter() - start_wall,
                        "remaining_deadline_sec": max(0.0, deadline - time.perf_counter()),
                        "device_snapshot": snapshot["environment"],
                        "synthetic_input": True,
                    }
                )
                append_jsonl(args.records, row)
                records.append(row)
                write_json(
                    args.summary,
                    {
                        "generated_utc": utc_now(),
                        "scope": "synthetic_calibration_timing_only",
                        "environment": snapshot["environment"],
                        "rows": aggregate(records),
                    },
                )
                args.report.write_text(render_markdown(snapshot, records, aggregate(records)), encoding="utf-8")
                print(
                    f"[{utc_now()}] {row.get('status')} wall={row.get('attribution_wall_sec')}s forwards={row.get('forward_samples')} remaining={row.get('remaining_deadline_sec'):.1f}s",
                    flush=True,
                )
    finally:
        snapshot["finished_utc"] = utc_now()
        snapshot["elapsed_wall_sec"] = time.perf_counter() - start_wall
        snapshot["rows_written"] = len(records)
        write_json(ROOT / "environment_snapshot.json", snapshot)
        write_json(
            args.summary,
            {
                "generated_utc": utc_now(),
                "scope": "synthetic_calibration_timing_only",
                "environment": snapshot["environment"],
                "rows": aggregate(records),
            },
        )
        args.report.write_text(render_markdown(snapshot, records, aggregate(records)), encoding="utf-8")
        for bundle in bundles.values():
            with contextlib.suppress(Exception):
                bundle.model._backward_hook.remove()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
