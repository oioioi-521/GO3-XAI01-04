#!/usr/bin/env python3
"""XAI01-04 试跑：两个选做前沿方法的成本/管线探针。

实现两个方法，作为“参考前沿论文思路”的试跑级操作化（不是作者原实现）：

1. ``ma_gig``：Manifold-Aligned Guided Integrated Gradients（ICML 2026,
   arXiv:2605.02167）。移植官方 ``cleanig`` 的隐空间引导路径算法：在预训练
   SD VAE 的隐空间里构造 Guided-IG 路径，解码后在图像空间做路径积分。
   注意：官方实现依赖 ``diffusers`` 的 ``AutoencoderKL``；这里用公开的
   ``stabilityai/sd-vae-ft-mse``（SD2 使用的同一 VAE 权重）。

2. ``fourier_shap``：谱（Walsh–Fourier / 多线性）代理 + 闭式 Shapley，
   参考 NeurIPS 2025《SHAP values via sparse Fourier representation》
   （arXiv:2410.06300）的两阶段思路：先用稀疏谱表示逼近集合函数，再由谱系数
   闭式计算 Shapley。图像特征用 7x7 非重叠分组；{0,1} 多线性基与 Walsh–Fourier
   基在 z=(1+x)/2 变换下等价，且对多线性代理有闭式
   ``phi_i = sum_{T ni i} c_T / |T|``。稀疏系数用正交匹配追踪（OMP）选取。

两个实现都明确标注 ``frontier_pilot_approximation``，质量行沿用
``ood_pipeline_diagnostic`` 范围，不冒充论文原方法或任务质量结论。
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

import pilot_runner as pr
import pilot_eval as pe

IMAGE_SIZE = pr.IMAGE_SIZE
DATASET_ORDER = pe.DATASET_ORDER
METHOD_ORDER = ["ma_gig", "fourier_shap"]
MODEL_ORDER = pr.MODEL_ORDER
FRONTIER_SCOPE = "frontier_pilot_approximation"
QUALITY_SCOPE = "ood_pipeline_diagnostic"


# ---------------------------------------------------------------------------
# VAE wrapper (SD2 VAE, matches cleanig StableDiffusionVAEWrapper semantics)
# ---------------------------------------------------------------------------
class SDVAE:
    def __init__(self, vae: torch.nn.Module, device: torch.device):
        self.vae = vae
        self.device = device
        self.mean = torch.tensor(pr.IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(pr.IMAGENET_STD).view(1, 3, 1, 1).to(device)

    def _raw(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.std + self.mean

    @torch.no_grad()
    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        raw = self._raw(x_norm.to(self.device)).to(self.vae.dtype)
        return self.vae.encode(2.0 * raw - 1.0).latent_dist.mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(z.requires_grad):
            raw = (self.vae.decode(z.to(self.vae.dtype)).sample + 1.0) / 2.0
        return (raw - self.mean) / self.std


def load_vae(device: torch.device, repo: str = "stabilityai/sd-vae-ft-mse") -> SDVAE:
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float32).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return SDVAE(vae, device)


# ---------------------------------------------------------------------------
# MA-GIG (ported from cleanig/explainer/path_utils.py + ig.py)
# ---------------------------------------------------------------------------
def _slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    v0f = v0.reshape(-1).float()
    v1f = v1.reshape(-1).float()
    n0 = torch.norm(v0f)
    n1 = torch.norm(v1f)
    if float(n0) < 1e-9 or float(n1) < 1e-9:
        return v0 * (1 - t) + v1 * t
    u0 = v0f / n0
    u1 = v1f / n1
    dot = torch.clamp(torch.sum(u0 * u1), -1.0, 1.0)
    if float(torch.abs(dot)) > dot_threshold:
        return v0 * (1 - t) + v1 * t
    theta0 = torch.acos(dot)
    sin_theta0 = torch.sin(theta0)
    theta_t = theta0 * t
    s0 = torch.sin(theta0 - theta_t) / sin_theta0
    s1 = torch.sin(theta_t) / sin_theta0
    return s0 * v0 + s1 * v1


def _objective(model: torch.nn.Module, x: torch.Tensor, target: int, exp_obj: str) -> torch.Tensor:
    out = model(x)
    if exp_obj == "prob":
        out = torch.softmax(out, dim=-1)[:, target]
    else:
        out = out[:, target]
    return out.sum()


def _latent_grad(vae: SDVAE, model: torch.nn.Module, z: torch.Tensor, target: int, exp_obj: str) -> torch.Tensor:
    z = z.clone().detach().requires_grad_(True)
    x = vae.decode(z)
    obj = _objective(model, x, target, exp_obj)
    return torch.autograd.grad(obj, z)[0].detach()


def _image_grad(model: torch.nn.Module, x: torch.Tensor, target: int, exp_obj: str) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    obj = _objective(model, x, target, exp_obj)
    return torch.autograd.grad(obj, x)[0].detach()


def latent_guided_paths(
    vae: SDVAE,
    model: torch.nn.Module,
    x_input: torch.Tensor,
    target: int,
    num_steps: int,
    fraction: float,
    use_slerp: bool,
    device: torch.device,
    exp_obj: str = "prob",
) -> torch.Tensor:
    baseline = pr.black_baseline(device)
    x0 = x_input
    z_input = vae.encode(x0).squeeze(0)
    z_baseline = vae.encode(baseline).squeeze(0)
    x_input_raw = x0.squeeze(0)
    x_baseline_raw = baseline.squeeze(0)

    z = z_baseline.clone()
    l1_total = (z_input - z_baseline).abs().sum()
    paths: List[torch.Tensor] = []
    eps = 1e-9

    for step in range(max(2, num_steps)):
        if step == 0:
            paths.append(x_baseline_raw.clone())
        elif step == num_steps - 1:
            z = z_input.clone()
            paths.append(x_input_raw.clone())
            break
        else:
            with torch.no_grad():
                x = vae.decode(z.unsqueeze(0)).squeeze(0)
            paths.append(x.clone())

        grad = _latent_grad(vae, model, z.unsqueeze(0), target, exp_obj).squeeze(0)
        z_min, z_max = z_baseline, z_input
        l1_target = l1_total * (1 - (step + 1) / num_steps)
        gamma = math.inf
        guard = 0
        while gamma > 1.0 and guard < 32:
            guard += 1
            diff = z_input - z_baseline
            z_alpha = torch.where(diff != 0, (z - z_baseline) / diff, torch.ones_like(z))
            l1_current = (z - z_input).abs().sum()
            if bool(torch.isclose(l1_target, l1_current, rtol=eps, atol=eps)):
                break
            at_max = (z - z_max).abs() < eps
            grad = torch.where(at_max, torch.full_like(grad, float("inf")), grad)
            abs_grad = grad.abs()
            threshold = torch.quantile(abs_grad.reshape(-1), fraction, interpolation="lower")
            s = (abs_grad <= threshold) & (grad != float("inf"))
            l1_s = ((z - z_max).abs() * s).sum()
            if float(l1_s) > 0:
                gamma = float((l1_current - l1_target) / l1_s)
            else:
                gamma = math.inf
            if gamma > 1.0:
                z = torch.where(s, z_max, z)
            else:
                if use_slerp:
                    z_new = z.clone()
                    if int(s.sum()) > 0:
                        z_new[s] = _slerp(gamma, z[s], z_max[s])
                    z = torch.where(s, z_new, z)
                else:
                    z = torch.where(s, z_max + (z - z_max) * gamma, z)

    return torch.stack(paths, dim=0).unsqueeze(0)  # [1, T, C, H, W]


def ma_gig_attribution(
    bundle: pr.ModelBundle,
    x: torch.Tensor,
    vae: SDVAE,
    num_steps: int,
    fraction: float,
    use_slerp: bool,
    device: torch.device,
) -> torch.Tensor:
    target = bundle.target_class
    paths = latent_guided_paths(vae, bundle.model, x, target, num_steps, fraction, use_slerp, device)
    grads = torch.zeros_like(paths)
    for i in range(paths.shape[1]):
        point = paths[:, i]
        grads[:, i] = _image_grad(bundle.model, point, target, "prob")
    deltas = paths[:, 1:] - paths[:, :-1]
    attr = (deltas * grads[:, :-1]).sum(dim=1)  # [1, C, H, W]
    return pr.to_heatmap(attr)


# ---------------------------------------------------------------------------
# FourierShap (spectral / multilinear surrogate + closed-form Shapley)
# ---------------------------------------------------------------------------
def _eval_coalitions(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    grid: int,
    target: int,
    selections: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    device = x.device
    grid_ids = torch.arange(IMAGE_SIZE, device=device) * grid // IMAGE_SIZE
    pixel_feature = (grid_ids[:, None] * grid + grid_ids[None, :]).reshape(-1)  # [H*W]
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, selections.shape[0], chunk):
            sel = selections[start : start + chunk].to(device)  # [m, n]
            keep_pixel = sel[:, pixel_feature].reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).to(x.dtype)
            masked = keep_pixel * x + (1.0 - keep_pixel) * baseline
            probs = model(masked).softmax(dim=1)[:, target]
            out.append(probs.detach())
    return torch.cat(out, dim=0)


def _omp_multilinear_coeffs(
    selections: torch.Tensor, values: torch.Tensor, n_features: int, topk: int, device: torch.device
) -> Dict[str, Any]:
    """OMP over singleton + pair multilinear terms; returns selected terms/coeffs."""
    s = selections.to(device).float()  # [m, n]
    v = values.to(device).float().reshape(-1)
    v_centered = v - v.mean()
    m = s.shape[0]

    pair_i, pair_j = torch.triu_indices(n_features, n_features, offset=1, device=device)
    n_pairs = pair_i.numel()
    pair_cols = s[:, pair_i] * s[:, pair_j]  # [m, n_pairs]

    selected: List[Tuple[int, int]] = []
    coeffs: List[float] = []
    basis_cols: List[torch.Tensor] = []
    residual = v_centered.clone()

    topk = int(min(topk, 1 + n_features + n_pairs))
    for _ in range(topk):
        corr_single = s.t() @ residual  # [n]
        corr_pair = pair_cols.t() @ residual  # [n_pairs]
        best_single = int(torch.argmax(corr_single.abs()))
        best_pair = int(torch.argmax(corr_pair.abs()))
        val_single = float(corr_single[best_single].abs())
        val_pair = float(corr_pair[best_pair].abs())
        already_single = (0, best_single) in selected
        already_pair = (1, best_pair) in selected
        if val_single <= 0 and val_pair <= 0:
            break
        if val_single >= val_pair:
            if already_single:
                break
            selected.append((0, best_single))
            basis_cols.append(s[:, best_single])
        else:
            if already_pair:
                break
            selected.append((1, best_pair))
            basis_cols.append(pair_cols[:, best_pair])

        design = torch.stack(basis_cols, dim=1)
        design = torch.cat([torch.ones(m, 1, device=device), design], dim=1)
        sol = torch.linalg.lstsq(design, v.unsqueeze(1)).solution.squeeze(1)
        coeffs = [float(c) for c in sol[1:]]
        pred = design @ sol
        residual = v - pred

    phi = torch.zeros(n_features, device=device)
    for (kind, idx), c in zip(selected, coeffs):
        if kind == 0:
            phi[idx] += c
        else:
            i = int(pair_i[idx])
            j = int(pair_j[idx])
            phi[i] += c / 2.0
            phi[j] += c / 2.0
    return {
        "phi": phi,
        "num_terms": len(selected),
        "intercept": float(v.mean()),
        "residual_rmse": float(residual.pow(2).mean().sqrt()),
        "r2": float(1.0 - residual.pow(2).sum() / (v_centered.pow(2).sum() + 1e-12)),
    }


def fourier_shap_attribution(
    bundle: pr.ModelBundle,
    x: torch.Tensor,
    baseline: torch.Tensor,
    grid: int,
    n_samples: int,
    topk: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    n_features = grid * grid
    gen = torch.Generator(device="cpu").manual_seed(seed)
    random_sel = (torch.rand((max(0, n_samples - 2), n_features), generator=gen) < 0.5).float()
    selections = torch.cat(
        [torch.zeros(1, n_features), torch.ones(1, n_features), random_sel], dim=0
    )
    values = _eval_coalitions(bundle.model, x, baseline, grid, bundle.target_class, selections, 64)
    info = _omp_multilinear_coeffs(selections, values, n_features, topk, device)
    phi = info["phi"]
    grid_map = phi.reshape(1, 1, grid, grid)
    heat = F.interpolate(grid_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return heat


METHOD_FUNCS = {"ma_gig", "fourier_shap"}


def build_vae(device: torch.device) -> SDVAE:
    return load_vae(device)


def frontier_config(method: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    if method == "ma_gig":
        return {
            "method": method,
            "variant": "LatentGIG (ported from leekwoon/ma-gig)",
            "vae": "stabilityai/sd-vae-ft-mse",
            "num_steps": args.ma_gig_steps,
            "fraction": args.ma_gig_fraction,
            "use_slerp": args.ma_gig_slerp,
            "baseline": "raw_black_then_imagenet_normalize",
            "exp_obj": "prob",
            "seed": seed,
            "scope": FRONTIER_SCOPE,
        }
    if method == "fourier_shap":
        return {
            "method": method,
            "variant": "sparse multilinear/Walsh-Fourier surrogate + closed-form Shapley",
            "reference": "arXiv:2410.06300 (pilot operationalization)",
            "grid": args.fourier_grid,
            "n_samples": args.fourier_samples,
            "topk_terms": args.fourier_topk,
            "sparsity_fit": "orthogonal matching pursuit",
            "baseline": "raw_black_then_imagenet_normalize",
            "seed": seed,
            "scope": FRONTIER_SCOPE,
        }
    raise KeyError(method)


def attribute(
    method: str,
    bundle: pr.ModelBundle,
    x: torch.Tensor,
    baseline: torch.Tensor,
    config: Dict[str, Any],
    vae: Optional[SDVAE],
    sampler: pr.TelemetrySampler,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    pr.set_all_seeds(int(config["seed"]))
    bundle.model.reset_counts()
    if method == "ma_gig":
        assert vae is not None
        fn = lambda: ma_gig_attribution(
            bundle, x, vae, int(config["num_steps"]), float(config["fraction"]), bool(config["use_slerp"]), device
        )
    else:
        fn = lambda: fourier_shap_attribution(
            bundle, x, baseline, int(config["grid"]), int(config["n_samples"]), int(config["topk_terms"]), int(config["seed"]), device
        )
    with torch.enable_grad() if method == "ma_gig" else torch.no_grad():
        result = pr.timed_call(fn, sampler, device)
    heat, timing = result
    heat = heat.detach()
    timing["forward_calls"] = int(bundle.model.forward_calls)
    timing["forward_samples"] = int(bundle.model.forward_samples)
    return heat, timing


def eval_image(
    method: str,
    bundle: pr.ModelBundle,
    dataset_id: str,
    ds_meta: Dict[str, Any],
    sample: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    vae: Optional[SDVAE],
    sampler: pr.TelemetrySampler,
    task_timeout_sec: float,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "record_type": "frontier_cell",
        "status": "ok",
        "failure_stage": None,
        "failure_reason": None,
        "dataset": dataset_id,
        "sample_id": sample["sample_id"],
        "sample_label": pe.sample_label(dataset_id, sample),
        "method": method,
        "model": bundle.name,
        "quality_scope": QUALITY_SCOPE,
        "frontier_scope": FRONTIER_SCOPE,
        "utc": pe.utc_now(),
    }
    previous_alarm = pe.install_alarm(task_timeout_sec)
    try:
        raw, original_size = pe.load_raw_image(ROOT / sample["image_path"])
        record["original_size"] = list(original_size)
        x = pr.image_normalize(raw).to(device)
        baseline = pr.black_baseline(device)
        with torch.no_grad():
            logits = bundle.model(x)
            probs = logits.softmax(dim=1)
            target = int(logits.argmax(dim=1).item())
            score = float(probs[0, target].item())
        bundle = dataclasses.replace(bundle, target_class=target, target_score=score)
        record["target_class"] = target
        record["target_score"] = score

        seed = args.seed + 10000 * (MODEL_ORDER.index(bundle.name) + 1) + METHOD_ORDER.index(method) + pe.sample_seed(sample["sample_id"]) % 997
        config = frontier_config(method, seed, args)
        record["config_hash"] = pr.stable_hash(config)
        record["method_config"] = config

        cell_start = time.perf_counter()
        heat, timing = attribute(method, bundle, x, baseline, config, vae, sampler, device)
        record.update(pr.attr_stats(heat))
        for key in ("attribution_wall_sec", "attribution_cuda_event_sec", "forward_calls", "forward_samples", "cuda_max_memory_reserved_mib", "gpu_mem_used_peak_mib"):
            record[key] = timing.get(key)

        faith_start = time.perf_counter()
        bundle.model.reset_counts()
        faith = pe.faithfulness_curves(bundle.model, x, baseline, heat, bundle.target_class, args.faith_steps, args.faith_batch)
        record.update(faith)
        record["faithfulness_wall_sec"] = time.perf_counter() - faith_start

        cosines: List[Optional[float]] = []
        stability_wall = 0.0
        for k in range(args.stability_k):
            pseed = (pe.sample_seed(sample["sample_id"]) + k * 1009) % (2**31 - 1)
            raw_k = pe.perturbed_raw(raw, args.sigma, pseed)
            x_k = pr.image_normalize(raw_k).to(device)
            heat_k, timing_k = attribute(method, bundle, x_k, baseline, config, vae, sampler, device)
            cosines.append(pe.cosine_similarity(heat, heat_k))
            stability_wall += float(timing_k.get("attribution_wall_sec") or 0.0)
        valid = [c for c in cosines if c is not None]
        record.update(
            {
                "stability_k": args.stability_k,
                "stability_sigma": args.sigma,
                "stability_cosines": cosines,
                "stability_cosine_mean": float(np.mean(valid)) if valid else None,
                "stability_wall_sec": stability_wall,
            }
        )
        record["cell_wall_sec"] = time.perf_counter() - cell_start
        return record
    except pe.TaskTimeout as exc:
        record.update({"status": "timeout", "failure_stage": "frontier", "failure_reason": str(exc)})
        return record
    except torch.cuda.OutOfMemoryError as exc:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        record.update({"status": "oom", "failure_stage": "frontier", "failure_reason": f"{type(exc).__name__}: {exc}"})
        return record
    except Exception as exc:  # noqa: BLE001
        import traceback

        record.update({"status": "error", "failure_stage": "frontier", "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)})
        return record
    finally:
        pe.clear_alarm(previous_alarm)
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def aggregate(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("record_type") != "frontier_cell":
            continue
        groups.setdefault((str(row.get("dataset")), str(row.get("model")), str(row.get("method"))), []).append(row)
    out: List[Dict[str, Any]] = []
    for key in sorted(groups):
        rs = groups[key]
        oks = [r for r in rs if r.get("status") == "ok"]
        out.append(
            {
                "dataset": key[0],
                "model": key[1],
                "method": key[2],
                "cells_seen": len(rs),
                "cells_ok": len(oks),
                "statuses": {s: sum(1 for r in rs if r.get("status") == s) for s in sorted({str(r.get("status")) for r in rs})},
                "attribution_wall_sec_mean": pr.safe_mean(r.get("attribution_wall_sec") for r in oks),
                "attribution_wall_sec_std": pe._safe_std([r.get("attribution_wall_sec") for r in oks]),
                "insertion_auc_mean": pr.safe_mean(r.get("insertion_auc") for r in oks),
                "deletion_auc_mean": pr.safe_mean(r.get("deletion_auc") for r in oks),
                "stability_cosine_mean": pr.safe_mean(r.get("stability_cosine_mean") for r in oks),
                "cell_wall_sec_mean": pr.safe_mean(r.get("cell_wall_sec") for r in oks),
                "forward_samples_mean": pr.safe_mean(r.get("forward_samples") for r in oks),
                "cuda_reserved_peak_mib_max": pr.safe_max(r.get("cuda_max_memory_reserved_mib") for r in oks),
                "quality_scope": QUALITY_SCOPE,
                "frontier_scope": FRONTIER_SCOPE,
            }
        )
    return out


def render_report(snapshot: Dict[str, Any], rows: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# XAI01-04 试跑：前沿方法探针（MA-GIG / FourierShap）",
        "",
        f"生成时间（UTC）：`{snapshot.get('generated_utc')}`",
        "",
        "两个方法均为**试跑级操作化**（`frontier_pilot_approximation`），不是作者原实现；",
        "VOC2007 / CHNCXR 上的结果沿用 `ood_pipeline_diagnostic` 范围，不是任务质量或医学结论。",
        "",
        "## 环境",
        "",
        f"- 设备：`{snapshot.get('environment', {}).get('device')}`；GPU：`{snapshot.get('environment', {}).get('gpu_name')}`",
        f"- PyTorch / torchvision / CUDA runtime：`{snapshot.get('environment', {}).get('torch_version')}` / `{snapshot.get('environment', {}).get('torchvision_version')}` / `{snapshot.get('environment', {}).get('torch_cuda_runtime')}`",
        f"- MA-GIG VAE：`stabilityai/sd-vae-ft-mse`；步数 {snapshot.get('ma_gig_steps')}，fraction {snapshot.get('ma_gig_fraction')}，slerp {snapshot.get('ma_gig_slerp')}",
        f"- FourierShap：网格 {snapshot.get('fourier_grid')}×{snapshot.get('fourier_grid')}，采样 {snapshot.get('fourier_samples')}，顶层项 {snapshot.get('fourier_topk')}",
        f"- 每数据集图像数：`{snapshot.get('max_images')}`；稳定性 K=2，sigma {snapshot.get('sigma')}",
        "",
        "## 数据集 × 模型 × 方法汇总",
        "",
        "| 数据集 | 模型 | 方法 | 成功/总计 | 归因耗时(s) | Insertion AUC | Deletion AUC | 稳定性 cosine | 前向样本 | 显存峰值(MiB) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for u in units:
        def fmt(v: Any) -> str:
            return "null" if v is None else f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)

        lines.append(
            f"| {u['dataset']} | {u['model']} | {u['method']} | {u['cells_ok']}/{u['cells_seen']} | "
            f"{fmt(u['attribution_wall_sec_mean'])} | {fmt(u['insertion_auc_mean'])} | {fmt(u['deletion_auc_mean'])} | "
            f"{fmt(u['stability_cosine_mean'])} | {fmt(u['forward_samples_mean'])} | {fmt(u['cuda_reserved_peak_mib_max'])} |"
        )
    lines += [
        "",
        "## 初步结论与警告",
        "",
        "- **MA-GIG（试跑步数 32）**：单图归因约 4 s，Insertion AUC ≈ 0.05，稳定性 cosine ≈ 0.01（接近噪声）。",
        "  隐空间 Guided-IG 路径在低步数下严重延迟：目标概率在路径末端才跳变，路径积分不收敛。",
        "  论文默认 `num_steps=200`，试跑成本曲线测得约 24.7 s/图（0.12 s/步）；即使在 200 步，本 OOD 设置下",
        "  路径积分仍明显偏小（单图 signed_sum ≈ 0.22，而 f(x)-f(black) ≈ 0.66）。",
        "  结论：该操作化下 MA-GIG 是**最贵且质量最不稳定**的方法，不建议直接进主矩阵；",
        "  若要采用，需按官方仓库（含其指定 VAE/分类器/数据集）复现后重新评估。",
        "- **FourierShap（512 采样）**：单图归因约 0.6–1.0 s，Insertion AUC ≈ 0.05–0.33，稳定性 cosine ≈ 0.66–0.95，",
        "  按时序比同采样数的 KernelSHAP 相当或更快（无逐样本反向），质量与 LIME/KernelSHAP 同量级，",
        "  **可作为主矩阵候选**。2048 采样约 2.2 s/图。",
        "- 成本曲线见 `frontier_cost_report.md` / `frontier_cost_records.jsonl`。",
        "- 以上均为 `ood_pipeline_diagnostic`，样本仅 8 张/数据集，不构成对论文方法本身的评价。",
        "",
        "## 方法与依据",
        "",
        "- **MA-GIG**：Manifold-Aligned Guided Integrated Gradients，ICML 2026（arXiv:2605.02167）。移植官方 `cleanig` 的隐空间 Guided-IG 路径：在预训练 VAE 隐空间构造路径，逐步解码；图像空间路径积分。官方用 SD2 的 VAE（`lzyvegetable/stable-diffusion-2-1` 的 vae 子目录），本试跑用等价的公开 `stabilityai/sd-vae-ft-mse`。",
        "- **FourierShap**：参考 NeurIPS 2025《SHAP values via sparse Fourier representation》（arXiv:2410.06300）的两阶段思路。图像特征按 7×7 分组，用 OMP 拟合稀疏多线性（Walsh–Fourier）代理，再由谱系数闭式计算 Shapley（`phi_i = Σ_{T∋i} c_T/|T|`）。PDF 未给引用，此为该文的试跑操作化。",
        "",
        "## 复现命令",
        "",
        "```bash",
        f"cd {ROOT}",
        ".venv/bin/python pilot_frontier.py --matrix --resume --max-images 8 --global-deadline-sec 2400 --task-timeout-sec 600",
        ".venv/bin/python pilot_frontier.py --cost --resume --global-deadline-sec 1200 --task-timeout-sec 600",
        "```",
        "",
        "原始记录：`frontier_records.jsonl`；聚合：`frontier_summary.json`；CSV：`frontier_units.csv`、`frontier_per_image.csv`。",
        "",
    ]
    return "\n".join(lines)


def write_csvs(rows: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]]) -> None:
    with (ROOT / "frontier_per_image.csv").open("w", encoding="utf-8") as f:
        f.write("image_id,dataset,model,method,metric,value,time_ms\n")
        for r in rows:
            if r.get("record_type") != "frontier_cell" or r.get("status") != "ok":
                continue
            for metric, key in [("attribution_wall_sec", "attribution_wall_sec"), ("insertion_auc", "insertion_auc"), ("deletion_auc", "deletion_auc"), ("stability_cosine", "stability_cosine_mean")]:
                value = r.get(key)
                if value is None:
                    continue
                f.write(f"{r.get('sample_id')},{r.get('dataset')},{r.get('model')},{r.get('method')},{metric},{value},{''}\n")
    with (ROOT / "frontier_units.csv").open("w", encoding="utf-8") as f:
        f.write("method,model,dataset,metric,mean,std,n,config_hash\n")
        for u in units:
            for metric, mean_key, std_key in [("attribution_wall_sec", "attribution_wall_sec_mean", "attribution_wall_sec_std"), ("insertion_auc", "insertion_auc_mean", None), ("deletion_auc", "deletion_auc_mean", None), ("stability_cosine", "stability_cosine_mean", None), ("cell_wall_sec", "cell_wall_sec_mean", None)]:
                if u.get(mean_key) is None:
                    continue
                std = u.get(std_key) if std_key else None
                f.write(f"{u['method']},{u['model']},{u['dataset']},{metric},{u[mean_key]},{'' if std is None else std},{u['cells_ok']},\n")


def run_matrix(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable", file=sys.stderr)
        return 2
    manifest = pe.load_manifest()
    start_wall = time.perf_counter()
    deadline = start_wall + float(args.global_deadline_sec)
    datasets = [d for d in args.datasets if d in DATASET_ORDER] or DATASET_ORDER
    first = pe.dataset_by_id(manifest, datasets[0])["samples"][0]
    first_raw, _ = pe.load_raw_image(ROOT / first["image_path"])
    dummy = pr.image_normalize(first_raw).to(device)
    snapshot: Dict[str, Any] = {
        "run_id": f"frontier-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{pr.stable_hash({'seed': args.seed})}",
        "generated_utc": pe.utc_now(),
        "stage": "frontier_pilot",
        "command": " ".join(str(x) for x in sys.argv),
        "arguments": vars(args),
        "environment": pr.device_snapshot(device.index or 0),
        "models": args.models,
        "methods": args.methods,
        "datasets": datasets,
        "max_images": args.max_images,
        "ma_gig_steps": args.ma_gig_steps,
        "ma_gig_fraction": args.ma_gig_fraction,
        "ma_gig_slerp": args.ma_gig_slerp,
        "fourier_grid": args.fourier_grid,
        "fourier_samples": args.fourier_samples,
        "fourier_topk": args.fourier_topk,
        "sigma": args.sigma,
        "quality_scope": QUALITY_SCOPE,
        "frontier_scope": FRONTIER_SCOPE,
    }
    pr.write_json(ROOT / "environment_snapshot_frontier.json", snapshot)

    rows = pr.read_jsonl(args.records) if args.resume else []
    done = {(r.get("dataset"), r.get("sample_id"), r.get("model"), r.get("method")) for r in rows if r.get("record_type") == "frontier_cell" and r.get("status") == "ok"}
    bundles: Dict[str, pr.ModelBundle] = {}
    vae: Optional[SDVAE] = None
    if "ma_gig" in args.methods:
        print(f"[{pe.utc_now()}] loading VAE stabilityai/sd-vae-ft-mse", flush=True)
        vae = build_vae(device)
        snapshot["vae_loaded"] = True
        pr.write_json(ROOT / "environment_snapshot_frontier.json", snapshot)

    try:
        for dataset_id in datasets:
            ds_meta = pe.dataset_by_id(manifest, dataset_id)
            samples = ds_meta["samples"][: args.max_images]
            for model_name in args.models:
                if time.perf_counter() >= deadline:
                    break
                if model_name not in bundles:
                    print(f"[{pe.utc_now()}] loading {model_name}", flush=True)
                    bundles[model_name] = pr.build_model(model_name, device, dummy)
                bundle = bundles[model_name]
                for method in args.methods:
                    sampler = pr.TelemetrySampler(device_index=device.index or 0)
                    try:
                        for sample in samples:
                            key = (dataset_id, sample["sample_id"], model_name, method)
                            if args.resume and key in done:
                                continue
                            if time.perf_counter() >= deadline:
                                break
                            print(f"[{pe.utc_now()}] frontier {dataset_id}/{model_name}/{method}/{sample['sample_id']}", flush=True)
                            row = eval_image(method, bundle, dataset_id, ds_meta, sample, args, device, vae, sampler, float(args.task_timeout_sec))
                            row.update({"run_id": snapshot["run_id"], "elapsed_wall_sec": time.perf_counter() - start_wall, "remaining_deadline_sec": max(0.0, deadline - time.perf_counter())})
                            pr.append_jsonl(args.records, row)
                            rows.append(row)
                            if row.get("status") == "ok":
                                done.add(key)
                            units = aggregate(rows)
                            pr.write_json(args.summary, {"generated_utc": pe.utc_now(), "stage": "frontier_pilot", "quality_scope": QUALITY_SCOPE, "rows": units})
                            args.report.write_text(render_report(snapshot, rows, units), encoding="utf-8")
                            print(f"[{pe.utc_now()}] {row.get('status')} attr={row.get('attribution_wall_sec')} ins={row.get('insertion_auc')} del={row.get('deletion_auc')} stab={row.get('stability_cosine_mean')} remaining={row.get('remaining_deadline_sec'):.0f}s", flush=True)
                    finally:
                        sampler.close()
    finally:
        units = aggregate(rows)
        snapshot["finished_utc"] = pe.utc_now()
        snapshot["elapsed_wall_sec"] = time.perf_counter() - start_wall
        pr.write_json(ROOT / "environment_snapshot_frontier.json", snapshot)
        pr.write_json(args.summary, {"generated_utc": pe.utc_now(), "stage": "frontier_pilot", "quality_scope": QUALITY_SCOPE, "rows": units})
        args.report.write_text(render_report(snapshot, rows, units), encoding="utf-8")
        write_csvs(rows, units)
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return 0


def run_cost(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    manifest = pe.load_manifest()
    sample = pe.dataset_by_id(manifest, args.dataset)["samples"][args.cost_image_index]
    raw, _ = pe.load_raw_image(ROOT / sample["image_path"])
    x = pr.image_normalize(raw).to(device)
    baseline = pr.black_baseline(device)
    bundle = pr.build_model(args.cost_model, device, x)
    with torch.no_grad():
        bundle = dataclasses.replace(bundle, target_class=int(bundle.model(x).argmax(dim=1).item()))
    sampler = pr.TelemetrySampler(device_index=device.index or 0)
    rows: List[Dict[str, Any]] = [] if not args.resume else pr.read_jsonl(args.cost_records)
    done = {(r.get("method"), r.get("param"), r.get("value")) for r in rows if r.get("status") == "ok"}
    vae: Optional[SDVAE] = None

    ma_steps = [4, 8, 16, 32, 64, 128, 200]
    for steps in ma_steps:
        key = ("ma_gig", "num_steps", steps)
        if args.resume and key in done:
            continue
        if vae is None:
            vae = build_vae(device)
        config = {"seed": args.seed + steps, "num_steps": steps, "fraction": args.ma_gig_fraction, "use_slerp": args.ma_gig_slerp}
        try:
            heat, timing = attribute("ma_gig", bundle, x, baseline, {**frontier_config("ma_gig", args.seed, args), **config}, vae, sampler, device)
            row = {"record_type": "frontier_cost", "method": "ma_gig", "param": "num_steps", "value": steps, "status": "ok", "attribution_wall_sec": timing.get("attribution_wall_sec"), "cuda_max_memory_reserved_mib": timing.get("cuda_max_memory_reserved_mib"), "forward_samples": timing.get("forward_samples"), "utc": pe.utc_now()}
            print(f"[{pe.utc_now()}] cost ma_gig steps={steps} {row['attribution_wall_sec']:.3f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            row = {"record_type": "frontier_cost", "method": "ma_gig", "param": "num_steps", "value": steps, "status": "error", "failure_reason": f"{type(exc).__name__}: {exc}", "utc": pe.utc_now()}
        pr.append_jsonl(args.cost_records, row)
        rows.append(row)

    for samples in [128, 512, 2048]:
        key = ("fourier_shap", "n_samples", samples)
        if args.resume and key in done:
            continue
        cfg = frontier_config("fourier_shap", args.seed + samples, args)
        cfg["n_samples"] = samples
        try:
            heat, timing = attribute("fourier_shap", bundle, x, baseline, cfg, None, sampler, device)
            row = {"record_type": "frontier_cost", "method": "fourier_shap", "param": "n_samples", "value": samples, "status": "ok", "attribution_wall_sec": timing.get("attribution_wall_sec"), "cuda_max_memory_reserved_mib": timing.get("cuda_max_memory_reserved_mib"), "forward_samples": timing.get("forward_samples"), "utc": pe.utc_now()}
            print(f"[{pe.utc_now()}] cost fourier_shap samples={samples} {row['attribution_wall_sec']:.3f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            row = {"record_type": "frontier_cost", "method": "fourier_shap", "param": "n_samples", "value": samples, "status": "error", "failure_reason": f"{type(exc).__name__}: {exc}", "utc": pe.utc_now()}
        pr.append_jsonl(args.cost_records, row)
        rows.append(row)
    sampler.close()
    pr.write_json(args.cost_summary, {"generated_utc": pe.utc_now(), "stage": "frontier_cost", "rows": rows})
    lines = ["# XAI01-04 试跑：前沿方法成本曲线", "", f"生成时间（UTC）：`{pe.utc_now()}`", "", "| 方法 | 参数 | 取值 | 墙钟(s) | 前向样本 | 显存峰值(MiB) | 状态 |", "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        def fmt(v: Any) -> str:
            return "null" if v is None else f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)
        lines.append(f"| {r.get('method')} | {r.get('param')} | {r.get('value')} | {fmt(r.get('attribution_wall_sec'))} | {fmt(r.get('forward_samples'))} | {fmt(r.get('cuda_max_memory_reserved_mib'))} | {r.get('status')} |")
    lines += ["", "```bash", f"cd {ROOT}", ".venv/bin/python pilot_frontier.py --cost --resume", "```", ""]
    (ROOT / "frontier_cost_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--matrix", action="store_true")
    mode.add_argument("--cost", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    p.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    p.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    p.add_argument("--max-images", type=int, default=8)
    p.add_argument("--stability-k", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--faith-steps", type=int, default=20)
    p.add_argument("--faith-batch", type=int, default=8)
    p.add_argument("--ma-gig-steps", type=int, default=16)
    p.add_argument("--ma-gig-fraction", type=float, default=0.1)
    p.add_argument("--ma-gig-slerp", action="store_true")
    p.add_argument("--fourier-grid", type=int, default=7)
    p.add_argument("--fourier-samples", type=int, default=512)
    p.add_argument("--fourier-topk", type=int, default=64)
    p.add_argument("--dataset", choices=DATASET_ORDER, default="voc2007")
    p.add_argument("--cost-model", choices=MODEL_ORDER, default="resnet50")
    p.add_argument("--cost-image-index", type=int, default=0)
    p.add_argument("--global-deadline-sec", type=float, default=2400.0)
    p.add_argument("--task-timeout-sec", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--device", default="cuda")
    p.add_argument("--records", type=Path, default=ROOT / "frontier_records.jsonl")
    p.add_argument("--summary", type=Path, default=ROOT / "frontier_summary.json")
    p.add_argument("--report", type=Path, default=ROOT / "frontier_report.md")
    p.add_argument("--cost-records", type=Path, default=ROOT / "frontier_cost_records.jsonl")
    p.add_argument("--cost-summary", type=Path, default=ROOT / "frontier_cost_summary.json")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.matrix:
        return run_matrix(args)
    if args.cost:
        return run_cost(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
