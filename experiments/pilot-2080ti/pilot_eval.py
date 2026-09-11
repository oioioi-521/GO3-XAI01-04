#!/usr/bin/env python3
"""XAI01-04 pilot stages P2/P3/P4 on real VOC2007 and CHNCXR inputs.

P2 (``--evaluate``) runs the small end-to-end matrix: every method x model x
dataset cell computes one cached attribution per image, Insertion/Deletion AUC
from that cached heatmap, and a K=2 input-perturbation stability cosine.  P3
(``--cost-curve``) sweeps the dominant sampling parameters on fixed images.
P4 (``--recheck``) measures Monte-Carlo seed variance for the stochastic
methods.

This module imports the already-validated helpers from ``pilot_runner`` instead
of re-implementing model loading, attribution, timing, telemetry, or JSONL
writing.  ImageNet-pretrained heads are out-of-domain for both pilot datasets,
so every quality row is explicitly scoped ``ood_pipeline_diagnostic``; nothing
here is a medical or VOC-task faithfulness claim.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
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
from PIL import Image

import pilot_runner as pr

GRAD_METHODS = {"gradcam", "ig"}
METHOD_ORDER = pr.METHOD_ORDER
MODEL_ORDER = pr.MODEL_ORDER
IMAGE_SIZE = pr.IMAGE_SIZE
ROOT_DIR = ROOT

DATASET_ORDER = ["voc2007", "chncxr_shenzhen"]


class TaskTimeout(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_manifest() -> Dict[str, Any]:
    with (ROOT / "data_manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def dataset_by_id(manifest: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    for ds in manifest["datasets"]:
        if ds["dataset_id"] == dataset_id:
            return ds
    raise KeyError(dataset_id)


def sample_label(dataset_id: str, sample: Dict[str, Any]) -> Optional[str]:
    if dataset_id == "chncxr_shenzhen":
        return sample.get("filename_label") or (sample.get("clinical_reading") or {}).get("abnormality")
    cats = sample.get("categories")
    if cats:
        return "|".join(sorted(set(cats)))
    return None


def load_raw_image(path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
    with Image.open(path) as img:
        original_size = (int(img.width), int(img.height))
        rgb = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    raw = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return raw, original_size


def sample_seed(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)


def perturbed_raw(raw: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(raw.shape, generator=gen, dtype=raw.dtype)
    return (raw.cpu() + sigma * noise).clamp(0.0, 1.0)


def trapezoid(y: Sequence[float], x: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=np.float64)
    xx = np.asarray(x, dtype=np.float64)
    if yy.size < 2:
        return float("nan")
    return float(np.sum((yy[:-1] + yy[1:]) * (xx[1:] - xx[:-1]) / 2.0))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    af = a.detach().float().reshape(-1)
    bf = b.detach().float().reshape(-1)
    if af.numel() != bf.numel():
        return None
    if not (torch.isfinite(af).all() and torch.isfinite(bf).all()):
        return None
    na = af.norm()
    nb = bf.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return None
    return float(torch.dot(af, bf).item() / (float(na) * float(nb)))


def attribute(
    bundle: pr.ModelBundle,
    x: torch.Tensor,
    baseline: torch.Tensor,
    method: str,
    config: Dict[str, Any],
    sampler: pr.TelemetrySampler,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    pr.set_all_seeds(int(config["seed"]))
    bundle.model.reset_counts()
    context = torch.enable_grad() if method in GRAD_METHODS else torch.no_grad()
    with context:
        heat, timing = pr.timed_call(
            lambda: pr.METHOD_FUNCS[method](bundle, x, baseline, config), sampler, device
        )
    heat = heat.detach()
    timing["forward_calls"] = int(bundle.model.forward_calls)
    timing["forward_samples"] = int(bundle.model.forward_samples)
    timing["backward_hook_calls"] = int(bundle.model.backward_hook_calls)
    return heat, timing


@torch.no_grad()
def faithfulness_curves(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    heat: torch.Tensor,
    target_class: int,
    steps: int,
    chunk: int,
) -> Dict[str, Any]:
    n_pixels = IMAGE_SIZE * IMAGE_SIZE
    order = torch.argsort(heat.detach().reshape(-1).abs(), descending=True)
    fractions = torch.linspace(0.0, 1.0, steps + 1, device=x.device)
    ks = torch.round(fractions * n_pixels).long().clamp(0, n_pixels).tolist()
    xf = x.reshape(1, 3, n_pixels)
    bf = baseline.reshape(1, 3, n_pixels)

    def curve(mode: str) -> Tuple[List[float], int]:
        probs: List[float] = []
        forward_samples = 0
        for start in range(0, len(ks), chunk):
            batch_items: List[torch.Tensor] = []
            for k in ks[start : start + chunk]:
                idx = order[:k]
                if mode == "insertion":
                    masked = bf.clone()
                    masked[:, :, idx] = xf[:, :, idx]
                else:
                    masked = xf.clone()
                    masked[:, :, idx] = bf[:, :, idx]
                batch_items.append(masked.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE))
            batch = torch.cat(batch_items, dim=0).to(x.device)
            logits = model(batch)
            forward_samples += int(batch.shape[0])
            batch_probs = logits.softmax(dim=1)[:, target_class]
            probs.extend(float(v) for v in batch_probs.detach().cpu().tolist())
        return probs, forward_samples

    insertion, ins_samples = curve("insertion")
    deletion, del_samples = curve("deletion")
    fractions_list = [float(v) for v in fractions.detach().cpu().tolist()]
    return {
        "faithfulness_steps": steps,
        "faithfulness_nodes": steps + 1,
        "faithfulness_fractions": fractions_list,
        "insertion_curve": insertion,
        "deletion_curve": deletion,
        "insertion_auc": trapezoid(insertion, fractions_list),
        "deletion_auc": trapezoid(deletion, fractions_list),
        "faithfulness_forward_samples": ins_samples + del_samples,
        "faithfulness_ranking_rule": "absolute_signed_summed_heatmap_descending",
        "faithfulness_replacement": "raw_black_then_imagenet_normalize",
    }


def device_from_args(args: argparse.Namespace) -> Tuple[torch.device, Optional[int]]:
    if args.device == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            print(
                "CUDA is unavailable in this process; pass --allow-cpu only for a non-GPU dry run.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return torch.device("cpu"), None
    device = torch.device(args.device)
    return device, (device.index or 0) if device.type == "cuda" else None


def install_alarm(seconds: float) -> Any:
    if seconds <= 0:
        return None
    import signal

    previous = signal.getsignal(signal.SIGALRM)

    def handler(_signum: int, _frame: Any) -> None:
        raise TaskTimeout(f"task exceeded {seconds:.1f}s timeout")

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous


def clear_alarm(previous: Any) -> None:
    import signal

    with contextlib.suppress(Exception):
        signal.setitimer(signal.ITIMER_REAL, 0)
    if previous is not None:
        signal.signal(signal.SIGALRM, previous)


def eval_image(
    bundle: pr.ModelBundle,
    dataset_id: str,
    ds_meta: Dict[str, Any],
    sample: Dict[str, Any],
    method: str,
    args: argparse.Namespace,
    device: torch.device,
    sampler: pr.TelemetrySampler,
    global_seed: int,
    task_timeout_sec: float,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "record_type": "evaluation_cell",
        "status": "ok",
        "failure_stage": None,
        "failure_reason": None,
        "dataset": dataset_id,
        "dataset_version": ds_meta.get("version"),
        "dataset_split": ds_meta.get("split"),
        "quality_scope": "ood_pipeline_diagnostic",
        "sample_id": sample["sample_id"],
        "sample_label": sample_label(dataset_id, sample),
        "image_path": sample.get("image_path"),
        "image_sha256": sample.get("sha256"),
        "method": method,
        "model": bundle.name,
        "utc": utc_now(),
    }
    previous_alarm = install_alarm(task_timeout_sec)
    try:
        raw, original_size = load_raw_image(ROOT / sample["image_path"])
        record["original_size"] = list(original_size)
        x = pr.image_normalize(raw).to(device)
        baseline = pr.black_baseline(device)

        with torch.no_grad():
            logits = bundle.model(x)
            probs = logits.softmax(dim=1)
            target_class = int(logits.argmax(dim=1).item())
            predicted_class = target_class
            target_score = float(probs[0, target_class].item())
        bundle = dataclasses.replace(bundle, target_class=target_class, target_score=target_score)
        record["target_class"] = target_class
        record["predicted_class"] = predicted_class
        record["target_score"] = target_score

        seed = global_seed + 10000 * (MODEL_ORDER.index(bundle.name) + 1) + METHOD_ORDER.index(method) + sample_seed(sample["sample_id"]) % 997
        config = pr.method_config(method, args.internal_batch_size, seed, overrides={"stability_sigma": args.sigma})
        record["config_hash"] = pr.stable_hash(config)
        record["method_config"] = config

        cell_start = time.perf_counter()
        heat, timing = attribute(bundle, x, baseline, method, config, sampler, device)
        stats = pr.attr_stats(heat)
        record.update(stats)
        record.update(
            {
                "attribution_wall_sec": timing.get("attribution_wall_sec"),
                "attribution_cuda_event_sec": timing.get("attribution_cuda_event_sec"),
                "attribution_forward_calls": timing.get("forward_calls"),
                "attribution_forward_samples": timing.get("forward_samples"),
                "attribution_backward_hook_calls": timing.get("backward_hook_calls"),
                "cuda_max_memory_allocated_mib": timing.get("cuda_max_memory_allocated_mib"),
                "cuda_max_memory_reserved_mib": timing.get("cuda_max_memory_reserved_mib"),
                "cpu_rss_peak_mib": timing.get("cpu_rss_peak_mib"),
                "gpu_mem_used_peak_mib": timing.get("gpu_mem_used_peak_mib"),
            }
        )

        faith_start = time.perf_counter()
        bundle.model.reset_counts()
        faith = faithfulness_curves(
            bundle.model, x, baseline, heat, bundle.target_class, args.faith_steps, args.faith_batch
        )
        faith_wall = time.perf_counter() - faith_start
        record.update(faith)
        record["faithfulness_wall_sec"] = faith_wall

        cosines: List[Optional[float]] = []
        stability_wall = 0.0
        stability_invalid = 0
        stability_seeds: List[int] = []
        for k in range(args.stability_k):
            pseed = (sample_seed(sample["sample_id"]) + k * 1009) % (2**31 - 1)
            stability_seeds.append(pseed)
            raw_k = perturbed_raw(raw, args.sigma, pseed)
            raw_k_sha = hashlib.sha256(raw_k.numpy().tobytes()).hexdigest()
            x_k = pr.image_normalize(raw_k).to(device)
            heat_k, timing_k = attribute(bundle, x_k, baseline, method, config, sampler, device)
            cos = cosine_similarity(heat, heat_k)
            if cos is None:
                stability_invalid += 1
            cosines.append(cos)
            stability_wall += float(timing_k.get("attribution_wall_sec") or 0.0)
            record[f"stability_perturbed_sha256_k{k}"] = raw_k_sha
        valid_cosines = [c for c in cosines if c is not None]
        record.update(
            {
                "stability_k": args.stability_k,
                "stability_sigma": args.sigma,
                "stability_seeds": stability_seeds,
                "stability_cosines": cosines,
                "stability_cosine_mean": float(np.mean(valid_cosines)) if valid_cosines else None,
                "stability_invalid_count": stability_invalid,
                "stability_wall_sec": stability_wall,
            }
        )
        record["cell_wall_sec"] = time.perf_counter() - cell_start
        record["efficiency_metric_definition"] = "attribution_wall_sec is synchronized single-image attribution time"
        return record
    except TaskTimeout as exc:
        record.update({"status": "timeout", "failure_stage": "evaluation", "failure_reason": str(exc)})
        return record
    except torch.cuda.OutOfMemoryError as exc:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        record.update(
            {"status": "oom", "failure_stage": "evaluation", "failure_reason": f"{type(exc).__name__}: {exc}"}
        )
        return record
    except Exception as exc:  # noqa: BLE001 - record the failure instead of aborting the batch
        import traceback

        record.update(
            {
                "status": "error",
                "failure_stage": "evaluation",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }
        )
        return record
    finally:
        clear_alarm(previous_alarm)
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def _safe_std(values: Sequence[Any]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else (0.0 if len(xs) == 1 else None)


def aggregate_eval(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("record_type") != "evaluation_cell":
            continue
        key = (str(row.get("dataset")), str(row.get("model")), str(row.get("method")))
        groups.setdefault(key, []).append(row)
    result: List[Dict[str, Any]] = []
    for key in sorted(
        groups,
        key=lambda k: (
            DATASET_ORDER.index(k[0]) if k[0] in DATASET_ORDER else 99,
            MODEL_ORDER.index(k[1]) if k[1] in MODEL_ORDER else 99,
            METHOD_ORDER.index(k[2]) if k[2] in METHOD_ORDER else 99,
        ),
    ):
        rs = groups[key]
        oks = [r for r in rs if r.get("status") == "ok"]
        result.append(
            {
                "dataset": key[0],
                "model": key[1],
                "method": key[2],
                "cells_seen": len(rs),
                "cells_ok": len(oks),
                "statuses": {s: sum(1 for r in rs if r.get("status") == s) for s in sorted({str(r.get("status")) for r in rs})},
                "attribution_wall_sec_mean": pr.safe_mean(r.get("attribution_wall_sec") for r in oks),
                "attribution_wall_sec_std": _safe_std([r.get("attribution_wall_sec") for r in oks]),
                "attribution_wall_sec_median": float(np.median([r["attribution_wall_sec"] for r in oks]))
                if oks and all(r.get("attribution_wall_sec") is not None for r in oks)
                else None,
                "insertion_auc_mean": pr.safe_mean(r.get("insertion_auc") for r in oks),
                "insertion_auc_std": _safe_std([r.get("insertion_auc") for r in oks]),
                "deletion_auc_mean": pr.safe_mean(r.get("deletion_auc") for r in oks),
                "deletion_auc_std": _safe_std([r.get("deletion_auc") for r in oks]),
                "stability_cosine_mean": pr.safe_mean(r.get("stability_cosine_mean") for r in oks),
                "stability_cosine_std": _safe_std([r.get("stability_cosine_mean") for r in oks]),
                "cell_wall_sec_mean": pr.safe_mean(r.get("cell_wall_sec") for r in oks),
                "cell_wall_sec_std": _safe_std([r.get("cell_wall_sec") for r in oks]),
                "stability_invalid_cells": sum(int(r.get("stability_invalid_count") or 0) for r in oks),
                "attribution_forward_samples_mean": pr.safe_mean(r.get("attribution_forward_samples") for r in oks),
                "faithfulness_forward_samples_mean": pr.safe_mean(r.get("faithfulness_forward_samples") for r in oks),
                "cuda_reserved_peak_mib_max": pr.safe_max(r.get("cuda_max_memory_reserved_mib") for r in oks),
                "quality_scope": "ood_pipeline_diagnostic",
            }
        )
    return result


def render_eval_markdown(snapshot: Dict[str, Any], rows: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# XAI01-04 试跑 P2：小规模端到端评价",
        "",
        f"生成时间（UTC）：`{snapshot.get('generated_utc')}`",
        "",
        "VOC2007 与 CHNCXR 上的 ImageNet 预训练分类头均属域外，因此这些行是",
        "`ood_pipeline_diagnostic`（域外管线诊断）的测量，不是任务忠实性结论或医学结论。",
        "",
        "## 环境",
        "",
        f"- 设备：`{snapshot.get('environment', {}).get('device')}`；GPU：`{snapshot.get('environment', {}).get('gpu_name')}`",
        f"- PyTorch / torchvision / CUDA runtime：`{snapshot.get('environment', {}).get('torch_version')}` / `{snapshot.get('environment', {}).get('torchvision_version')}` / `{snapshot.get('environment', {}).get('torch_cuda_runtime')}`",
        f"- 每数据集图像数：`{snapshot.get('max_images')}`；稳定性 K：`{snapshot.get('stability_k')}`；噪声 sigma：`{snapshot.get('sigma')}`",
        f"- 记录单元数：`{len(rows)}`",
        "",
        "## 数据集 × 模型 × 方法汇总",
        "",
        "| 数据集 | 模型 | 方法 | 成功/总计 | 归因耗时均值(s) | Insertion AUC | Deletion AUC | 稳定性 cosine | 归因前向样本数 | 状态 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        def fmt(v: Any) -> str:
            return "null" if v is None else f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)

        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['method']} | {row['cells_ok']}/{row['cells_seen']} | "
            f"{fmt(row['attribution_wall_sec_mean'])} | {fmt(row['insertion_auc_mean'])} | {fmt(row['deletion_auc_mean'])} | "
            f"{fmt(row['stability_cosine_mean'])} | {fmt(row['attribution_forward_samples_mean'])} | {row['statuses']} |"
        )
    lines += [
        "",
        "## 协议",
        "",
        "- 归因目标取模型在原图上的 ImageNet top-1 预测类别，逐单元记录。",
        "- 忠实性使用缓存热图：20 步 / 21 个节点，梯形积分 AUC，替换基线为归一化黑图。",
        "- 稳定性为 K=2 高斯噪声（原始 [0,1] 空间下的 sigma）重新归因，同一图像共享扰动；cosine 在带符号热图上计算。",
        "- 随机方法在原图与扰动图上复用同一归因种子，使 cosine 只反映输入变化，而非 Monte-Carlo 噪声（P4 单独覆盖后者）。",
        "",
        "## 复现命令",
        "",
        "```bash",
        f"cd {ROOT_DIR}",
        ".venv/bin/python pilot_eval.py --evaluate --resume --max-images 16 --global-deadline-sec 2400 --task-timeout-sec 180",
        "```",
        "",
        "原始记录：`eval_records.jsonl`；聚合 JSON：`eval_summary.json`；CSV：`eval_per_image.csv`、`eval_units.csv`。",
        "",
    ]
    return "\n".join(lines)


def write_eval_csvs(rows: Sequence[Dict[str, Any]], units: Sequence[Dict[str, Any]]) -> None:
    per_image_path = ROOT / "eval_per_image.csv"
    metric_extractors = [
        ("attribution_wall_sec", lambda r: r.get("attribution_wall_sec")),
        ("insertion_auc", lambda r: r.get("insertion_auc")),
        ("deletion_auc", lambda r: r.get("deletion_auc")),
        ("stability_cosine", lambda r: r.get("stability_cosine_mean")),
    ]
    with per_image_path.open("w", encoding="utf-8") as f:
        f.write("image_id,dataset,model,method,metric,value,time_ms\n")
        for r in rows:
            if r.get("record_type") != "evaluation_cell" or r.get("status") != "ok":
                continue
            for metric, getter in metric_extractors:
                value = getter(r)
                if value is None:
                    continue
                time_ms = (r.get("attribution_wall_sec") or 0.0) * 1000.0 if metric == "attribution_wall_sec" else ""
                f.write(f"{r.get('sample_id')},{r.get('dataset')},{r.get('model')},{r.get('method')},{metric},{value},{time_ms}\n")

    units_path = ROOT / "eval_units.csv"
    first_hash: Dict[Tuple[str, str, str], str] = {}
    for r in rows:
        if r.get("record_type") == "evaluation_cell" and r.get("status") == "ok":
            key = (str(r.get("dataset")), str(r.get("model")), str(r.get("method")))
            first_hash.setdefault(key, str(r.get("config_hash")))
    with units_path.open("w", encoding="utf-8") as f:
        f.write("method,model,dataset,metric,mean,std,n,config_hash\n")
        for u in units:
            key = (u["dataset"], u["model"], u["method"])
            for metric, mean_key, std_key in [
                ("attribution_wall_sec", "attribution_wall_sec_mean", "attribution_wall_sec_std"),
                ("insertion_auc", "insertion_auc_mean", "insertion_auc_std"),
                ("deletion_auc", "deletion_auc_mean", "deletion_auc_std"),
                ("stability_cosine", "stability_cosine_mean", "stability_cosine_std"),
                ("cell_wall_sec", "cell_wall_sec_mean", "cell_wall_sec_std"),
            ]:
                value = u.get(mean_key)
                if value is None:
                    continue
                std = u.get(std_key)
                f.write(f"{u['method']},{u['model']},{u['dataset']},{metric},{value},{'' if std is None else std},{u['cells_ok']},{first_hash.get(key, '')}\n")


def run_evaluate(args: argparse.Namespace) -> int:
    device, device_index = device_from_args(args)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
    manifest = load_manifest()
    start_wall = time.perf_counter()
    deadline = start_wall + float(args.global_deadline_sec)

    datasets = [d for d in args.datasets if d in DATASET_ORDER] or DATASET_ORDER
    first_sample = dataset_by_id(manifest, datasets[0])["samples"][0]
    first_raw, _ = load_raw_image(ROOT / first_sample["image_path"])
    dummy_x = pr.image_normalize(first_raw).to(device)

    snapshot: Dict[str, Any] = {
        "run_id": f"p2-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{pr.stable_hash({'seed': args.seed})}",
        "generated_utc": utc_now(),
        "stage": "P2_small_end_to_end_evaluation",
        "command": " ".join(str(x) for x in sys.argv),
        "arguments": vars(args),
        "environment": pr.device_snapshot(device_index or 0),
        "datasets": datasets,
        "models": args.models,
        "methods": args.methods,
        "max_images": args.max_images,
        "stability_k": args.stability_k,
        "sigma": args.sigma,
        "seed": args.seed,
        "quality_scope": "ood_pipeline_diagnostic",
    }
    pr.write_json(ROOT / "environment_snapshot_p2.json", snapshot)

    rows = pr.read_jsonl(args.records) if args.resume else []
    done_keys = {
        (r.get("dataset"), r.get("sample_id"), r.get("model"), r.get("method"))
        for r in rows
        if r.get("record_type") == "evaluation_cell" and r.get("status") == "ok"
    }

    combos = [(d, m, meth) for d in datasets for m in args.models for meth in args.methods]
    bundles: Dict[str, pr.ModelBundle] = {}
    try:
        for dataset_id, model_name, method in combos:
            if time.perf_counter() >= deadline:
                print(f"[{utc_now()}] global deadline reached before {dataset_id}/{model_name}/{method}", flush=True)
                break
            ds_meta = dataset_by_id(manifest, dataset_id)
            samples = ds_meta["samples"][: args.max_images]
            if model_name not in bundles:
                print(f"[{utc_now()}] loading {model_name}", flush=True)
                bundles[model_name] = pr.build_model(model_name, device, dummy_x)
                snapshot["models_loaded"] = {n: pr.model_signature(b) for n, b in bundles.items()}
                pr.write_json(ROOT / "environment_snapshot_p2.json", snapshot)
            bundle = bundles[model_name]
            sampler = pr.TelemetrySampler(device_index=device_index or 0)
            try:
                for sample in samples:
                    key = (dataset_id, sample["sample_id"], model_name, method)
                    if args.resume and key in done_keys:
                        continue
                    if time.perf_counter() >= deadline:
                        print(f"[{utc_now()}] deadline inside {dataset_id}/{model_name}/{method}", flush=True)
                        break
                    print(f"[{utc_now()}] eval {dataset_id}/{model_name}/{method}/{sample['sample_id']}", flush=True)
                    row = eval_image(
                        bundle,
                        dataset_id,
                        ds_meta,
                        sample,
                        method,
                        args,
                        device,
                        sampler,
                        args.seed,
                        float(args.task_timeout_sec),
                    )
                    row.update(
                        {
                            "run_id": snapshot["run_id"],
                            "elapsed_wall_sec": time.perf_counter() - start_wall,
                            "remaining_deadline_sec": max(0.0, deadline - time.perf_counter()),
                            "real_input": True,
                        }
                    )
                    pr.append_jsonl(args.records, row)
                    rows.append(row)
                    if row.get("status") == "ok":
                        done_keys.add(key)
                    summary = aggregate_eval(rows)
                    pr.write_json(
                        args.summary,
                        {
                            "generated_utc": utc_now(),
                            "stage": "P2_small_end_to_end_evaluation",
                            "quality_scope": "ood_pipeline_diagnostic",
                            "environment": snapshot["environment"],
                            "rows": summary,
                        },
                    )
                    args.report.write_text(render_eval_markdown(snapshot, rows, summary), encoding="utf-8")
                    print(
                        f"[{utc_now()}] {row.get('status')} {dataset_id}/{model_name}/{method} "
                        f"attr={row.get('attribution_wall_sec')} ins={row.get('insertion_auc')} del={row.get('deletion_auc')} "
                        f"stab={row.get('stability_cosine_mean')} remaining={row.get('remaining_deadline_sec'):.0f}s",
                        flush=True,
                    )
            finally:
                sampler.close()
    finally:
        units = aggregate_eval(rows)
        snapshot["finished_utc"] = utc_now()
        snapshot["elapsed_wall_sec"] = time.perf_counter() - start_wall
        snapshot["rows_written"] = len(rows)
        pr.write_json(ROOT / "environment_snapshot_p2.json", snapshot)
        pr.write_json(
            args.summary,
            {
                "generated_utc": utc_now(),
                "stage": "P2_small_end_to_end_evaluation",
                "quality_scope": "ood_pipeline_diagnostic",
                "environment": snapshot["environment"],
                "rows": units,
            },
        )
        args.report.write_text(render_eval_markdown(snapshot, rows, units), encoding="utf-8")
        write_eval_csvs(rows, units)
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return 0


def run_cost_curve(args: argparse.Namespace) -> int:
    device, device_index = device_from_args(args)
    manifest = load_manifest()
    ds_meta = dataset_by_id(manifest, args.dataset)
    sample = ds_meta["samples"][args.cost_image_index]
    raw, _ = load_raw_image(ROOT / sample["image_path"])
    x = pr.image_normalize(raw).to(device)
    baseline = pr.black_baseline(device)
    bundle = pr.build_model(args.cost_model, device, x)
    with torch.no_grad():
        logits = bundle.model(x)
        target_class = int(logits.argmax(dim=1).item())
    bundle = dataclasses.replace(bundle, target_class=target_class)
    sampler = pr.TelemetrySampler(device_index=device_index or 0)

    sweeps: List[Dict[str, Any]] = [
        {"method": "ig", "param": "n_steps", "values": [16, 32, 64]},
        {"method": "rise", "param": "n_masks", "values": [128, 512, 2048]},
        {"method": "kernelshap", "param": "n_samples", "values": [128, 512, 2048]},
        {"method": "lime", "param": "n_samples", "values": [128, 512, 2048]},
        {"method": "ablation", "param": "grid", "values": [7, 8]},
        {"method": "rise", "param": "internal_batch_size", "values": [8, 16, 32]},
        {"method": "kernelshap", "param": "internal_batch_size", "values": [8, 16, 32]},
    ]
    rows: List[Dict[str, Any]] = [] if not args.resume else pr.read_jsonl(args.cost_records)
    done = {(r.get("method"), r.get("param"), r.get("value")) for r in rows if r.get("status") == "ok"}
    for sweep in sweeps:
        for value in sweep["values"]:
            key = (sweep["method"], sweep["param"], value)
            if args.resume and key in done:
                continue
            overrides = {sweep["param"]: value}
            if sweep["param"] in {"n_steps", "n_masks", "n_samples", "grid", "internal_batch_size"}:
                base_batch = args.internal_batch_size if sweep["param"] != "internal_batch_size" else value
            else:
                base_batch = args.internal_batch_size
            config = pr.method_config(
                sweep["method"],
                int(base_batch),
                args.seed + 7 * len(rows),
                overrides=overrides,
            )
            row: Dict[str, Any] = {
                "record_type": "cost_curve",
                "method": sweep["method"],
                "param": sweep["param"],
                "value": value,
                "dataset": args.dataset,
                "sample_id": sample["sample_id"],
                "model": bundle.name,
                "config_hash": pr.stable_hash(config),
                "method_config": config,
                "utc": utc_now(),
            }
            try:
                bundle_to_use = dataclasses.replace(bundle, target_class=target_class)
                heat, timing = attribute(bundle_to_use, x, baseline, sweep["method"], config, sampler, device)
                row.update(
                    {
                        "status": "ok",
                        "attribution_wall_sec": timing.get("attribution_wall_sec"),
                        "attribution_cuda_event_sec": timing.get("attribution_cuda_event_sec"),
                        "attribution_forward_samples": timing.get("forward_samples"),
                        "cuda_max_memory_reserved_mib": timing.get("cuda_max_memory_reserved_mib"),
                        "attribution_abs_sum": float(heat.abs().sum().item()),
                    }
                )
                print(f"[{utc_now()}] curve {sweep['method']}/{sweep['param']}={value} {row['attribution_wall_sec']:.4f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                row.update({"status": "error", "failure_reason": f"{type(exc).__name__}: {exc}"})
                print(f"[{utc_now()}] curve FAIL {sweep['method']}/{sweep['param']}={value}: {exc}", flush=True)
            pr.append_jsonl(args.cost_records, row)
            rows.append(row)
    sampler.close()
    pr.write_json(
        args.cost_summary,
        {"generated_utc": utc_now(), "stage": "P3_cost_curve", "rows": rows},
    )
    write_cost_report(args, rows)
    return 0


def write_cost_report(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# XAI01-04 试跑 P3：成本曲线",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        f"数据集 `{args.dataset}` 样本 `{rows[0].get('sample_id') if rows else ''}`，模型 `{args.cost_model}`，"
        "单张 224x224 输入，同步墙钟计时。",
        "",
        "| 方法 | 参数 | 取值 | 墙钟(s) | CUDA event(s) | 前向样本数 | 显存峰值(MiB) | 状态 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def fmt(v: Any) -> str:
            return "null" if v is None else f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)

        lines.append(
            f"| {r.get('method')} | {r.get('param')} | {r.get('value')} | {fmt(r.get('attribution_wall_sec'))} | "
            f"{fmt(r.get('attribution_cuda_event_sec'))} | {fmt(r.get('attribution_forward_samples'))} | "
            f"{fmt(r.get('cuda_max_memory_reserved_mib'))} | {r.get('status')} |"
        )
    lines += [
        "",
        "> 解读边界：每个配置只测了同一张图的一次，适合判断成本量级，不适合拟合精确缩放曲线。512→2048 时 RISE / KernelSHAP / LIME 都约增至 4 倍；KernelSHAP 的 128 样本点反而慢于 512，属于需要重复测量才能解释的波动。internal batch 16→32 的加速约 4–5%。",
        "",
        "## 复现命令",
        "",
        "```bash",
        f"cd {ROOT_DIR}",
        ".venv/bin/python pilot_eval.py --cost-curve --resume --global-deadline-sec 1200 --task-timeout-sec 300",
        "```",
        "",
    ]
    (ROOT / "cost_curve_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_recheck_report(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# XAI01-04 试跑 P4：随机方法种子方差",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        "跨种子 cosine 比较同一图像上相互独立的归因种子，与 P2 的 K=2 输入扰动稳定性相互独立。",
        "",
        "| 方法 | 样本 | 重复数 | 耗时均值(s) | 耗时标准差(s) | 跨种子 cosine 均值 | 最小值 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(v: Any) -> str:
            return "null" if v is None else f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)

        lines.append(
            f"| {r.get('method')} | {r.get('sample_id')} | {r.get('replicates')} | {fmt(r.get('attribution_wall_sec_mean'))} | "
            f"{fmt(r.get('attribution_wall_sec_std'))} | {fmt(r.get('cross_seed_cosine_mean'))} | {fmt(r.get('cross_seed_cosine_min'))} |"
        )
    lines += [
        "",
        "## 复现命令",
        "",
        "```bash",
        f"cd {ROOT_DIR}",
        f".venv/bin/python pilot_eval.py --recheck --resume --replicates {args.replicates} --max-images {args.max_images} --global-deadline-sec 900",
        "```",
        "",
    ]
    (ROOT / "recheck_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_recheck(args: argparse.Namespace) -> int:
    device, device_index = device_from_args(args)
    manifest = load_manifest()
    ds_meta = dataset_by_id(manifest, args.dataset)
    rows: List[Dict[str, Any]] = [] if not args.resume else pr.read_jsonl(args.recheck_records)
    sampler = pr.TelemetrySampler(device_index=device_index or 0)
    first = ds_meta["samples"][0]
    first_raw, _ = load_raw_image(ROOT / first["image_path"])
    bundle = pr.build_model(args.cost_model, device, pr.image_normalize(first_raw).to(device))
    samples = ds_meta["samples"][: args.max_images]
    for method in ["rise", "kernelshap", "lime"]:
        for sample in samples:
            raw, _ = load_raw_image(ROOT / sample["image_path"])
            x = pr.image_normalize(raw).to(device)
            baseline = pr.black_baseline(device)
            with torch.no_grad():
                target = int(bundle.model(x).argmax(dim=1).item())
            b = dataclasses.replace(bundle, target_class=target)
            heats: List[torch.Tensor] = []
            times: List[float] = []
            for rep in range(args.replicates):
                config = pr.method_config(
                    method, args.internal_batch_size, args.seed + rep * 1009 + sample_seed(sample["sample_id"]) % 997
                )
                heat, timing = attribute(b, x, baseline, method, config, sampler, device)
                heats.append(heat.detach().cpu())
                times.append(float(timing.get("attribution_wall_sec") or float("nan")))
            pair_cos = [
                cosine_similarity(heats[i], heats[j])
                for i in range(len(heats))
                for j in range(i + 1, len(heats))
            ]
            pair_cos = [c for c in pair_cos if c is not None]
            row = {
                "record_type": "recheck_seed_variance",
                "method": method,
                "dataset": args.dataset,
                "sample_id": sample["sample_id"],
                "model": bundle.name,
                "target_class": target,
                "replicates": args.replicates,
                "attribution_wall_sec_values": times,
                "attribution_wall_sec_mean": pr.safe_mean(times),
                "attribution_wall_sec_std": float(np.std(times)) if times else None,
                "cross_seed_cosine_values": pair_cos,
                "cross_seed_cosine_mean": pr.safe_mean(pair_cos),
                "cross_seed_cosine_min": pr.safe_min(pair_cos),
                "utc": utc_now(),
                "status": "ok",
            }
            pr.append_jsonl(args.recheck_records, row)
            rows.append(row)
            print(f"[{utc_now()}] recheck {method}/{sample['sample_id']} seedCosMean={row['cross_seed_cosine_mean']}", flush=True)
    sampler.close()
    pr.write_json(args.recheck_summary, {"generated_utc": utc_now(), "stage": "P4_recheck", "rows": rows})
    write_recheck_report(args, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate", action="store_true", help="P2 small end-to-end evaluation")
    mode.add_argument("--cost-curve", action="store_true", help="P3 sampling cost curves")
    mode.add_argument("--recheck", action="store_true", help="P4 stochastic seed variance")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    p.add_argument("--dataset", choices=DATASET_ORDER, default="voc2007")
    p.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    p.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER)
    p.add_argument("--max-images", type=int, default=16)
    p.add_argument("--stability-k", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--faith-steps", type=int, default=20)
    p.add_argument("--faith-batch", type=int, default=8)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--internal-batch-size", type=int, default=pr.DEFAULT_INTERNAL_BATCH)
    p.add_argument("--global-deadline-sec", type=float, default=2400.0)
    p.add_argument("--task-timeout-sec", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--cost-model", choices=MODEL_ORDER, default="resnet50")
    p.add_argument("--cost-image-index", type=int, default=0)
    p.add_argument("--records", type=Path, default=ROOT / "eval_records.jsonl")
    p.add_argument("--summary", type=Path, default=ROOT / "eval_summary.json")
    p.add_argument("--report", type=Path, default=ROOT / "eval_report.md")
    p.add_argument("--cost-records", type=Path, default=ROOT / "cost_curve_records.jsonl")
    p.add_argument("--cost-summary", type=Path, default=ROOT / "cost_curve_summary.json")
    p.add_argument("--recheck-records", type=Path, default=ROOT / "recheck_records.jsonl")
    p.add_argument("--recheck-summary", type=Path, default=ROOT / "recheck_summary.json")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.evaluate:
        return run_evaluate(args)
    if args.cost_curve:
        return run_cost_curve(args)
    if args.recheck:
        return run_recheck(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
