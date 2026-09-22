"""Append formal stability metrics without rerunning original MoRF metrics.

The runner consumes one existing formal unit configuration and a separately
versioned stability protocol. It reuses the original float32 attribution map
saved by ``run_unit.py`` and computes only perturbed-input attributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_unit import (  # noqa: E402
    PER_IMAGE_COLUMNS,
    UNIT_COLUMNS,
    _black_baseline,
    _build_attributor,
    _config_hash,
    _load_config,
    _resolve,
    _select_device,
    _synchronize,
)
from experiments.stability import (  # noqa: E402
    PROTOCOL_VERSION,
    perturb_rgb,
    stability_similarity,
    stable_seed,
)
from models import load_model  # noqa: E402
from preprocessing.dataset import MetadataDataset  # noqa: E402


STABILITY_METRICS = ("stability_spearman", "stability_valid_rate")
TRACE_COLUMNS = [
    "image_id",
    "dataset",
    "model",
    "method",
    "repeat",
    "seed",
    "protocol_version",
    "config_hash",
    "target",
    "sigma",
    "status",
    "spearman",
    "score",
    "top10_jaccard",
    "original_pred",
    "perturbed_pred",
    "prediction_preserved",
    "original_target_score",
    "perturbed_target_score",
    "target_score_abs_delta",
    "perturbation_mae_pixel",
    "attribution_time_ms",
]


def _load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("stability protocol root must be a mapping")
    required = {
        "protocol_version",
        "noise_distribution",
        "noise_domain",
        "sigma",
        "clip",
        "repeats",
        "target",
        "similarity",
        "direction",
        "top_fraction",
        "seed_fields",
        "degenerate_score",
        "runtime",
        "output",
    }
    missing = required - set(protocol)
    unknown = set(protocol) - required
    if missing or unknown:
        raise ValueError(
            f"stability protocol keys mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    frozen = {
        "protocol_version": PROTOCOL_VERSION,
        "noise_distribution": "gaussian",
        "noise_domain": "rgb_0_1",
        "sigma": 0.005,
        "clip": [0.0, 1.0],
        "repeats": 5,
        "target": "metadata",
        "similarity": "spearman",
        "direction": "max",
        "top_fraction": 0.1,
        "seed_fields": ["dataset", "image_id", "repeat"],
        "degenerate_score": 0.0,
    }
    for key, expected in frozen.items():
        if protocol[key] != expected:
            raise ValueError(
                f"stability protocol {key} must be {expected!r}, got {protocol[key]!r}"
            )
    runtime = protocol["runtime"]
    output = protocol["output"]
    if not isinstance(runtime, dict) or set(runtime) != {"device", "warmup_runs"}:
        raise ValueError("stability runtime must contain only device and warmup_runs")
    if (
        isinstance(runtime["warmup_runs"], bool)
        or not isinstance(runtime["warmup_runs"], int)
        or runtime["warmup_runs"] != 1
    ):
        raise ValueError("stability runtime.warmup_runs must be 1")
    if not isinstance(runtime["device"], str) or not runtime["device"]:
        raise ValueError("stability runtime.device must be a non-empty string")
    required_output = {"trace_csv", "state_dir", "config_dir", "run_log"}
    if not isinstance(output, dict) or set(output) != required_output:
        raise ValueError(
            "stability output must contain trace_csv, state_dir, config_dir, and run_log"
        )
    if any(not isinstance(output[key], str) or not output[key] for key in required_output):
        raise ValueError("stability output paths must be non-empty strings")
    return protocol


def _safe_image_id(image_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(image_id)
    )


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _validate_existing_csv(path: Path, columns: list[str], label: str) -> None:
    frame = _read_frame(path)
    if frame.empty and not path.is_file():
        return
    if list(frame.columns) != columns:
        raise ValueError(
            f"existing {label} has incompatible columns: {list(frame.columns)}"
        )


def _unit_mask(frame: pd.DataFrame, unit: dict[str, str]) -> pd.Series:
    required = {"dataset", "model", "method"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    return (
        frame["dataset"].eq(unit["dataset"])
        & frame["model"].eq(unit["model"])
        & frame["method"].eq(unit["method"])
    )


def _replace_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _remove_stability_rows(
    path: Path,
    unit: dict[str, str],
    *,
    image_ids: set[str] | None = None,
) -> None:
    frame = _read_frame(path)
    if frame.empty or "metric" not in frame.columns:
        return
    remove = _unit_mask(frame, unit) & frame["metric"].isin(STABILITY_METRICS)
    if image_ids is not None:
        remove &= frame["image_id"].astype(str).isin(image_ids)
    if remove.any():
        _replace_frame(path, frame[~remove])


def _remove_stability_summaries(path: Path, unit: dict[str, str]) -> None:
    frame = _read_frame(path)
    if frame.empty or "metric" not in frame.columns:
        return
    remove = _unit_mask(frame, unit) & frame["metric"].isin(STABILITY_METRICS)
    if remove.any():
        _replace_frame(path, frame[~remove])


def _remove_trace_rows(
    path: Path,
    unit: dict[str, str],
    *,
    image_ids: set[str] | None = None,
) -> None:
    frame = _read_frame(path)
    if frame.empty:
        return
    remove = _unit_mask(frame, unit)
    if image_ids is not None:
        remove &= frame["image_id"].astype(str).isin(image_ids)
    if remove.any():
        _replace_frame(path, frame[~remove])


def _state_path(state_dir: Path, unit: dict[str, str]) -> Path:
    safe = "_".join(unit[key] for key in ("method", "model", "dataset"))
    return state_dir / f"stability_{safe}.json"


def _read_state_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("config_hash") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def _prepare_state(
    state_path: Path,
    per_image_path: Path,
    units_path: Path,
    trace_path: Path,
    unit: dict[str, str],
    config_hash: str,
    force: bool,
) -> None:
    previous_hash = _read_state_hash(state_path)
    if force or previous_hash != config_hash:
        _remove_stability_rows(per_image_path, unit)
        _remove_stability_summaries(units_path, unit)
        _remove_trace_rows(trace_path, unit)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({**unit, "config_hash": config_hash}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(state_path)


def _completed_images(
    per_image_path: Path,
    trace_path: Path,
    unit: dict[str, str],
    config_hash: str,
    repeats: int,
) -> set[str]:
    per_image = _read_frame(per_image_path)
    if per_image.empty or not set(PER_IMAGE_COLUMNS).issubset(per_image.columns):
        return set()
    metric_rows = per_image[
        _unit_mask(per_image, unit) & per_image["metric"].isin(STABILITY_METRICS)
    ].copy()
    if metric_rows.empty:
        return set()
    metric_rows["_image_id"] = metric_rows["image_id"].astype(str)
    grouped_metrics = metric_rows.groupby("_image_id")["metric"]
    metric_unique = grouped_metrics.nunique()
    metric_size = grouped_metrics.size()
    metric_complete = set(
        metric_unique[
            (metric_unique == len(STABILITY_METRICS))
            & (metric_size == len(STABILITY_METRICS))
        ].index
    )

    trace = _read_frame(trace_path)
    if trace.empty or not set(TRACE_COLUMNS).issubset(trace.columns):
        return set()
    selected = trace[
        _unit_mask(trace, unit) & trace["config_hash"].astype(str).eq(config_hash)
    ].copy()
    if selected.empty:
        return set()
    selected["_image_id"] = selected["image_id"].astype(str)
    expected = set(range(repeats))
    trace_complete = {
        image_id
        for image_id, group in selected.groupby("_image_id")
        if len(group) == repeats
        and set(pd.to_numeric(group["repeat"], errors="coerce").dropna().astype(int))
        == expected
        and group["status"].notna().all()
    }
    return metric_complete & trace_complete


def _cleanup_partial_images(
    per_image_path: Path,
    trace_path: Path,
    unit: dict[str, str],
    completed: set[str],
) -> None:
    partial: set[str] = set()
    per_image = _read_frame(per_image_path)
    if not per_image.empty and "metric" in per_image.columns:
        rows = per_image[
            _unit_mask(per_image, unit) & per_image["metric"].isin(STABILITY_METRICS)
        ]
        partial.update(rows["image_id"].astype(str))
    trace = _read_frame(trace_path)
    if not trace.empty:
        partial.update(trace[_unit_mask(trace, unit)]["image_id"].astype(str))
    partial -= completed
    if partial:
        _remove_stability_rows(per_image_path, unit, image_ids=partial)
        _remove_trace_rows(trace_path, unit, image_ids=partial)


def _append_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def _upsert_summary(
    per_image_path: Path,
    units_path: Path,
    unit: dict[str, str],
    config_hash: str,
) -> None:
    frame = _read_frame(per_image_path)
    selected = frame[
        _unit_mask(frame, unit) & frame["metric"].isin(STABILITY_METRICS)
    ].copy()
    if selected.empty:
        return
    summary = selected.groupby("metric")["value"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    summary.insert(0, "dataset", unit["dataset"])
    summary.insert(0, "model", unit["model"])
    summary.insert(0, "method", unit["method"])
    summary["std"] = summary["std"].fillna(0.0)
    summary["config_hash"] = config_hash
    summary = summary[UNIT_COLUMNS]

    old = _read_frame(units_path)
    if not old.empty:
        keep = ~(_unit_mask(old, unit) & old["metric"].isin(STABILITY_METRICS))
        if keep.any():
            summary = pd.concat([old[keep], summary], ignore_index=True)
    _replace_frame(units_path, summary)


def _load_reference_map(
    base_config: dict[str, Any],
    unit: dict[str, str],
    image_id: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    output = base_config["output"]
    if not output.get("save_float_maps", False):
        raise ValueError("base config must enable output.save_float_maps")
    root = _resolve(str(output.get("float_maps_dir", "results/maps_float")))
    path = (
        root
        / f"{unit['method']}_{unit['model']}_{unit['dataset']}"
        / f"{_safe_image_id(image_id)}.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing original float32 attribution map: {path}")
    attribution = np.load(path, allow_pickle=False)
    if attribution.dtype != np.float32:
        raise ValueError(f"original attribution must be float32: {path}")
    if attribution.shape != expected_shape:
        raise ValueError(
            f"original attribution has shape {attribution.shape}, expected {expected_shape}: {path}"
        )
    if not np.isfinite(attribution).all():
        raise ValueError(f"original attribution contains non-finite values: {path}")
    return attribution


def _scores(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_activation: str,
) -> tuple[torch.Tensor, int]:
    with torch.inference_mode():
        logits = model(image)
    if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise ValueError("model must return one finite logits row")
    if output_activation == "softmax":
        scores = logits.softmax(dim=1)
    elif output_activation == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise ValueError("output_activation must be softmax or sigmoid")
    return scores, int(scores.argmax(dim=1).item())


def _finite_or_blank(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def run(
    config_path: Path,
    protocol_path: Path,
    *,
    max_images: int | None = None,
    force: bool = False,
) -> None:
    base_config = _load_config(config_path)
    protocol = _load_protocol(protocol_path)
    if max_images is not None and (
        isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0
    ):
        raise ValueError("max_images override must be a positive integer")

    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
    dataset_name = str(base_config["dataset"]["name"]).lower()
    split = str(base_config["dataset"].get("split", "debug")).lower()
    model_config = base_config["model"]
    model_name = str(model_config["name"]).lower()
    method = str(base_config["method"]).lower()
    unit = {"dataset": dataset_name, "model": model_name, "method": method}
    limit = max_images if max_images is not None else base_config["runtime"].get("max_images")
    execution_config = {
        "base_config": base_config,
        "base_config_hash": _config_hash(base_config),
        "stability": protocol,
        "max_images": limit,
    }
    digest = _config_hash(execution_config)

    device = _select_device(str(protocol["runtime"]["device"]))
    method_seed = int(base_config["runtime"].get("seed", 42))
    torch.manual_seed(method_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(method_seed)
    repeats = int(protocol["repeats"])
    sigma = float(protocol["sigma"])
    output_activation = str(
        model_config.get(
            "output_activation", "sigmoid" if dataset_name == "voc" else "softmax"
        )
    )
    checkpoint = model_config.get("checkpoint")
    checkpoint_path = _resolve(str(checkpoint)) if checkpoint else None
    model = load_model(
        name=model_name,
        device=device,
        weights=str(model_config.get("weights", "default")),
        num_classes=int(model_config.get("num_classes", 1000)),
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
        output_activation=output_activation,
    )
    model.eval()
    dataset = MetadataDataset(dataset_name, split=split, limit=limit)
    baseline = _black_baseline(device, torch.float32)
    attributor = _build_attributor(
        method, model=model, attribution=base_config["attribution"]
    )

    per_image_path = _resolve(str(base_config["output"]["per_image_csv"]))
    units_path = _resolve(str(base_config["output"]["units_csv"]))
    trace_path = _resolve(str(protocol["output"]["trace_csv"]))
    _validate_existing_csv(per_image_path, PER_IMAGE_COLUMNS, "per-image CSV")
    _validate_existing_csv(units_path, UNIT_COLUMNS, "unit CSV")
    _validate_existing_csv(trace_path, TRACE_COLUMNS, "stability trace CSV")
    state_dir = _resolve(str(protocol["output"]["state_dir"]))
    state_path = _state_path(state_dir, unit)
    _prepare_state(
        state_path,
        per_image_path,
        units_path,
        trace_path,
        unit,
        digest,
        force,
    )
    completed = set() if force else _completed_images(
        per_image_path, trace_path, unit, digest, repeats
    )
    _cleanup_partial_images(per_image_path, trace_path, unit, completed)

    pending_sample = next(
        (sample for sample in dataset if str(sample["image_id"]) not in completed), None
    )
    warmup_runs = int(protocol["runtime"]["warmup_runs"])
    if pending_sample is not None:
        warmup_seed = stable_seed(dataset_name, str(pending_sample["image_id"]), 0)
        warmup_cpu, _ = perturb_rgb(
            pending_sample["image"].unsqueeze(0), sigma=sigma, seed=warmup_seed
        )
        warmup_image = warmup_cpu.to(device)
        for _ in range(warmup_runs):
            attributor.attribute(
                warmup_image,
                target=int(pending_sample["target"]),
                baseline=baseline,
            )
        _synchronize(device)

    snapshot_dir = _resolve(str(protocol["output"]["config_dir"]))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{digest}_stability_{config_path.name}"
    temporary_snapshot = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    snapshot = {
        **execution_config,
        "base_config_path": str(config_path),
        "stability_protocol_path": str(protocol_path),
    }
    with temporary_snapshot.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)
    temporary_snapshot.replace(snapshot_path)
    print(
        f"stability={unit['method']}/{model_name}/{dataset_name}:{split} "
        f"device={device} images={len(dataset)} repeats={repeats} "
        f"protocol={PROTOCOL_VERSION} config_hash={digest}"
    )

    processed = 0
    degenerate_repeats = 0
    for sample in tqdm(dataset, desc=f"STABILITY {method.upper()} {model_name} {dataset_name}"):
        image_id = str(sample["image_id"])
        if image_id in completed:
            continue
        _remove_stability_rows(per_image_path, unit, image_ids={image_id})
        _remove_trace_rows(trace_path, unit, image_ids={image_id})

        image = sample["image"].unsqueeze(0).to(device)
        target = int(sample["target"])
        reference = _load_reference_map(
            base_config, unit, image_id, tuple(image.shape[-2:])
        )
        original_scores, original_pred = _scores(model, image, output_activation)
        original_target = float(original_scores[0, target].item())
        repeat_rows: list[dict[str, Any]] = []
        aggregate_scores: list[float] = []
        valid_repeats = 0

        for repeat in range(repeats):
            seed = stable_seed(dataset_name, image_id, repeat)
            perturbed_cpu, perturbation_mae = perturb_rgb(
                sample["image"].unsqueeze(0), sigma=sigma, seed=seed
            )
            perturbed = perturbed_cpu.to(device)
            perturbed_scores, perturbed_pred = _scores(
                model, perturbed, output_activation
            )
            _synchronize(device)
            started = time.perf_counter()
            candidate = attributor.attribute(
                perturbed, target=target, baseline=baseline
            ).numpy().astype(np.float32, copy=False)
            _synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            similarity = stability_similarity(
                candidate,
                reference,
                top_fraction=float(protocol["top_fraction"]),
                degenerate_score=float(protocol["degenerate_score"]),
            )
            aggregate_scores.append(similarity.score)
            if similarity.status == "valid":
                valid_repeats += 1
            else:
                degenerate_repeats += 1
            perturbed_target = float(perturbed_scores[0, target].item())
            repeat_rows.append({
                "image_id": image_id,
                **unit,
                "repeat": repeat,
                "seed": seed,
                "protocol_version": PROTOCOL_VERSION,
                "config_hash": digest,
                "target": target,
                "sigma": sigma,
                "status": similarity.status,
                "spearman": _finite_or_blank(similarity.spearman),
                "score": similarity.score,
                "top10_jaccard": _finite_or_blank(similarity.top_jaccard),
                "original_pred": original_pred,
                "perturbed_pred": perturbed_pred,
                "prediction_preserved": original_pred == perturbed_pred,
                "original_target_score": original_target,
                "perturbed_target_score": perturbed_target,
                "target_score_abs_delta": abs(perturbed_target - original_target),
                "perturbation_mae_pixel": perturbation_mae,
                "attribution_time_ms": elapsed_ms,
            })

        _append_csv(trace_path, TRACE_COLUMNS, repeat_rows)
        common = {"image_id": image_id, **unit, "time_ms": ""}
        rows = [
            {
                **common,
                "metric": "stability_spearman",
                "value": float(np.mean(aggregate_scores)),
            },
            {
                **common,
                "metric": "stability_valid_rate",
                "value": valid_repeats / repeats,
            },
        ]
        _append_csv(per_image_path, PER_IMAGE_COLUMNS, rows)
        processed += 1
        tqdm.write(
            f"{image_id}: stability={np.mean(aggregate_scores):.4f} "
            f"valid={valid_repeats}/{repeats}"
        )

    _upsert_summary(per_image_path, units_path, unit, digest)
    log_path = _resolve(str(protocol["output"]["run_log"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            **unit,
            "split": split,
            "protocol_version": PROTOCOL_VERSION,
            "config_hash": digest,
            "base_config_hash": _config_hash(base_config),
            "device": str(device),
            "scipy_version": scipy.__version__,
            "warmup_runs": warmup_runs,
            "method_seed": method_seed,
            "repeats": repeats,
            "processed": processed,
            "skipped": len(completed),
            "degenerate_repeats": degenerate_repeats,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, ensure_ascii=False) + "\n")
    print(
        f"done: processed={processed}, skipped={len(completed)}, "
        f"degenerate_repeats={degenerate_repeats}, output={per_image_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/stability_protocol_v1.yaml"),
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    protocol_path = (
        args.protocol if args.protocol.is_absolute() else PROJECT_ROOT / args.protocol
    )
    run(
        config_path.resolve(),
        protocol_path.resolve(),
        max_images=args.max_images,
        force=args.force,
    )


if __name__ == "__main__":
    main()
