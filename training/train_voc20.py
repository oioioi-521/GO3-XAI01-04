"""Leakage-safe VOC20 multi-label training with resumable state.

The frozen 500-image evaluation manifest is deliberately not an input to this
module. Training and validation are driven only by the generated manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.factory import load_model
from preprocessing.dataset import load_image
from preprocessing.extract_voc_labels import VOC_CLASSES
from training.checkpoints import (
    atomic_torch_save,
    resume_payload,
    restore_rng,
    validate_identity,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class ManifestDataset(Dataset):
    """VOC manifest dataset; paths are project-relative to ``data``."""

    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"empty manifest: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        target = torch.tensor(json.loads(row["targets"]), dtype=torch.float32)
        if target.shape != (len(VOC_CLASSES),):
            raise ValueError(f"invalid target shape for {row['image_id']}: {target.shape}")
        return load_image(row["image_path"]), target, row["image_id"]


def _average_precision(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    order = torch.argsort(probabilities, descending=True)
    sorted_targets = targets[order]
    positives = sorted_targets.sum().item()
    if positives == 0:
        return 0.0
    ranks = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
    precision_at_hits = sorted_targets * sorted_targets.cumsum(0) / ranks
    return float(precision_at_hits.sum().item() / positives)


def validation_map(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Calculate finite multi-label mAP without consuming frozen data."""
    logits, targets = [], []
    model.eval()
    with torch.inference_mode():
        for images, batch_targets, _ in loader:
            logits.append(model(images.to(device)).cpu())
            targets.append(batch_targets)
    probabilities = torch.cat(logits).sigmoid()
    target_matrix = torch.cat(targets)
    ap = [_average_precision(probabilities[:, index], target_matrix[:, index]) for index in range(20)]
    report = {"mAP": sum(ap) / len(ap), "per_class_ap": dict(zip(VOC_CLASSES, ap))}
    if not torch.isfinite(torch.tensor(report["mAP"])):
        raise ValueError("validation mAP is not finite")
    return report


def _paths(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    output = config["output"]
    best = ROOT / output["checkpoint"]
    last = ROOT / output.get("last_checkpoint", str(best.with_name(f"{best.stem}.last.pt")))
    record = ROOT / output["record"]
    return best, last, record


def _identity(config: dict[str, Any], train_manifest: Path) -> dict[str, Any]:
    return {
        "model": config["model"]["name"],
        "classes": VOC_CLASSES,
        "manifest_sha256": sha256(train_manifest),
        "batch_size": int(config["runtime"]["batch_size"]),
        "lr": float(config["optimizer"]["lr"]),
        "early_stopping": config.get("early_stopping", {}),
    }


def _resume(
    path: Path,
    identity: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
) -> tuple[int, float, int | None, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_identity(payload, identity)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng(payload["rng"])
    return (
        int(payload["epoch"]) + 1,
        float(payload["best_metric"]),
        payload.get("best_epoch"),
        int(payload["patience_count"]),
    )


def run(config_path: str, resume: str | None = None, max_epochs: int | None = None) -> dict[str, Any]:
    """Train up to ``max_epochs`` and atomically save state at each full epoch."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("VOC20 training requires CUDA; use .venv-gpu")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")

    train_manifest = ROOT / config["data"]["train_manifest"]
    validation_manifest = ROOT / config["data"]["validation_manifest"]
    train_set = ManifestDataset(train_manifest)
    validation_set = ManifestDataset(validation_manifest)
    train_ids = {row["image_id"] for row in train_set.rows}
    validation_ids = {row["image_id"] for row in validation_set.rows}
    if train_ids & validation_ids:
        raise ValueError("training and validation manifests overlap")

    identity = _identity(config, train_manifest)
    model = load_model(
        config["model"]["name"], device, "default", 20, output_activation="sigmoid"
    ).train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(config["optimizer"]["lr"]), momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    scaler = torch.amp.GradScaler("cuda")
    class_counts = torch.tensor(
        [sum(json.loads(row["targets"])[index] for row in train_set.rows) for index in range(20)],
        device=device,
        dtype=torch.float32,
    )
    if (class_counts == 0).any():
        raise ValueError("training manifest does not cover every VOC class")
    pos_weight = ((len(train_set) - class_counts) / class_counts).clamp(max=20)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = int(config["runtime"]["batch_size"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size)
    best_path, last_path, record_path = _paths(config)
    early_stopping = config.get("early_stopping", {})
    patience = int(early_stopping.get("patience", 6))
    min_delta = float(early_stopping.get("min_delta", 0.0))
    if patience < 1 or min_delta < 0:
        raise ValueError("early_stopping requires patience >= 1 and min_delta >= 0")

    start_epoch, best_metric, best_epoch, patience_count = 1, -1.0, None, 0
    if resume is not None:
        start_epoch, best_metric, best_epoch, patience_count = _resume(
            ROOT / resume, identity, model, optimizer, scheduler, scaler
        )

    requested_epochs = int(max_epochs if max_epochs is not None else config["epochs"])
    if requested_epochs < start_epoch:
        raise ValueError("max epochs precedes resume checkpoint")
    last_report: dict[str, Any] | None = None
    for epoch in range(start_epoch, requested_epochs + 1):
        model.train()
        for images, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(images.to(device)), targets.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        last_report = validation_map(model, validation_loader, device)
        scheduler.step(last_report["mAP"])
        if last_report["mAP"] > best_metric + min_delta:
            best_metric = last_report["mAP"]
            best_epoch = epoch
            patience_count = 0
            atomic_torch_save(
                {
                    "state_dict": model.state_dict(),
                    "VOC_CLASSES": VOC_CLASSES,
                    "task_type": "multilabel",
                    "output_activation": "sigmoid",
                    "manifest_hash": identity["manifest_sha256"],
                    "best_epoch": epoch,
                    "validation": last_report,
                },
                best_path,
            )
        else:
            patience_count += 1
        atomic_torch_save(
            resume_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                best_epoch,
                patience_count,
                identity,
                last_report,
            ),
            last_path,
        )
        if patience_count >= patience:
            break

    if last_report is None or not best_path.is_file():
        raise RuntimeError("no completed epoch was available to record")
    record = {
        "best_epoch": best_epoch,
        "validation": last_report,
        "checkpoint_sha256": sha256(best_path),
        "last_checkpoint": str(last_path),
        "early_stopping": {
            "monitor": "mAP",
            "patience": patience,
            "min_delta": min_delta,
            "patience_count": patience_count,
        },
        "frozen_eval_used": False,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.resume, args.max_epochs), ensure_ascii=False))
