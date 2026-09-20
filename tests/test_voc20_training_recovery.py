"""Contracts for rolling VOC20 recovery state and final artifact isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from preprocessing.extract_voc_labels import VOC_CLASSES
from training.checkpoints import atomic_torch_save, resume_payload, validate_identity
from training.multilabel_metrics import multilabel_report
from training.train_voc20 import _resume


def _training_objects():
    model = torch.nn.Linear(3, 20)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    identity = {
        "model": "resnet50",
        "classes": VOC_CLASSES,
        "manifest_sha256": "manifest",
        "batch_size": 32,
        "lr": 0.0001,
        "early_stopping": {"patience": 2, "min_delta": 0.0},
        "threshold": 0.5,
    }
    return model, optimizer, scheduler, scaler, identity


def test_rolling_payload_is_atomic_and_contains_recovery_state(tmp_path: Path):
    model, optimizer, scheduler, scaler, identity = _training_objects()
    optimizer.zero_grad()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step(0.4)
    payload = resume_payload(
        model, optimizer, scheduler, scaler, 1, 0.4, 1, 0, identity, {"mAP": 0.4}
    )
    target = tmp_path / "resume" / "last.pt"
    atomic_torch_save(payload, target)
    restored = torch.load(target, map_location="cpu", weights_only=False)

    assert target.is_file()
    assert not target.with_suffix(".pt.tmp").exists()
    assert restored["epoch"] == 1
    assert restored["best_epoch"] == 1
    assert restored["patience_count"] == 0
    assert restored["optimizer"] == optimizer.state_dict()
    assert restored["scheduler"] == scheduler.state_dict()
    assert restored["scaler"] == scaler.state_dict()
    validate_identity(restored, identity)


def test_resume_identity_rejects_model_class_and_manifest_mismatch():
    _, _, _, _, identity = _training_objects()
    payload = {"identity": identity}
    for field, replacement in (("model", "vgg16"), ("classes", list(reversed(VOC_CLASSES))), ("manifest_sha256", "other")):
        mismatched = dict(identity)
        mismatched[field] = replacement
        with pytest.raises(ValueError, match="identity mismatch"):
            validate_identity(payload, mismatched)


def test_resume_restores_next_epoch_optimizer_scheduler_scaler_and_patience(tmp_path: Path):
    model, optimizer, scheduler, scaler, identity = _training_objects()
    optimizer.zero_grad()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step(0.3)
    saved_weight = model.weight.detach().clone()
    path = tmp_path / "last.pt"
    atomic_torch_save(
        resume_payload(model, optimizer, scheduler, scaler, 1, 0.3, 1, 2, identity, {"mAP": 0.3}),
        path,
    )
    restored_model, restored_optimizer, restored_scheduler, restored_scaler, _ = _training_objects()
    start, best, best_epoch, patience_count = _resume(
        path, identity, restored_model, restored_optimizer, restored_scheduler, restored_scaler
    )

    assert start == 2
    assert (best, best_epoch, patience_count) == (0.3, 1, 2)
    assert torch.equal(restored_model.weight, saved_weight)
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_scaler.state_dict() == scaler.state_dict()


def test_f1_map_metrics_are_finite_for_empty_positive_predictions():
    logits = torch.full((3, 20), -100.0)
    targets = torch.zeros((3, 20))
    targets[0, 0] = 1
    report = multilabel_report(logits, targets, VOC_CLASSES, threshold=0.5)

    assert report["threshold"] == 0.5
    assert all(torch.isfinite(torch.tensor(report[key])) for key in ("mAP", "macro_f1", "micro_f1"))
    assert report["macro_f1"] == 0.0
    assert report["micro_f1"] == 0.0


def test_final_and_smoke_artifacts_are_separate():
    root = Path(__file__).resolve().parents[1]
    final = yaml.safe_load((root / "configs/train_voc20_resnet50.yaml").read_text(encoding="utf-8"))
    smoke = yaml.safe_load(
        (root / "configs/train_voc20_resnet50_resume_smoke.yaml").read_text(encoding="utf-8")
    )

    assert final["output"]["checkpoint"] == "models/checkpoints/resnet50_voc20.pt"
    assert final["output"]["last_checkpoint"] == "models/checkpoints/resume/resnet50_voc20.last.pt"
    assert "smoke" in smoke["output"]["checkpoint"]
    assert "smoke" in smoke["output"]["last_checkpoint"]
    assert set(final["output"].values()).isdisjoint(set(smoke["output"].values()))
