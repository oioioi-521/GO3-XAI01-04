"""Atomic resumable checkpoint helpers for VOC20 training."""
from __future__ import annotations
import math
import os, random
from pathlib import Path
from typing import Any
import torch

def rng_state() -> dict[str, Any]:
    state={"torch":torch.get_rng_state(),"python":random.getstate()}
    if torch.cuda.is_available(): state["cuda"]=torch.cuda.get_rng_state_all()
    return state

def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"]); random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])

def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,temporary); os.replace(temporary,path)

def validate_identity(payload: dict[str, Any], identity: dict[str, Any]) -> None:
    saved=payload.get("identity")
    if saved != identity: raise ValueError("resume checkpoint identity mismatch")


def saved_elapsed_seconds(payload: dict[str, Any]) -> float:
    """Read a prior cumulative duration, treating pre-v2 checkpoints as zero.

    A historical checkpoint has no trustworthy cumulative duration, so callers
    must not infer one from timestamps or epoch counts.
    """
    value = float(payload.get("elapsed_seconds_total", 0.0))
    if not math.isfinite(value) or value < 0:
        raise ValueError("resume checkpoint elapsed_seconds_total must be finite and non-negative")
    return value

def resume_payload(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_metric,
    best_epoch,
    patience_count,
    identity,
    report,
    elapsed_seconds_total=0.0,
):
    """Create the complete rolling-state payload after a successful epoch."""
    elapsed_seconds_total = saved_elapsed_seconds(
        {"elapsed_seconds_total": elapsed_seconds_total}
    )
    return {
        "schema": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "patience_count": patience_count,
        "elapsed_seconds_total": elapsed_seconds_total,
        "identity": identity,
        "rng": rng_state(),
        "report": report,
    }
