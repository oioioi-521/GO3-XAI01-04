"""Atomic resumable checkpoint helpers for VOC20 training."""
from __future__ import annotations
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

def resume_payload(model,optimizer,scheduler,scaler,epoch,best_metric,patience_count,identity,report):
    return {"schema":1,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,"scaler":scaler.state_dict(),"epoch":epoch,"best_metric":best_metric,"patience_count":patience_count,"identity":identity,"rng":rng_state(),"report":report}
