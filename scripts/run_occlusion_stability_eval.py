"""Run one frozen Occlusion stability eval unit with process-level diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from experiments import run_stability
from scripts.prepare_occlusion_stability_eval import DEST, ROOT, UNITS


def run(unit: str) -> None:
    if unit not in UNITS:
        raise ValueError(f"unknown unit: {unit}")
    output = DEST / unit
    protocol = output / "protocol.yaml"
    config = ROOT / "configs" / f"occlusion_{unit}.yaml"
    backup = DEST / "backup" / "backup_manifest.json"
    if not output.is_dir() or not protocol.is_file() or not backup.is_file():
        raise FileNotFoundError("verified eval backup and isolated protocol are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU formal run")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    record: dict = {"unit": unit, "status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        run_stability.run(config, protocol)
        record["status"] = "success"
        log = ROOT / "results/occlusion_stability_eval" / unit / "run_log.jsonl"
        record["runner_log"] = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    except Exception as error:
        record.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            torch.cuda.synchronize()
            record["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            record["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
        except Exception as error:
            record["cuda_diagnostic_error"] = repr(error)
        record["wall_seconds_this_process"] = time.perf_counter() - start
        with (output / "process_runs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", choices=UNITS, required=True)
    args = parser.parse_args()
    run(args.unit)
