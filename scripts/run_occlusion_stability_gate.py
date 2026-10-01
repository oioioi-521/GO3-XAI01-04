"""Run one isolated stability gate with measured process time and CUDA peak.

The actual algorithm and CSV schema remain owned by experiments/run_stability.py.
This wrapper adds an ignored diagnostic run record; it never invokes MoRF.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch

from experiments import run_stability


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    gate = args.gate_dir.resolve()
    allowed = (ROOT / "results/occlusion_stability_gates").resolve()
    if not gate.is_relative_to(allowed) or not gate.is_dir():
        raise ValueError("gate-dir must be an existing isolated stability gate")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU stability gate")
    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache/torch"))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    record: dict = {"gate_dir": gate.relative_to(ROOT).as_posix(), "force": args.force}
    try:
        run_stability.run(gate / "base.yaml", gate / "protocol.yaml", force=args.force)
        record["status"] = "success"
        log = gate / "stability_run_log.jsonl"
        record["runner_log"] = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    except Exception as error:
        record.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        torch.cuda.synchronize()
        record["wall_seconds_this_process"] = time.perf_counter() - started
        record["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        record["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
        with (gate / "gate_process_runs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
