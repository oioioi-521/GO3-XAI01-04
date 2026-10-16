"""Recover only the missing postrun directory/check; never launch workers."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/xai-lime-20261013')
TOOLS = Path('/workspace/lime-optimization-20261014-v1')
OUT = ROOT/'results/dcu-lime-20261013/optimization_v1'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))
os.environ['TORCH_HOME'] = str(ROOT/'.cache/torch')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    from run_dcu_formal_queue import preflight, dump
    assert (TOOLS/'controller.exit').read_text().strip() == '1'
    log = (TOOLS/'controller.log').read_text()
    assert 'FileNotFoundError' in log and 'optimization_v1/postrun' in log
    assert not (OUT/'COMPLETE.json').exists(), 'do not overwrite a completion record'
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert sha(TOOLS/'benchmark_lime_vectorized.py') == manifest['source_sha256']
    assert sha(TOOLS/'lime_vectorized.py') == manifest['prototype_sha256']
    units, map_count = {}, 0
    for job in manifest['jobs']:
        unit = OUT/job
        summary = json.loads((unit/'summary.json').read_text())
        assert summary['state'] == 'complete' and not summary['formal_started']
        assert not summary['failures'] and summary['numeric_pass']
        assert len(summary['records']) == 46 and len(summary['comparisons']) == 32
        assert len({(r['case'],r['budget'],r['seed'],r['variant']) for r in summary['records']}) == 46
        for row in summary['records']:
            path = unit/row['map_path']; assert path.parent == unit
            assert sha(path) == row['map_sha256']
            a=np.load(path,allow_pickle=False)
            assert a.shape==(224,224) and a.dtype==np.float32 and np.isfinite(a).all()
            map_count+=1
        units[job] = sha(unit/'summary.json')
    (OUT/'postrun').mkdir(exist_ok=True)
    preflight(OUT/'postrun')
    recovery = dict(reason='controller postrun directory missing before shutil.disk_usage',
        original_controller_exit=1, original_controller_log_sha256=sha(TOOLS/'controller.log'),
        action='create missing postrun directory, verify frozen release and completed outputs; no worker reruns',
        source_sha256=sha(Path(__file__)), completed_at=time.time(), maps_verified=map_count,
        summary_sha256=units, frozen_postrun_preflight='passed', formal_started=False)
    dump(OUT/'finalization_recovery.json',recovery)
    dump(OUT/'COMPLETE.json',dict(status='probe_complete',numeric_pass=True,formal_started=False,
        original_controller_exit=1,recovered_finalization=True,recovery_sha256=sha(OUT/'finalization_recovery.json')))
    print(json.dumps(recovery,indent=2))


if __name__=='__main__':main()
