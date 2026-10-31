"""Regression coverage for native stderr and failed-run evidence."""

import pytest

from analysis import rise_formal_batch as batch


def runner_root(tmp_path, monkeypatch, source):
    experiments = tmp_path / 'experiments'
    experiments.mkdir()
    (experiments / 'run_unit.py').write_text(source, encoding='utf-8')
    monkeypatch.setattr(batch, 'ROOT', tmp_path)
    return tmp_path / 'config.yaml'


def test_stderr_progress_is_logged_without_failing(tmp_path, monkeypatch):
    config = runner_root(tmp_path, monkeypatch, "import sys\nprint('progress', file=sys.stderr)\nprint('done')\n")
    log = tmp_path / 'console.log'
    batch.execute(config, log)
    assert 'progress' in log.read_text()
    assert 'done' in log.read_text()


def test_nonzero_exit_preserves_log_and_stops_batch(tmp_path, monkeypatch):
    config = runner_root(tmp_path, monkeypatch, "import sys\nprint('failure evidence', flush=True)\nsys.exit(7)\n")
    log = tmp_path / 'console.log'
    with pytest.raises(ValueError, match='exit=7'):
        batch.execute(config, log)
    assert 'failure evidence' in log.read_text()
    with pytest.raises(FileExistsError):
        batch.execute(config, log)
