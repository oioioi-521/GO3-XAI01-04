import numpy as np
import pytest

from analysis.rise_sampling_gate import _map_similarity


def test_map_similarity_reports_identity_and_change():
    reference = np.arange(16, dtype=np.float32).reshape(4, 4)
    correlation, mae = _map_similarity(reference.copy(), reference)
    assert correlation == pytest.approx(1.0)
    assert mae == 0.0

    candidate = reference[::-1].copy()
    correlation, mae = _map_similarity(candidate, reference)
    assert correlation < 1.0
    assert mae > 0.0
