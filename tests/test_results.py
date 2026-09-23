from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import joblib
import numpy as np
import pytest

from selfx.backend import results


def test_concurrent_saves_publish_complete_results(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "DEFAULT_RESULTS_DIR", tmp_path)
    finished_writing = Barrier(2)
    original_dump = joblib.dump

    def overlapping_dump(value, file, **kwargs):
        original_dump(value, file, **kwargs)
        if hasattr(file, "flush"):
            file.flush()
        finished_writing.wait(timeout=10)

    monkeypatch.setattr(results.joblib, "dump", overlapping_dump)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(results.store_result, None, "feature", np.full(10000, value))
            for value in (1, 2)
        ]
        for future in futures:
            future.result(timeout=15)

    stored = results.get_result("Online/feature.joblib")
    assert stored.shape == (10000,)
    assert np.all(stored == 1) or np.all(stored == 2)
    assert list((tmp_path / "Online").iterdir()) == [
        tmp_path / "Online" / "feature.joblib"
    ]


def test_failed_save_preserves_previous_result(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "DEFAULT_RESULTS_DIR", tmp_path)
    results.store_result(None, "feature", {"value": "previous"})

    def failed_dump(value, file, **kwargs):
        file.write(b"partial result")
        raise ValueError("Serialization failed")

    monkeypatch.setattr(results.joblib, "dump", failed_dump)
    with pytest.raises(ValueError, match="Serialization failed"):
        results.store_result(None, "feature", {"value": "new"})

    assert results.get_result("Online/feature.joblib") == {"value": "previous"}
    assert list((tmp_path / "Online").iterdir()) == [
        tmp_path / "Online" / "feature.joblib"
    ]
