import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "backend.celery_config", "backend.datetime_utils", "backend.features",
    "backend.perform", "backend.results", "backend.utils", "dash.colors",
    "dash.dashboard", "dash.layouts", "dash.routing_utils",
])
def test_legacy_modules_are_exact_aliases(module):
    legacy = importlib.import_module(f"selfx.{module}")
    maintained = importlib.import_module(f"ai4cps.{module}")
    assert legacy is maintained


def test_deprecation_warning_is_visible_once():
    result = subprocess.run(
        [sys.executable, "-c", "import selfx; import selfx; print(selfx.__version__)"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "0.1.42"
    assert result.stderr.count("FutureWarning: The selfx package is deprecated.") == 1
    assert "Install ai4cps" in result.stderr
