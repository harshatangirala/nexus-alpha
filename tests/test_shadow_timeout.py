"""The shadow forecaster must never hang on one request (a run hung for six hours on 2026-11-06)."""
import importlib.util
import time
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "shadow_forecast", Path(__file__).resolve().parents[1] / "scripts" / "shadow" / "shadow_forecast.py")
SF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SF)


def test_timeout_abandons_a_hung_call():
    t0 = time.time()
    with pytest.raises(TimeoutError):
        SF.with_timeout(lambda: time.sleep(30), 0.5)
    assert time.time() - t0 < 5


def test_result_and_errors_pass_through():
    assert SF.with_timeout(lambda: 42, 5) == 42
    with pytest.raises(ValueError, match="no daily data"):
        SF.with_timeout(lambda: (_ for _ in ()).throw(ValueError("no daily data for X")), 5)


def test_timeouts_are_retryable():
    assert any(x in repr(TimeoutError("no answer within 180s")) for x in SF.RETRYABLE)
