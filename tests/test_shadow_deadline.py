"""A-81 (audit 7): one stock with an older last bar must not stop the nightly forecaster.

The deadline used to be the earliest decision time over the forecasts; a single stale stock (halted, or Yahoo lagging
as on 2026-11-13, A-80) then set a deadline in the past and the run stopped after one ticker.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _forecaster():
    spec = importlib.util.spec_from_file_location("shadow_forecast_under_test",
                                                  REPO / "scripts" / "shadow" / "shadow_forecast.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(monkeypatch, tmp_path, capsys, stale: set[str]):
    mod = _forecaster()
    today = pd.Timestamp.now().normalize()            # its decision time (next weekday 09:00 New York) is ahead
    old = today - pd.Timedelta(days=14)               # a stale stock: its decision time has long passed
    tickers = ["AAA", "BBB", "CCC", "DDD"]

    class FakePredictor:
        def __init__(self, bdir):
            pass

        def predict(self, t, mode, explain):
            f = SimpleNamespace(down_probability=0.2, sideways_probability=0.5, up_probability=0.3, return_q10=-0.01,
                                expected_return=0.0, return_q90=0.01, direction_skill_validated=False)
            asof = (old if t in stale else today).strftime("%Y-%m-%d")
            return SimpleNamespace(instrument=SimpleNamespace(asof_session=asof, reference_price=1.0,
                                                              data_quality="ok"), forecasts={"1D": f})

    monkeypatch.setattr(mod, "NexusPredictor", FakePredictor)
    monkeypatch.setattr(mod.ST, "sample_index", lambda: pd.DataFrame({"ticker": tickers}))
    monkeypatch.setattr(mod.SH, "verify_frozen", lambda v: (tmp_path, {"bundle_id": "test"}))
    monkeypatch.setattr(mod.SH, "SHADOW_ROOT", tmp_path)
    monkeypatch.setattr(mod.SH, "expected_asof", lambda now=None: today)
    monkeypatch.setattr(mod.SH, "verify_chain", lambda v: [])

    def fake_exit(code):
        raise SystemExit(code)
    monkeypatch.setattr(mod.os, "_exit", fake_exit)
    monkeypatch.setattr(sys, "argv", ["shadow_forecast.py", "--version", "v1.1", "--dry-run"])
    with pytest.raises(SystemExit):
        mod.main()
    return capsys.readouterr().out, today.strftime("%Y-%m-%d")


def test_a_stale_first_stock_does_not_stop_the_run(monkeypatch, tmp_path, capsys):
    out, today = _run(monkeypatch, tmp_path, capsys, stale={"AAA"})
    assert "stopped at the decision-time deadline" not in out
    assert f"[dry-run] {today}: 3 tickers, late rows 0" in out       # every current stock was forecast


def test_all_current_stocks_are_forecast(monkeypatch, tmp_path, capsys):
    out, today = _run(monkeypatch, tmp_path, capsys, stale=set())
    assert f"[dry-run] {today}: 4 tickers, late rows 0" in out
