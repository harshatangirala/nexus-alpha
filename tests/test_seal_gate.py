"""The evaluation gate: closed by default, opened only by open_for_evaluation() after the owner's unlock."""

import pandas as pd
import pytest

from nexus_alpha import seal
from nexus_alpha.data import bars as B, etf_bars as EB, news_select as NS, sec as S


def test_gate_is_closed_by_default():
    assert not seal.is_open()
    assert seal.read_limit_date() == seal.SEAL_DATE
    assert seal.read_limit_utc() == seal.SEAL_UTC
    with pytest.raises(seal.HoldoutSealed):
        seal.require_open("test")


def test_open_requires_the_owner_key(monkeypatch):
    monkeypatch.delenv("NEXUS_HOLDOUT_CONFIG", raising=False)
    monkeypatch.delenv("NEXUS_HOLDOUT_UNLOCK", raising=False)
    with pytest.raises(seal.HoldoutSealed):
        seal.open_for_evaluation("pytest")
    assert not seal.is_open()


def test_readers_never_return_sealed_rows_while_closed():
    t = B.available_tickers()[0]
    assert B.read_daily(t).date.max() < seal.SEAL_DATE
    assert pd.Timestamp(B.read_hourly(t).timestamp.max()) < seal.SEAL_UTC
    if "SPY" in EB.available():
        assert EB.read_daily("SPY").date.max() < seal.SEAL_DATE


def test_beyond_seal_loaders_refuse_while_closed():
    with pytest.raises(seal.HoldoutSealed):
        NS.load_index(dev_only=False)
    with pytest.raises(seal.HoldoutSealed):
        S.load_events(dev_only=False)
