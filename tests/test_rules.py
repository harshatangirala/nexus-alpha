import json

import numpy as np
import pandas as pd
import pytest

from nexus_alpha.rag.sources import RULEBOOKS
from nexus_alpha.rules import detectors as D
from nexus_alpha.rules.catalog import build_catalog

from .conftest import synthetic_daily


def _bars(rows, start="2015-01-05"):
    from nexus_alpha.data.calendar import sessions
    s = sessions()
    s = s[s >= pd.Timestamp(start)][: len(rows)]
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"date": s, "open": o, "high": h, "low": l, "close": c, "volume": [1e6] * len(rows)})


def test_pivot_is_known_only_after_confirmation():
    hi = np.array([1, 2, 3, 9, 3, 2, 1, 2, 3], float)
    lo = hi - 0.5
    p = D.confirmed_pivots(hi, lo, k=3)
    # bar 3 is the swing high; it is confirmed at bar 6, not earlier
    assert np.isnan(p.sh_last.iloc[5]) and p.sh_last.iloc[6] == 9 and p.sh_last_idx.iloc[6] == 3


def test_pivots_and_patterns_ignore_future_bars():
    d = synthetic_daily(300, seed=11)
    t = 250
    p0 = D.confirmed_pivots(d.high.to_numpy(), d.low.to_numpy(), 3).iloc[: t + 1]
    c0 = D.candle_patterns(d).iloc[: t + 1]
    d2 = d.copy()
    d2.loc[t + 1:, ["open", "high", "low", "close"]] *= 1.7
    p1 = D.confirmed_pivots(d2.high.to_numpy(), d2.low.to_numpy(), 3).iloc[: t + 1]
    c1 = D.candle_patterns(d2).iloc[: t + 1]
    pd.testing.assert_frame_equal(p0, p1)
    pd.testing.assert_frame_equal(c0, c1)


def _decline_then(last_two):
    rows = [(100 - i, 100.5 - i, 98.5 - i, 99 - i) for i in range(25)]      # steady decline
    return _bars(rows + last_two)


def test_bullish_engulfing_and_near_miss():
    d = _decline_then([(76.0, 76.2, 74.8, 75.0), (74.9, 77.5, 74.5, 77.0)])  # bearish, then bullish body engulfing it
    p = D.candle_patterns(d)
    assert p.bull_engulfing.iloc[-1]
    miss = _decline_then([(76.0, 76.2, 74.8, 75.0), (75.2, 76.0, 74.9, 75.9)])  # body does not reach prior open
    assert not D.candle_patterns(miss).bull_engulfing.iloc[-1]


def test_hammer_requires_decline_context():
    hammer = (75.0, 75.3, 71.0, 75.2)                     # long lower wick, small body near the high
    assert D.candle_patterns(_decline_then([(76, 76.2, 75.5, 75.8), hammer])).hammer.iloc[-1]
    up = _bars([(50 + i, 51 + i, 49.5 + i, 50.8 + i) for i in range(25)] + [(75.0, 75.3, 71.0, 75.2)])
    assert not D.candle_patterns(up).hammer.iloc[-1]


def test_morning_star():
    star = [(78.0, 78.2, 74.0, 74.2),                     # long bearish
            (73.9, 74.3, 73.5, 74.0),                     # small body
            (74.2, 77.5, 74.0, 77.2)]                     # bullish close above the first body's midpoint (76.1)
    d = _bars([(100 - i, 100.5 - i, 98.5 - i, 99 - i) for i in range(25)] + star)
    assert D.candle_patterns(d).morning_star.iloc[-1]


def _weekly(n, drift, seed=0):
    rng = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(drift + rng.normal(0, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6,
                         "last_day": pd.date_range("2010-01-08", periods=n, freq="W-FRI")})


def test_weekly_regime_classes():
    assert D.weekly_regime(_weekly(20, 0.0))["regime"] == "UNKNOWN"
    up = D.weekly_regime(_weekly(120, 0.02, seed=1))
    dn = D.weekly_regime(_weekly(120, -0.02, seed=2))
    assert up["regime"] in ("BULL_TREND", "HIGH_VOL_DISORDERLY", "TRANSITION")
    assert dn["regime"] in ("BEAR_TREND", "HIGH_VOL_DISORDERLY", "TRANSITION")
    assert up["permission"] == D.WEEKLY_PERMISSION[up["regime"]]
    # a clean trend should be recognised as a trend for at least one seed
    assert any(D.weekly_regime(_weekly(120, 0.02, seed=s))["regime"] == "BULL_TREND" for s in range(6))
    assert any(D.weekly_regime(_weekly(120, -0.02, seed=s))["regime"] == "BEAR_TREND" for s in range(6))


def test_hourly_structure_break_trigger():
    rows = [(10, 10.2, 9.8, 10.0)] * 8 + [(10.0, 10.6, 9.95, 10.5)]
    hb = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    assert D.hourly_trigger(hb, "LONG")["trigger"] == "STRUCTURE_BREAK"
    assert D.hourly_trigger(hb, "SHORT")["trigger"] == "NO_TRIGGER"


def test_daily_setups_carry_valid_invalidation():
    d = synthetic_daily(500, seed=21, vol=0.015)
    for k in range(300, 480, 7):
        for s in D.daily_setups(d, k, {"permission": "REDUCED_CONFIRMATION_REQUIRED"}):
            if s["direction"] == "LONG":
                assert s["invalidation"] < s["close"], s
            else:
                assert s["invalidation"] > s["close"], s


def test_data_gates():
    d = synthetic_daily(260, seed=3)
    ev = {e.rule_id: e for e in D.data_gates(d, 250, hourly_bars_in_session=7)}
    assert ev["E003"].state == "FIRED" and ev["E007"].state == "FIRED"
    assert D.data_gates(d, 100, 7)[1].state == "BLOCKED"                     # warm-up
    assert {e.rule_id: e for e in D.data_gates(d, 250, 5)}["E001"].state == "BLOCKED"


def test_catalog_parses_all_rules_with_provenance():
    cat = build_catalog(RULEBOOKS["entry_rulebook_150"], RULEBOOKS["exit_rulebook_120"])
    ids = [r["rule_id"] for r in cat["rules"]]
    assert cat["n_rules"] == 270 and len(set(ids)) == 270
    assert ids[0] == "E001" and ids[149] == "E150" and ids[150] == "X001" and ids[-1] == "X120"
    assert all(r["definition"] for r in cat["rules"])
    lines = open(RULEBOOKS["entry_rulebook_150"], encoding="utf-8").read().splitlines()
    r = cat["rules"][40]
    assert lines[r["source_line"] - 1].startswith(f"### {r['rule_id']}")
    implemented = {r["rule_id"] for r in cat["rules"] if r["implementation"] == "executable_detector"}
    assert implemented == set(D.IMPLEMENTED)
