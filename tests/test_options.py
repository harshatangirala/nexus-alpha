"""Option-implied volatility (E26 definitions): ATM selection, total-variance interpolation, implied probabilities."""
import math

import numpy as np
import pandas as pd
import pytest

from nexus_alpha import options as O


def _chain(spot=100.0, exp_ivs=((10, 0.30), (45, 0.40)), skew=0.0):
    rows = []
    for dte, iv in exp_ivs:
        for k in (90.0, 95.0, 100.0, 105.0, 110.0):
            for t in ("call", "put"):
                rows.append({"expiration": f"2026-12-{dte:02d}" if dte < 32 else f"2027-01-{dte - 31:02d}", "dte": dte,
                             "strike": k, "option_type": t, "bid": 1.0, "ask": 1.2,
                             "implied_volatility": iv + skew * (k - spot) / 100, "underlying_price": spot,
                             "open_interest": 100 if t == "call" else 150})
    return pd.DataFrame(rows)


def test_atm_is_the_strike_closest_to_spot_and_averages_call_and_put():
    t = O.atm_term_structure(_chain(spot=101.0))
    assert list(t.dte) == [10, 45] and set(t.strike) == {100.0}
    assert t.atm_iv.tolist() == pytest.approx([0.30, 0.40])


def test_iv30_interpolates_total_variance():
    t = O.atm_term_structure(_chain())
    w = 0.30 ** 2 * 10 + (0.40 ** 2 * 45 - 0.30 ** 2 * 10) * (30 - 10) / (45 - 10)
    assert O.iv30(t) == pytest.approx(math.sqrt(w / 30))
    assert O.iv30(O.atm_term_structure(_chain(exp_ivs=((40, 0.5), (60, 0.6))))) == pytest.approx(0.5)   # all > 30 days
    assert O.iv30(O.atm_term_structure(_chain(exp_ivs=((5, 0.2), (12, 0.25))))) == pytest.approx(0.25)   # all < 30
    assert np.isnan(O.iv30(pd.DataFrame(columns=["dte", "atm_iv"])))


def test_bad_quotes_and_expiring_contracts_are_dropped():
    c = _chain(exp_ivs=((1, 0.9), (10, 0.3)))
    c.loc[c.strike == 100.0, "bid"] = 0.0          # no bid at the money: the next-closest strike is used
    t = O.atm_term_structure(c)
    assert list(t.dte) == [10] and t.strike.iloc[0] in (95.0, 105.0)


def test_p_sideways_matches_the_normal_model():
    iv, tau, h = 0.32, 0.03, 5
    s = iv * math.sqrt(h / 252)
    expect = math.erf(tau / s / math.sqrt(2))
    assert O.p_sideways(iv, tau, h) == pytest.approx(expect)
    assert O.p_sideways(iv, tau, 20) < O.p_sideways(iv, tau, 5)           # wider horizon, more likely to move
    assert np.isnan(O.p_sideways(float("nan"), tau, h))
    assert O.implied_move(0.32, 252) == pytest.approx(0.32)


def test_summary_reports_put_call_open_interest():
    s = O.summarize(_chain())
    assert s["put_call_oi"] == pytest.approx(1.5) and s["n_expirations"] == 2 and s["underlying_price"] == 100.0
