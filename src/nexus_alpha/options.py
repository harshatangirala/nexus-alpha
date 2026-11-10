"""Option-implied volatility from a listed option chain (E26, pre-registered in docs/V3_RESEARCH.md).

The chain comes from CBOE's delayed quotes through OpenBB, which runs in its own environment
(scripts/openbb/options_chains.py writes a parquet; this module only reads tables, so the project environment never
imports OpenBB). Definitions fixed by the E26 pre-registration:

* Contracts: a positive bid and ask, and an implied volatility in (0.01, 5.0).
* ATM IV of an expiration (at least 2 calendar days away): the mean of the call and put implied volatilities at the
  listed strike closest to the underlying price, or the one that exists.
* IV30: linear interpolation in total variance (iv^2 x days) between the two expirations that bracket 30 calendar days;
  the nearest expiration if all are beyond 30 days, the longest if all are shorter.
* For h sessions, sigma_h = IV30 x sqrt(h / 252). P(sideways) under a zero-drift normal model with the label rule's
  threshold tau: Phi(tau / sigma_h) - Phi(-tau / sigma_h).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

MIN_DTE, TARGET_DAYS = 2, 30
IV_RANGE = (0.01, 5.0)
COLS = ["expiration", "dte", "strike", "option_type", "bid", "ask", "implied_volatility", "underlying_price"]


def clean(chain: pd.DataFrame) -> pd.DataFrame:
    c = chain[[x for x in COLS if x in chain.columns]].copy()
    c["option_type"] = c.option_type.astype(str).str.lower().str[0]           # "call"/"put" -> "c"/"p"
    ok = ((c.bid > 0) & (c.ask > 0) & c.implied_volatility.between(*IV_RANGE, inclusive="neither")
          & (c.dte >= MIN_DTE) & c.underlying_price.gt(0))
    return c[ok]


def atm_term_structure(chain: pd.DataFrame) -> pd.DataFrame:
    """One row per expiration: days to expiry and ATM implied volatility."""
    c = clean(chain)
    rows = []
    for (exp, dte), g in c.groupby(["expiration", "dte"]):
        spot = float(g.underlying_price.iloc[0])
        k = g.strike.iloc[(g.strike - spot).abs().argsort().iloc[0]]
        at = g[g.strike == k]
        ivs = [float(at[at.option_type == t].implied_volatility.mean()) for t in ("c", "p")
               if (at.option_type == t).any()]
        if ivs:
            rows.append({"expiration": exp, "dte": int(dte), "strike": float(k), "atm_iv": float(np.mean(ivs))})
    return pd.DataFrame(rows, columns=["expiration", "dte", "strike", "atm_iv"]).sort_values("dte").reset_index(drop=True)


def iv30(term: pd.DataFrame) -> float:
    """30-day ATM implied volatility by linear interpolation in total variance."""
    if term is None or not len(term):
        return float("nan")
    t = term.sort_values("dte")
    d, v = t.dte.to_numpy(float), t.atm_iv.to_numpy(float)
    if d[0] >= TARGET_DAYS:
        return float(v[0])
    if d[-1] <= TARGET_DAYS:
        return float(v[-1])
    j = int(np.searchsorted(d, TARGET_DAYS))
    d0, d1, w0, w1 = d[j - 1], d[j], v[j - 1] ** 2 * d[j - 1], v[j] ** 2 * d[j]
    w = w0 + (w1 - w0) * (TARGET_DAYS - d0) / (d1 - d0)
    return float(math.sqrt(w / TARGET_DAYS))


def p_sideways(iv: float, tau: float, sessions: int) -> float:
    """P(|R_h| <= tau) under a zero-drift normal model with sigma_h = IV30 sqrt(h / 252)."""
    if not (np.isfinite(iv) and iv > 0 and tau > 0):
        return float("nan")
    s = iv * math.sqrt(sessions / 252.0)
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))       # noqa: E731
    return phi(tau / s) - phi(-tau / s)


def implied_move(iv: float, sessions: int) -> float:
    """One-standard-deviation move over h sessions implied by an annualised volatility."""
    return float(iv * math.sqrt(sessions / 252.0)) if np.isfinite(iv) else float("nan")


def summarize(chain: pd.DataFrame) -> dict:
    """Per-stock summary used by the nightly snapshot and the dashboard."""
    c = clean(chain)
    term = atm_term_structure(chain)
    oi = chain.get("open_interest")
    out = {"underlying_price": float(c.underlying_price.iloc[0]) if len(c) else float("nan"),
           "iv30": iv30(term), "n_contracts": int(len(c)), "n_expirations": int(len(term)),
           "front_dte": int(term.dte.iloc[0]) if len(term) else None,
           "front_atm_iv": float(term.atm_iv.iloc[0]) if len(term) else float("nan")}
    if oi is not None and "option_type" in chain:
        t = chain.option_type.astype(str).str.lower().str[0]
        calls, puts = float(oi[t == "c"].sum()), float(oi[t == "p"].sum())
        out["put_call_oi"] = puts / calls if calls > 0 else float("nan")
    return out
