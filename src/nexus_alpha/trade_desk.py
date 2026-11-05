"""Trade Desk logic (the dashboard's main workflow): trade type -> model horizon, market/macro snapshot, screenshot check.

Pure functions so they can be tested without Streamlit. The model serving each request is the frozen v1.1 until v2 is
frozen; v1.1 has horizons 1D, 2D, 5D, 10D, 20D, 1M, so a swing of N days is served by the nearest v1.1 horizon and is
labelled as such, and 3M/6M/1Y are reported as not yet available rather than extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

V11_HORIZONS = {"1D": 1, "2D": 2, "5D": 5, "10D": 10, "20D": 20, "1M": 21}
LONG_TERM = {"4 weeks": 20, "3 months": 63, "6 months": 126, "1 year": 252}


@dataclass
class HorizonChoice:
    requested_sessions: int
    label: str                # what the user asked for, in words
    served_by: str | None     # model horizon key, or None if no model covers it
    exact: bool
    note: str


def choose_horizon(trade_type: str, swing_days: int | None = None, long_term: str | None = None,
                   available: dict[str, int] = V11_HORIZONS) -> HorizonChoice:
    if trade_type == "Intraday":
        n, label = 1, "intraday (enter at the open, exit at the same session's close)"
    elif trade_type == "Swing":
        n = int(swing_days or 5)
        if not 2 <= n <= 20:
            raise ValueError("swing length must be 2-20 trading days")
        label = f"swing of {n} trading days (enter at the next open, exit at the close of day {n})"
    elif trade_type == "Long-term":
        n = LONG_TERM[long_term or "4 weeks"]
        label = f"long-term, {long_term} ({n} sessions)"
    else:
        raise ValueError(trade_type)
    exact = [k for k, v in available.items() if v == n]
    if exact:
        return HorizonChoice(n, label, exact[0], True, "served by a model horizon of exactly this length")
    if n > max(available.values()) * 1.25:
        return HorizonChoice(n, label, None, False,
                             f"no deployed model covers {n} sessions; the v2 model adds 3M/6M/1Y (not yet frozen)")
    k = min(available, key=lambda x: abs(np.log(available[x] / n)))
    return HorizonChoice(n, label, k, False, f"nearest deployed horizon is {k} ({available[k]} sessions): "
                                             f"read it as an approximation for {n} sessions")


# ------------------------------------------------------------------------------------------------ context
def percentile_vs_history(value: float, history: pd.Series) -> float | None:
    h = history.dropna()
    if not np.isfinite(value) or len(h) < 100:
        return None
    return float((h < value).mean() * 100)


def context_table(latest: pd.Series, history: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for col, name in labels.items():
        if col not in latest or pd.isna(latest[col]):
            continue
        v = float(latest[col])
        rows.append({"indicator": name, "value": v,
                     "percentile vs 2003-2022": percentile_vs_history(v, history[col]) if col in history else None})
    return pd.DataFrame(rows)


CONTEXT_LABELS = {
    "mkt_spy_ret21": "S&P 500 (SPY) 1-month log return", "mkt_spy_ret252": "S&P 500 12-month log return",
    "mkt_spy_dd252": "S&P 500 drawdown from 12-month high", "mkt_spy_sma200": "S&P 500 vs 200-day average",
    "mkt_spy_rv20": "S&P 500 realised volatility (20d, annualised)", "mkt_vix_log": "VIX (log level)",
    "mkt_vix_term": "VIX term structure log(VIX3M/VIX)", "mkt_term_10y_3m": "Treasury curve 10y - 3m (pct pts)",
    "mkt_y10y": "10-year Treasury yield (%)", "mkt_dxy_ret63": "US dollar index 3-month log return",
    "mkt_wti_ret63": "WTI crude 3-month log return", "mkt_hyg_ief_rel21": "High yield vs Treasuries, 1-month relative",
    "mkt_sector_breadth50": "Share of sector ETFs above 50-day average",
}


# ------------------------------------------------------------------------------------------------ screenshots
def check_legend(bars: pd.DataFrame, legend: dict, tolerance_pct: float = 0.5, date_col: str = "date") -> dict:
    """Find the bar whose OHLC matches the legend values read from a screenshot (by Gemini or typed by the user).

    Returns the best-matching bar, its maximum percentage deviation, and a verdict: 'match' (all four within tolerance),
    'basis_mismatch' (a constant ratio off: e.g. a dividend-adjusted chart), or 'no_match'. Only the last 400 bars are
    searched (screenshots show recent data)."""
    need = [k for k in ("open", "high", "low", "close") if legend.get(k) not in (None, "")]
    if len(need) < 2:
        return {"verdict": "insufficient", "detail": "fewer than two legend values were read"}
    b = bars.tail(400).reset_index(drop=True)
    want = np.array([float(legend[k]) for k in need])
    got = b[need].to_numpy(float)
    dev = np.abs(got / want - 1) * 100
    worst = dev.max(axis=1)
    i = int(np.nanargmin(worst))
    out = {"matched_bar": str(b[date_col].iloc[i]), "max_dev_pct": float(worst[i]), "fields": need,
           "bars_from_end": int(len(b) - 1 - i)}
    if worst[i] <= tolerance_pct:
        out["verdict"] = "match"
        return out
    # a constant factor across all fields of one bar = same bar on a different price basis (e.g. dividend-adjusted)
    ratio = got / want
    spread = np.ptp(ratio, axis=1)
    plausible = np.abs(ratio.mean(axis=1) - 1) < 0.2
    spread[~plausible] = np.inf
    cand = np.where(spread < 0.002)[0]
    j = int(cand[np.argmin(np.abs(ratio[cand].mean(axis=1) - 1))]) if len(cand) else int(np.nanargmin(spread))
    if spread[j] < 0.002:
        out.update({"verdict": "basis_mismatch", "matched_bar": str(b[date_col].iloc[j]),
                    "bars_from_end": int(len(b) - 1 - j), "max_dev_pct": float(worst[j]),
                    "detail": f"all values differ by the same factor {ratio[j].mean():.4f} "
                              "(adjusted chart or different share basis)"})
    else:
        out["verdict"] = "no_match"
    return out
