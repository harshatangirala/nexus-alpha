"""Event-driven trade simulator and portfolio accounting (docs/VALIDATION_AND_BACKTESTING.md).

Execution conventions (frozen, D-010):
* decision before the open of session t+1; market entry at that session's open;
* cancel (no trade) if the open gaps beyond the max-chase price or already beyond the stop;
* from the second session, an open beyond the stop exits AT THE OPEN (gap-through, X083), never at the stop price;
  an open beyond the target exits at the open (favourable gap);
* inside a bar, if both stop and target are touched the stop is assumed first (ambiguous_bar flagged);
* otherwise exit at the close of session t+h (time exit, X071);
* costs: round-trip bps of entry value on every filled trade, plus borrow on shorts for the holding sessions.
Portfolio: each trade gets weight w = min(risk_pct / stop_distance_fraction, max_position_pct) of equity at entry; new
trades are skipped while gross exposure would exceed the cap. Daily portfolio return = sum of w * position daily return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation.stats import block_bootstrap, deflated_sharpe, month_blocks, probabilistic_sharpe, sharpe
from .costs import CostModel


@dataclass
class Order:
    sample_id: str
    ticker: str
    k: int                      # daily index of the as-of session
    side: int                   # +1 long, -1 short
    stop: float
    target: float
    horizon: int
    chase: float | None = None
    policy: str = ""


def simulate(o: Order, d: pd.DataFrame, cost: CostModel) -> dict:
    O, H, L, C = (d[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    dates = d.date.to_numpy()
    e = o.k + 1
    base = {"sample_id": o.sample_id, "ticker": o.ticker, "side": o.side, "policy": o.policy}
    if e + o.horizon - 1 >= len(d):
        return {**base, "fill_status": "NOT_SIMULATED", "reason": "path beyond sealed data"}
    op = O[e]
    if o.chase is not None and ((o.side > 0 and op > o.chase) or (o.side < 0 and op < o.chase)):
        return {**base, "fill_status": "CANCELLED", "reason": "open beyond max chase"}
    if (o.side > 0 and op <= o.stop) or (o.side < 0 and op >= o.stop):
        return {**base, "fill_status": "CANCELLED", "reason": "open beyond stop"}
    entry, exit_px, reason, ambiguous = op, None, "time", False
    path = []                                             # (date, close-or-exit) for daily returns
    for j in range(e, e + o.horizon):
        if j > e:
            if (o.side > 0 and O[j] <= o.stop) or (o.side < 0 and O[j] >= o.stop):
                exit_px, reason = O[j], "stop_gap"
            elif (o.side > 0 and O[j] >= o.target) or (o.side < 0 and O[j] <= o.target):
                exit_px, reason = O[j], "target_gap"
        if exit_px is None:
            hit_s = L[j] <= o.stop if o.side > 0 else H[j] >= o.stop
            hit_t = H[j] >= o.target if o.side > 0 else L[j] <= o.target
            if hit_s and hit_t:
                exit_px, reason, ambiguous = o.stop, "stop", True
            elif hit_s:
                exit_px, reason = o.stop, "stop"
            elif hit_t:
                exit_px, reason = o.target, "target"
        if exit_px is not None:
            path.append((dates[j], exit_px))
            break
        path.append((dates[j], C[j]))
    if exit_px is None:
        exit_px = C[e + o.horizon - 1]
    held = len(path)
    gross = o.side * (exit_px / entry - 1)
    c = cost.round_trip_bps / 1e4 + (cost.borrow_annual * held / 252 if o.side < 0 else 0.0)
    risk = abs(entry - o.stop) / entry
    prev = entry
    daily = []
    for i, (dt, px) in enumerate(path):
        r = o.side * (px / prev - 1)
        if i == 0:
            r -= cost.round_trip_bps / 2e4
        if i == len(path) - 1:
            r -= cost.round_trip_bps / 2e4 + (cost.borrow_annual * held / 252 if o.side < 0 else 0.0)
        daily.append((dt, r))
        prev = px
    return {**base, "fill_status": "FILLED", "entry_date": dates[e], "entry": entry, "exit": exit_px, "exit_reason": reason,
            "ambiguous_bar": ambiguous, "holding_sessions": held, "gross_return": gross, "net_return": gross - c,
            "risk_fraction": risk, "net_R": (gross - c) / risk if risk > 0 else np.nan,
            "mfe_note": "not computed", "daily": daily}


def portfolio(trades: pd.DataFrame, calendar_dates: pd.DatetimeIndex, risk_pct: float = 0.005,
              max_position_pct: float = 0.10, max_gross: float = 1.0) -> tuple[pd.Series, pd.DataFrame]:
    """Daily portfolio returns from filled trades with gross-exposure cap (trades processed in entry order)."""
    f = trades[trades.fill_status == "FILLED"].sort_values("entry_date").copy()
    f["weight"] = np.minimum(risk_pct / f.risk_fraction.clip(lower=1e-4), max_position_pct)
    open_until: list[tuple] = []
    kept = []
    for _, t in f.iterrows():
        d0 = pd.Timestamp(t.entry_date)
        open_until = [(end, w) for end, w in open_until if end >= d0]
        gross = sum(w for _, w in open_until)
        if gross + t.weight > max_gross + 1e-12:
            continue
        end = pd.Timestamp(t.daily[-1][0])
        open_until.append((end, t.weight))
        kept.append(t)
    k = pd.DataFrame(kept)
    ret = pd.Series(0.0, index=calendar_dates)
    expo = pd.Series(0.0, index=calendar_dates)
    for _, t in k.iterrows():
        for dt, r in t.daily:
            dt = pd.Timestamp(dt)
            if dt in ret.index:
                ret[dt] += t.weight * r
                expo[dt] += t.weight
    k_out = k.drop(columns=["daily"]) if len(k) else k
    return ret.rename("portfolio_return"), k_out.assign(exposure_mean=float(expo.mean()) if len(expo) else np.nan)


def performance(ret: pd.Series, trades: pd.DataFrame | None = None, n_trials: int = 1, trial_sr_var: float = 0.0) -> dict:
    r = ret.fillna(0.0)
    eq = (1 + r).cumprod()
    years = len(r) / 252
    dd = eq / eq.cummax() - 1
    neg = r[r < 0]
    out = {"n_days": int(len(r)), "total_return": float(eq.iloc[-1] - 1) if len(eq) else np.nan,
           "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else np.nan,
           "vol_annual": float(r.std(ddof=1) * np.sqrt(252)), "sharpe": sharpe(r.values),
           "sortino": float(r.mean() / neg.std(ddof=1) * np.sqrt(252)) if len(neg) > 2 else np.nan,
           "max_drawdown": float(dd.min()), "calmar": float((eq.iloc[-1] ** (1 / years) - 1) / abs(dd.min())) if dd.min() < 0 and years > 0 else np.nan,
           "psr_vs_0": probabilistic_sharpe(r.values),
           "exposure_days_share": float((r != 0).mean())}
    if n_trials > 1 and trial_sr_var > 0:
        out["deflated_sharpe"] = deflated_sharpe(r.values, n_trials, trial_sr_var)
    rr = r.to_numpy()
    out["sharpe_ci95_month_blocks"] = block_bootstrap(lambda x: np.mean(x) / (np.std(x, ddof=1) + 1e-12) * np.sqrt(252),
                                                      (rr,), month_blocks(r.index), n_boot=500)["ci95"]
    if trades is not None and len(trades):
        t = trades[trades.fill_status == "FILLED"]
        wins = t.net_return > 0
        out.update({"trades": int(len(t)), "win_rate": float(wins.mean()),
                    "profit_factor": float(t.net_return[wins].sum() / max(1e-12, -t.net_return[~wins].sum())),
                    "expectancy_R": float(t.net_R.mean()), "avg_holding_sessions": float(t.holding_sessions.mean()),
                    "long_trades": int((t.side > 0).sum()), "short_trades": int((t.side < 0).sum()),
                    "long_mean_net": float(t.net_return[t.side > 0].mean()) if (t.side > 0).any() else np.nan,
                    "short_mean_net": float(t.net_return[t.side < 0].mean()) if (t.side < 0).any() else np.nan,
                    "stop_exits": int((t.exit_reason.isin(["stop", "stop_gap"])).sum()),
                    "gap_through_stops": int((t.exit_reason == "stop_gap").sum()),
                    "ambiguous_bars": int(t.ambiguous_bar.sum()),
                    "turnover_trades_per_year": float(len(t) / max(years, 1e-9))})
    return out
