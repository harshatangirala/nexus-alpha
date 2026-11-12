"""Side-by-side comparison of 1 to 5 stocks (data layer of the dashboard's Compare tab; no Streamlit here).

Everything comes from files the nightly forward test already writes; nothing is re-estimated:
* the logged forecasts of the frozen models for one session (`shadow/<version>/forecasts/<session>*.csv`);
* the live bar cache (`artifacts/live_cache/<date>/<T>_daily.pkl`, Yahoo daily bars);
* every stock's live feature row (`artifacts/live_features/<session>/v2_x.parquet`), for universe percentiles;
* the live fundamentals panel (`artifacts/fundamentals/live_panel.parquet`, SEC XBRL ranks);
* the private option-implied summary (`artifacts/iv_snapshots/<session>.parquet`), when the nightly snapshot ran.

The layout follows the TriFrame dashboard's comparison view (risk ranking, risk map, side-by-side distributions),
extended to NEXUS's models. Display only: it never changes a forecast, a probability or the risk engine.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

MAX_STOCKS = 5
VERSIONS = ("v1.1", "v2", "v3_fund", "v3_alpha")
SHADOW = paths.REPO_ROOT / "shadow"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# validated evidence by horizon, quoted from docs/V2_RESULTS.md and docs/V3_RESEARCH.md (used only for captions)
RANK_IC_NOTE = ("Cross-sectional rank is the part with evidence (validation rank IC about 0.02-0.03 at 5-20 sessions). "
                "Single-stock direction did not hold up on the 2020-22 test years (up-vs-down AUROC 0.49-0.51), so the "
                "leans are information, not calls.")

# factor profile: (label, source column, higher-is-better sign). Percentiles are within tonight's universe.
TECH_FACTORS = [("Mom 3M", "d_ret63", 1), ("Mom 1M", "d_ret21", 1), ("Trend", "d_close_sma50", 1),
                ("RSI", "d_rsi14", 1), ("Near high", "d_dist_high252", 1), ("Low vol", "d_rv20", -1)]
FUND_FACTORS = [("Value", "fr_earnings_yield", 1), ("Quality", "fr_roic", 1),
                ("Growth", "fr_revenue_growth", 1), ("Low leverage", "fr_debt_to_equity", -1)]


def sessions(version: str = "v1.1") -> list[str]:
    ch = SHADOW / version / "chain.csv"
    if not ch.exists():
        return []
    return sorted(set(pd.read_csv(ch, usecols=["asof_session"]).asof_session.astype(str)))


def latest_session() -> str | None:
    s = sessions("v1.1")
    return s[-1] if s else None


def forecasts(session: str, tickers: list[str] | None = None, versions=VERSIONS) -> dict[str, pd.DataFrame]:
    """Logged rows of each version for one session (the base file plus any same-session additions)."""
    out = {}
    for v in versions:
        files = sorted((SHADOW / v / "forecasts").glob(f"{session}*.csv"))
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        d = d.drop_duplicates(["ticker", "horizon"], keep="first")      # first write wins, as in the forward test
        out[v] = d if tickers is None else d[d.ticker.isin(tickers)].copy()
    return out


def rank_percentile(session: str, version: str, horizon: str) -> pd.Series:
    """Percentile (0-100) of each stock's logged rank_score among all stocks logged that session and horizon."""
    d = forecasts(session, None, (version,)).get(version)
    if d is None or "rank_score" not in d:
        return pd.Series(dtype=float)
    d = d[(d.horizon == horizon) & d.rank_score.notna()]
    return (d.set_index("ticker").rank_score.rank(pct=True) * 100).rename(f"{version} {horizon}")


def lean_matrix(fc: pd.DataFrame, tickers: list[str], horizons: list[str]) -> pd.DataFrame:
    """P(up) - P(down) by stock (rows) and horizon (columns)."""
    d = fc[fc.ticker.isin(tickers) & fc.horizon.isin(horizons)].assign(lean=lambda x: x.p_up - x.p_down)
    m = d.pivot_table(index="ticker", columns="horizon", values="lean", aggfunc="first")
    return m.reindex(index=[t for t in tickers if t in m.index], columns=[h for h in horizons if h in m.columns])


def _cache_dirs() -> list[Path]:
    root = paths.ARTIFACTS / "live_cache"
    return sorted((p for p in root.glob("*") if p.is_dir() and _DATE.fullmatch(p.name)), reverse=True) if root.exists() else []


def bars(tickers: list[str], session: str | None = None) -> dict[str, pd.DataFrame]:
    """Newest cached Yahoo daily bars per ticker (date-indexed, up to the session when one is given)."""
    out = {}
    dirs = _cache_dirs()
    for t in tickers:
        for d in dirs:
            f = d / f"{t}_daily.pkl"
            if f.exists():
                b = pd.read_pickle(f)
                b = b.assign(date=pd.to_datetime(b.date)).set_index("date").sort_index()
                if session is not None:
                    b = b[b.index <= pd.Timestamp(session)]
                if len(b):
                    out[t] = b
                    break
    return out


def close_panel(bars_by: dict[str, pd.DataFrame], sessions_back: int | None = None) -> pd.DataFrame:
    p = pd.DataFrame({t: b.close for t, b in bars_by.items()}).sort_index().dropna(how="all")
    return p.iloc[-sessions_back:] if sessions_back else p


def normalized(panel: pd.DataFrame) -> pd.DataFrame:
    """Start = 100 at each stock's first valid close in the window."""
    first = panel.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    return panel / first * 100


def risk_stats(bars_by: dict[str, pd.DataFrame], window: int = 252) -> pd.DataFrame:
    rows = []
    for t, b in bars_by.items():
        c = b.close.dropna()
        r = np.log(c).diff().dropna()
        w = c.iloc[-window:]
        dd = w / w.cummax() - 1
        atr = (pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1)
               .max(axis=1).rolling(14).mean())
        rows.append({"ticker": t, "as of": str(c.index[-1].date()), "close": float(c.iloc[-1]),
                     "1-day change": float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else math.nan,
                     "1-month return": float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) > 22 else math.nan,
                     "1-year return": float(c.iloc[-1] / c.iloc[-253] - 1) if len(c) > 253 else math.nan,
                     "volatility 20D (a year)": float(r.iloc[-20:].std() * math.sqrt(252)) if len(r) >= 20 else math.nan,
                     "volatility 1Y (a year)": float(r.iloc[-252:].std() * math.sqrt(252)) if len(r) >= 60 else math.nan,
                     "max drawdown (window)": float(dd.min()), "drawdown now": float(dd.iloc[-1]),
                     "ATR14 % of price": float(atr.iloc[-1] / c.iloc[-1]) if len(c) > 15 else math.nan})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


def correlation(bars_by: dict[str, pd.DataFrame], sessions_back: int = 126) -> pd.DataFrame:
    p = close_panel(bars_by)
    return np.log(p).diff().iloc[-sessions_back:].corr()


def _features(session: str) -> pd.DataFrame | None:
    f = paths.ARTIFACTS / "live_features" / session / "v2_x.parquet"
    return pd.read_parquet(f) if f.exists() else None


def factor_profile(session: str, tickers: list[str]) -> pd.DataFrame:
    """Percentile (0-100, higher = more of the named quality) within tonight's universe, technical and fundamental."""
    cols = {}
    x = _features(session)
    if x is not None:
        x = x.drop_duplicates("ticker").set_index("ticker")
        for label, col, sign in TECH_FACTORS:
            if col in x:
                cols[label] = (sign * x[col]).rank(pct=True) * 100
    fp = paths.ARTIFACTS / "fundamentals" / "live_panel.parquet"
    if fp.exists():
        f = pd.read_parquet(fp).drop_duplicates("ticker").set_index("ticker")
        for label, col, sign in FUND_FACTORS:
            if col in f:
                cols[label] = (f[col] if sign > 0 else 1 - f[col]) * 100     # fr_* are already universe ranks in (0, 1]
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).reindex([t for t in tickers])


def iv_summary(session: str, tickers: list[str]) -> pd.DataFrame | None:
    f = paths.ARTIFACTS / "iv_snapshots" / f"{session}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    return d[d.ticker.isin(tickers)].set_index("ticker") if "ticker" in d else None


def verdict(stats: pd.DataFrame, lean: pd.DataFrame | None, ranks: pd.Series | None) -> str:
    """One paragraph in plain language. Risk and relative rank are stated; direction is not called."""
    if stats is None or stats.empty:
        return "No cached bars for the selected stocks."
    if len(stats) == 1:
        t = stats.index[0]
        s = stats.iloc[0]
        return (f"**{t}**: volatility {s['volatility 20D (a year)']:.0%} a year (20 sessions), "
                f"{s['drawdown now']:+.1%} from its 1-year high. " + RANK_IC_NOTE)
    v = stats["volatility 20D (a year)"].dropna().sort_values()
    txt = (f"**Most volatile: {v.index[-1]}** ({v.iloc[-1]:.0%} a year over 20 sessions). "
           f"**Least volatile: {v.index[0]}** ({v.iloc[0]:.0%}). ")
    dd = stats["drawdown now"].sort_values()
    txt += f"Deepest below its 1-year high: **{dd.index[0]}** ({dd.iloc[0]:+.1%}). "
    if ranks is not None and len(ranks.dropna()):
        r = ranks.dropna().sort_values()
        txt += (f"Highest cross-sectional rank ({ranks.name}): **{r.index[-1]}** (percentile {r.iloc[-1]:.0f}); "
                f"lowest: **{r.index[0]}** ({r.iloc[0]:.0f}). ")
    return txt + RANK_IC_NOTE
