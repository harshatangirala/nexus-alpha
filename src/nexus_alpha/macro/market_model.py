"""E19 deployed market-direction read (decision D-031): logistic M2 on SPY's own v2 label, at the horizons that passed
the keep decision and the D-015 direction gate on validation (20D, 3M).

Features are the E19 M2 set: SPY technicals and the other date-level market features of the v2 frame, without the
calendar index `mkt_n` (D-030). Six of them (`mkt_ret1/5/21`, `mkt_rv20`, `mkt_breadth_sma50`, `mkt_dispersion20`) are
computed in development from the cross-section of the 979-stock source universe (`indicators/market.py`); live, the same
arithmetic runs on Yahoo daily closes of the same tickers (`universe_context`). Parity against development values is
checked by `scripts/verify/e19_live_parity.py`.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

KIND = "market"
UNIVERSE_COLS = ["mkt_ret1", "mkt_ret5", "mkt_ret21", "mkt_rv20", "mkt_breadth_sma50", "mkt_dispersion20"]


def universe_tickers() -> list[str]:
    from ..data import bars as B
    return list(B.available_tickers())


def universe_context(closes: pd.DataFrame) -> pd.DataFrame:
    """Same arithmetic as indicators.market.market_context, from a (date x ticker) close panel. Each ticker's returns and
    50-session average use its own non-missing closes, as in development (per-ticker series, then pivot)."""
    r_cols, a_cols = {}, {}
    for t in closes.columns:
        c = closes[t].dropna().astype(float)
        if c.empty:
            continue
        r_cols[t] = c.pct_change()
        a_cols[t] = (c > c.rolling(50).mean()).astype(float).where(c.rolling(50).count() == 50)
    r = pd.DataFrame(r_cols).sort_index()
    a = pd.DataFrame(a_cols).reindex(r.index)
    ew = r.mean(axis=1)
    lvl = (1 + ew.fillna(0)).cumprod()
    out = pd.DataFrame(index=r.index)
    out["mkt_ret1"] = ew
    out["mkt_ret5"] = lvl / lvl.shift(5) - 1
    out["mkt_ret21"] = lvl / lvl.shift(21) - 1
    out["mkt_rv20"] = ew.rolling(20).std(ddof=0) * np.sqrt(252)
    out["mkt_breadth_sma50"] = a.mean(axis=1)
    out["mkt_dispersion20"] = r.std(axis=1).rolling(20).mean()
    out["n_tickers"] = r.notna().sum(axis=1)
    out.index.name = "date"
    return out


def yahoo_closes(tickers: list[str], start: str, end: str | None = None, chunk: int = 80) -> pd.DataFrame:
    """Daily closes (split-adjusted, not dividend-adjusted, as in live stock inference) for many tickers, in chunks."""
    import time

    import yfinance as yf
    from ..inference.pipeline import yahoo_symbol
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        ys = {yahoo_symbol(t): t for t in part}
        for attempt in range(3):
            try:
                d = yf.download(list(ys), start=start, end=end, interval="1d", auto_adjust=False, actions=False,
                                progress=False, threads=True, group_by="column")
                break
            except Exception:
                time.sleep(20 * (attempt + 1))
        else:
            continue
        if d is None or d.empty:
            continue
        c = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]].rename(columns={"Close": part[0]})
        c = c.rename(columns=ys)
        frames.append(c)
        time.sleep(2)
    out = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    di = pd.DatetimeIndex(out.index)
    out.index = (di.tz_localize(None) if di.tz is not None else di).normalize()
    return out.sort_index()


def live_features(days: int = 800, universe_days: int = 200) -> tuple[pd.DataFrame, dict]:
    """Date-level M2 features from Yahoo for recent sessions (index = session date), plus data-quality notes."""
    from .context import live_date_features
    f = live_date_features(days)
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=universe_days)).strftime("%Y-%m-%d")
    uni = universe_tickers()
    closes = yahoo_closes(uni, start)
    closes = closes.reindex(f.index)                # SPY's completed sessions only (today's forming bar is dropped)
    u = universe_context(closes)
    f = f.join(u[UNIVERSE_COLS])
    last_n = int(u.n_tickers.iloc[-1]) if len(u) else 0
    return f, {"universe_tickers": len(uni), "universe_with_data_last_session": last_n}


class MarketModel:
    def __init__(self, bundle_dir: Path):
        self.dir = Path(bundle_dir)
        self.meta = json.loads((self.dir / "bundle_meta.json").read_text(encoding="utf-8"))
        with open(self.dir / "market_model.pkl", "rb") as fh:
            self.models = pickle.load(fh)

    @property
    def horizons(self) -> dict:
        return self.meta["horizons"]

    def predict(self, row: pd.Series) -> dict:
        out = {}
        for h, m in self.models.items():
            x = row.reindex(m["cols"]).astype(float)
            missing = float(x.isna().mean())
            z = ((x.fillna(m["median"]) - m["mu"]) / m["sd"]).clip(-8, 8).to_numpy(np.float64)[None, :]
            p = m["model"].predict_proba(z)[0]
            out[h] = {"p_down": float(p[0]), "p_side": float(p[1]), "p_up": float(p[2]), "missing_share": missing,
                      **self.meta["evidence"][h]}
        return out


def latest_bundle() -> Path | None:
    cands = sorted((paths.ARTIFACTS / "bundles").glob("*_e19_refit_*"), reverse=True)
    return cands[0] if cands else None
