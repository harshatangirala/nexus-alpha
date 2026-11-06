"""Bar-derived evaluation rows for a list of (ticker, as-of) samples, built in batch per ticker.

Columns match the development frames (`artifacts/samples/s500_frame.parquet` and the v2 labels): structured
indicators, `x_*` visible-window features, v1 and v2 labels, `adv20_usd`, `close`, `n_daily_bars_before`,
`decision_utc`, per-view valid-bar counts and rule features. Chart triplets are rendered with the training renderer and
embedded with the frozen backbone in the parent process (batched), so images never accumulate in memory.
"""
from __future__ import annotations

import concurrent.futures as cf
import time

import numpy as np
import pandas as pd

VIEWS = ("W", "D", "H")


def continue_schedule(last_asof: dict, start: str, end: str | None = None, step: int = 10) -> pd.DataFrame:
    """Every `step`-th daily bar of each stock's own series after its last development as-of date (TriFrame's s500
    stepping, measured: 98.3% of development gaps are exactly 10 own bars). Keeps as-of dates in [start, end] that have
    a next session in the data (the decision is at 09:00 New York on t+1)."""
    from ..data import bars as B
    rows = []
    for t, last in last_asof.items():
        d = B.read_daily(t)
        dates = pd.DatetimeIndex(d.date)
        if pd.Timestamp(last) not in dates:
            continue
        k0 = int(dates.get_loc(pd.Timestamp(last)))
        for k in range(k0 + step, len(dates) - 1, step):
            a = dates[k]
            if a >= pd.Timestamp(start) and (end is None or a <= pd.Timestamp(end)):
                rows.append((t, a))
    s = pd.DataFrame(rows, columns=["ticker", "asof_date"])
    s["sample_id"] = s.ticker + ":" + s.asof_date.dt.strftime("%Y-%m-%d")
    return s


def _ticker_rows(args):
    """Worker: all bar-derived features and the rendered triplets of one ticker's samples."""
    ticker, asof_list = args
    from ..cv import render as R
    from ..data import bars as B
    from ..data.calendar import calendar
    from ..indicators import triframe_visible as FV
    from ..indicators.core import structured_features
    from ..rules import batch as RB
    from ..targets import make_labels
    from ..targets_v2 import HORIZONS_V2, make_labels_v2
    cal = calendar()
    early = set(cal.date[cal.early_close.astype(bool)])
    d = B.read_daily(ticker)
    h = R.rth_hourly(B.read_hourly(ticker), early)
    dates = pd.DatetimeIndex(d.date)
    asof = pd.DatetimeIndex(sorted(set(pd.to_datetime(asof_list)))).intersection(dates)
    if len(asof) == 0:
        return ticker, None, None, None
    k = dates.get_indexer(asof)
    f = structured_features(d, h, asof).reset_index(drop=True)
    ok = k >= FV.W - 1
    xv = pd.DataFrame(index=range(len(k)))
    if ok.any():
        xo = FV.from_series(d.open, d.high, d.low, d.close, d.volume, k[ok]).reset_index(drop=True)
        xv = xv.join(xo.set_axis(np.flatnonzero(ok)))          # keep each column's dtype (no object arrays)
    l1 = make_labels(d).set_index("date").reindex(asof).reset_index(drop=True)
    l2 = make_labels_v2(d).set_index("date").reindex(asof).reset_index(drop=True)
    l2 = l2[[c for c in l2.columns if c not in l1.columns and c != "sigma_ewma"]]
    out = pd.concat([f, xv, l1, l2], axis=1)
    out.insert(0, "asof_date", asof)
    out.insert(0, "ticker", ticker)
    adv = (d.close * d.volume).rolling(20).mean()
    out["adv20_usd"] = adv.to_numpy()[k]
    out["close"] = d.close.to_numpy()[k]
    out["n_daily_bars_before"] = k + 1
    ev = RB.evaluate_ticker(ticker, asof, daily=d, hourly=h)
    rule_cols = [c for c in ev.columns if c.startswith("rule_")]
    ev = ev.assign(asof_date=pd.to_datetime(ev.asof_date)).drop_duplicates("asof_date").set_index("asof_date")
    out = out.join(ev[rule_cols], on="asof_date")
    events = ev.drop(columns=rule_cols).reset_index()        # regime, setups, triggers, ATR, close (backtest input)
    events["sample_id"] = ticker + ":" + events.asof_date.dt.strftime("%Y-%m-%d")
    sb = B.StockBars(ticker, daily=d, hourly=h)
    imgs = {v: [] for v in VIEWS}
    nvalid = {v: [] for v in VIEWS}
    for kk in k:
        tr = R.triplet_at(sb, int(kk))
        for v in VIEWS:
            imgs[v].append(tr[v])
            nvalid[v].append(tr["n_valid"][v])
    for v in VIEWS:
        out[f"n_valid_{v}"] = nvalid[v]
    xcols = [c for c in out.columns if c.startswith("x_")]
    out[xcols] = out[xcols].astype(np.float32)          # as in the development frame (TriFrame's store is float32)
    out["sample_id"] = out.ticker + ":" + out.asof_date.dt.strftime("%Y-%m-%d")
    return ticker, out, imgs, events


def build(samples: pd.DataFrame, backbone: str = "convnext_tiny", workers: int = 5, embed_threads: int = 8,
          log=print) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """samples: (ticker, asof_date). Returns the frame (rows in samples' order where buildable), embeddings
    {view: float32 (n, dim)} aligned to the frame rows, and the rule events (one row per sample)."""
    import torch
    from ..cv import backbones as BB
    from ..data.calendar import decision_time_utc
    torch.set_num_threads(embed_threads)
    jobs = list(samples.groupby("ticker").asof_date.apply(list).items())
    frames, embs, evs, t0 = [], {v: [] for v in VIEWS}, [], time.time()
    n_img = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (t, fr, imgs, ev) in enumerate(ex.map(_ticker_rows, jobs, chunksize=1)):
            if fr is None:
                continue
            for v in VIEWS:
                embs[v].append(BB.embed(backbone, imgs[v], batch=64).astype(np.float32))
            n_img += 3 * len(fr)
            frames.append(fr)
            evs.append(ev)
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                log(f"{i + 1}/{len(jobs)} tickers, {sum(len(f) for f in frames):,} samples, {n_img / el:.1f} img/s, {el:.0f}s")
    fr = pd.concat(frames, ignore_index=True)
    E = {v: np.concatenate(embs[v]) for v in VIEWS}
    fr["decision_utc"] = decision_time_utc(fr.asof_date).values
    order = samples.assign(_o=np.arange(len(samples)))[["sample_id", "_o"]]
    fr = fr.merge(order, on="sample_id", how="left")
    perm = np.argsort(fr._o.to_numpy(), kind="stable")
    fr = fr.iloc[perm].drop(columns="_o").reset_index(drop=True)
    E = {v: e[perm] for v, e in E.items()}
    return fr, E, pd.concat(evs, ignore_index=True)
