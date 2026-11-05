"""TriFrame canonical chart renderer r1, vendored for NEXUS.

Source: triframe-cv/tf_render.py and us_stocks_ohlcv/scripts/tf_render.py (identical, sha256 a584c87e...dc00526).
`render`, `pack_triplet` and `unpack_triplet` are copied unchanged; `triplet_at` builds the three as-of windows from
NEXUS's sealed StockBars with the r1.1 rule (on early-close sessions only the first four hourly slots are kept, so
post-close prints are not drawn: TriFrame ERRATA E-14). Parity with TriFrame's cached training images and stored
embeddings is checked by scripts/verify/cv_parity.py.

Charts are binary OHLC bars (open tick left, high-low line centre, close tick right) with a volume panel; 104 weekly,
126 daily and 140 hourly bars; log-price scaled to each window, so a chart shows shape, not the volatility level.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

VERSION = "r1"
H, PRICE_H, GAP, VOL_H = 128, 100, 2, 26
VIEW_BARS = {"W": 104, "D": 126, "H": 140}
TRIPLET_BITS = H * 3 * sum(VIEW_BARS.values())          # 142,080
TRIPLET_BYTES = TRIPLET_BITS // 8                       # 17,760


def render(o, h, l, c, v, n_bars):
    """o,h,l,c,v: arrays of the m <= n_bars valid bars, oldest first. Returns (uint8 image [128, 3*n_bars], m)."""
    m = len(o); W = 3 * n_bars
    img = np.zeros((H, W), np.uint8)
    if m == 0:
        return img, 0
    o, h, l, c, v = (np.asarray(a, float) for a in (o, h, l, c, v))
    lo, hi = np.log(l.min()), np.log(h.max()); span = hi - lo
    if span > 0:
        f = lambda a: np.rint((np.log(a) - lo) / span * (PRICE_H - 1)).astype(int)
        yo, yh, yl, yc = f(o), f(h), f(l), f(c)
    else:
        yo = yh = yl = yc = np.full(m, PRICE_H // 2)
    x0 = 3 * (n_bars - m) + 3 * np.arange(m)
    rows = np.arange(PRICE_H)[:, None]
    price = np.zeros((PRICE_H, W), np.uint8)
    price[:, x0 + 1] = ((rows >= yl[None, :]) & (rows <= yh[None, :])).astype(np.uint8)
    price[yo, x0] = 1
    price[yc, x0 + 2] = 1
    img[:PRICE_H] = price[::-1]
    vmax = v.max()
    if vmax > 0:
        hv = np.rint(v / vmax * (VOL_H - 1)).astype(int) + (v > 0)
        vol = np.zeros((VOL_H, W), np.uint8)
        vol[:, x0 + 1] = (np.arange(VOL_H)[:, None] < hv[None, :]).astype(np.uint8)
        img[PRICE_H + GAP:] = vol[::-1]
    return img, m


def pack_triplet(wimg, dimg, himg):
    return np.packbits(np.concatenate([wimg.ravel(), dimg.ravel(), himg.ravel()]))


def unpack_triplet(buf):
    bits = np.unpackbits(buf)[:TRIPLET_BITS]
    out, p = {}, 0
    for k, nb in VIEW_BARS.items():
        n = H * 3 * nb
        out[k] = bits[p:p + n].reshape(H, 3 * nb); p += n
    return out


def sha(img) -> str:
    return hashlib.sha256(np.packbits(img).tobytes()).hexdigest()


def rth_hourly(hourly: pd.DataFrame, early_close_days) -> pd.DataFrame:
    """Regular-session hourly bars stamped 09:30 ... 15:30 New York; on early-close sessions only the first four slots
    (r1.1 rule). Mirrors tfcv_core.rth_only."""
    if len(hourly) == 0:
        return hourly
    et = pd.DatetimeIndex(hourly["timestamp"]).tz_convert("America/New_York")
    keep = np.asarray((et.minute == 30) & (et.hour >= 9) & (et.hour <= 15))
    day = et.tz_localize(None).normalize()
    keep &= ~(np.asarray(day.isin(list(early_close_days))) & np.asarray(et.hour - 9 >= 4))
    return hourly[keep].reset_index(drop=True)


def triplet_at(sb, k: int) -> dict:
    """The three chart images as of daily index k of a nexus_alpha.data.bars.StockBars (bars <= session k only)."""
    w = sb.weekly_window(k, VIEW_BARS["W"])
    d = sb.daily_window(k, VIEW_BARS["D"])
    hb = sb.hourly_window(k, VIEW_BARS["H"])
    out, nvalid = {}, {}
    for key, b in (("W", w), ("D", d), ("H", hb)):
        img, m = render(b.open.values, b.high.values, b.low.values, b.close.values, b.volume.values, VIEW_BARS[key])
        out[key], nvalid[key] = img, m
    out["n_valid"] = nvalid
    return out


def stock_bars_for_render(ticker: str):
    """StockBars with the r1.1 hourly rule applied (early-close post-close prints removed)."""
    from ..data.bars import StockBars, read_hourly
    from ..data.calendar import calendar
    cal = calendar()
    early = set(cal.date[cal.early_close.astype(bool)])
    return StockBars(ticker, hourly=rth_hourly(read_hourly(ticker), early))
