"""Qlib Alpha158 price-volume factors, re-implemented causally for the NEXUS daily bars.

Definitions: Microsoft Qlib (MIT licence), `qlib/contrib/data/loader.py`, `Alpha158DL.get_feature_config` with the
configuration of the `Alpha158` handler (kbar; price window 0 for OPEN, HIGH, LOW and VWAP; rolling windows 5, 10, 20,
30 and 60 with all 29 rolling features). Operator semantics follow `qlib/data/ops.py` and `qlib/data/_libs/rolling.pyx`:
* every rolling operator uses `min_periods=1`, so the first rows use the bars available so far;
* Slope, Rsquare and Resi regress the series on its position inside the window, and Resi is the residual of the
  newest bar;
* IdxMax and IdxMin are the 1-based position of the first extreme inside the window;
* Corr is missing where either rolling standard deviation is within 2e-5 of zero.

Differences from Qlib, by design (pre-registered with E25 in docs/V3_RESEARCH.md):
* VWAP0 is left out because the NEXUS daily bars have no VWAP, so the set has 157 of the 158 features;
* +/-inf is returned as missing. Qlib leaves that to a later processor.

Parity with Qlib's own expression engine is checked by `scripts/verify/alpha158_qlib_parity.py`.

Causality: every operator looks backwards (Ref with a positive lag, trailing windows), so the value at session t uses
bars up to and including t only (tested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
EPS = 1e-12
KBAR = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
PRICE = ["OPEN0", "HIGH0", "LOW0"]          # VWAP0 left out: the bars have no VWAP
ROLLING = ["ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN",
           "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP",
           "VSUMN", "VSUMD"]
NAMES = KBAR + PRICE + [f"{f}{d}" for f in ROLLING for d in WINDOWS]
PREFIX = "a158_"
COLUMNS = [PREFIX + n for n in NAMES]
OHLCV = ["open", "high", "low", "close", "volume"]


def _rolling_ols(y: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slope, R^2 and newest-bar residual of y regressed on its position in a trailing window of d bars (fewer at the
    start). The positions are 1..k for the k bars in the window; all three results are invariant to shifting them."""
    n = len(y)
    j = np.arange(n, dtype=np.float64)
    k = np.minimum(np.arange(1, n + 1), d).astype(np.float64)

    def rsum(a: np.ndarray) -> np.ndarray:
        return pd.Series(a).rolling(d, min_periods=1).sum().to_numpy()

    sy, syy, sjy = rsum(y), rsum(y * y), rsum(j * y)
    sxy = sjy - (j - k) * sy                 # position x = j - (t - k) for the bars j in (t - k, t]
    sx = k * (k + 1) / 2
    sxx = k * (k + 1) * (2 * k + 1) / 6
    num = k * sxy - sx * sy
    den_x = k * sxx - sx * sx
    den_y = k * syy - sy * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = num / den_x
        rsq = (num / np.sqrt(den_x * den_y)) ** 2
        resi = y - (slope * k + (sy - slope * sx) / k)
    return slope, rsq, resi


def _idx_extreme(a: np.ndarray, d: int, how: str) -> np.ndarray:
    """1-based position of the first maximum (how='max') or minimum inside the trailing window of d bars."""
    n = len(a)
    f = np.argmax if how == "max" else np.argmin
    out = np.empty(n, dtype=np.float64)
    for t in range(min(d - 1, n)):
        out[t] = f(a[: t + 1]) + 1
    if n >= d:
        out[d - 1:] = f(np.lib.stride_tricks.sliding_window_view(a, d), axis=1) + 1
    return out


def _corr(a: pd.Series, b: pd.Series, d: int) -> pd.Series:
    res = a.rolling(d, min_periods=1).corr(b)
    flat = (np.isclose(a.rolling(d, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(b.rolling(d, min_periods=1).std(), 0, atol=2e-05))
    return res.mask(flat)


MIN_BARS = 61        # the longest window is 60 sessions, and ROC60 needs one more bar


def features_at(bars: pd.DataFrame, asof) -> pd.Series:
    """The 157 features of the as-of session from one stock's daily bars (columns date, open, high, low, close,
    volume). Bars after the as-of session are ignored. Raises if the last bar is not the as-of session or the history
    is shorter than MIN_BARS, so a stale or partial bar set is never scored."""
    asof = pd.Timestamp(asof).normalize()
    b = bars[pd.to_datetime(bars.date).dt.normalize() <= asof].sort_values("date").reset_index(drop=True)
    if not len(b) or pd.Timestamp(b.date.iloc[-1]).normalize() != asof:
        raise ValueError(f"last bar is not the as-of session {asof.date()}")
    if len(b) < MIN_BARS:
        raise ValueError(f"only {len(b)} bars: Alpha158 needs at least {MIN_BARS}")
    return alpha158(b).iloc[-1]


def alpha158(bars: pd.DataFrame) -> pd.DataFrame:
    """The 157 Alpha158 features of one stock's daily bars.

    `bars` holds the columns open, high, low, close and volume in date order, with no missing values. The result has
    the same index as `bars`, the columns `COLUMNS` (float64), and +/-inf set to missing.
    """
    if bars[OHLCV].isna().any().any():
        raise ValueError("alpha158 needs complete bars: drop the rows with missing values first")
    o, h, lo, c, v = (bars[k].astype("float64").reset_index(drop=True) for k in OHLCV)
    out: dict[str, pd.Series | np.ndarray] = {}
    hl = h - lo + EPS
    top, bot = np.maximum(o, c), np.minimum(o, c)
    out["KMID"] = (c - o) / o
    out["KLEN"] = (h - lo) / o
    out["KMID2"] = (c - o) / hl
    out["KUP"] = (h - top) / o
    out["KUP2"] = (h - top) / hl
    out["KLOW"] = (bot - lo) / o
    out["KLOW2"] = (bot - lo) / hl
    out["KSFT"] = (2 * c - h - lo) / o
    out["KSFT2"] = (2 * c - h - lo) / hl
    out["OPEN0"], out["HIGH0"], out["LOW0"] = o / c, h / c, lo / c

    c1, v1 = c.shift(1), v.shift(1)
    dc, dv = c - c1, v - v1
    up, dn, adc = np.maximum(dc, 0), np.maximum(-dc, 0), dc.abs()         # NaN at the first bar, as in Qlib
    vup, vdn, adv = np.maximum(dv, 0), np.maximum(-dv, 0), dv.abs()
    gt, lt = (c > c1).astype("float64"), (c < c1).astype("float64")     # a comparison with NaN is False
    log_v = np.log(v + 1)
    ret = c / c1
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vr = np.log(v / v1 + 1)
    wv = (ret - 1).abs() * v
    cn, hn, ln = c.to_numpy(), h.to_numpy(), lo.to_numpy()

    per: dict[str, dict[int, pd.Series | np.ndarray]] = {f: {} for f in ROLLING}
    for d in WINDOWS:
        def r(s: pd.Series, d: int = d):
            return s.rolling(d, min_periods=1)

        slope, rsq, resi = _rolling_ols(cn, d)
        hmax, lmin = r(h).max(), r(lo).min()
        imax, imin = _idx_extreme(hn, d, "max"), _idx_extreme(ln, d, "min")
        cntp, cntn = r(gt).mean(), r(lt).mean()
        s_up, s_dn, s_abs = r(up).sum(), r(dn).sum(), r(adc).sum() + EPS
        v_up, v_dn, v_abs = r(vup).sum(), r(vdn).sum(), r(adv).sum() + EPS
        per["ROC"][d] = c.shift(d) / c
        per["MA"][d] = r(c).mean() / c
        per["STD"][d] = r(c).std() / c
        per["BETA"][d] = slope / cn
        per["RSQR"][d] = rsq
        per["RESI"][d] = resi / cn
        per["MAX"][d] = hmax / c
        per["MIN"][d] = lmin / c
        per["QTLU"][d] = r(c).quantile(0.8) / c
        per["QTLD"][d] = r(c).quantile(0.2) / c
        per["RANK"][d] = r(c).rank(pct=True)
        per["RSV"][d] = (c - lmin) / (hmax - lmin + EPS)
        per["IMAX"][d] = imax / d
        per["IMIN"][d] = imin / d
        per["IMXD"][d] = (imax - imin) / d
        per["CORR"][d] = _corr(c, log_v, d)
        per["CORD"][d] = _corr(ret, log_vr, d)
        per["CNTP"][d], per["CNTN"][d], per["CNTD"][d] = cntp, cntn, cntp - cntn
        per["SUMP"][d], per["SUMN"][d], per["SUMD"][d] = s_up / s_abs, s_dn / s_abs, (s_up - s_dn) / s_abs
        per["VMA"][d] = r(v).mean() / (v + EPS)
        per["VSTD"][d] = r(v).std() / (v + EPS)
        per["WVMA"][d] = r(wv).std() / (r(wv).mean() + EPS)
        per["VSUMP"][d], per["VSUMN"][d], per["VSUMD"][d] = v_up / v_abs, v_dn / v_abs, (v_up - v_dn) / v_abs
    for f in ROLLING:
        for d in WINDOWS:
            out[f"{f}{d}"] = per[f][d]
    df = pd.DataFrame({PREFIX + k: np.asarray(out[k], dtype=np.float64) for k in NAMES})
    df.index = bars.index
    return df.replace([np.inf, -np.inf], np.nan)
