# Vendored unchanged from triframe-cv/feat_visible.py (sha256 15796b45464744e66076ead4e6456b4692797d71f667a3a9ec2208dbdf980ecd) for NEXUS inference parity.
# Parity with TriFrame's frozen features_visible values: tests/test_parity_sources.py.
"""Visible-window numeric features (spec: playbook/features_visible_v1/FEATURES_VISIBLE_SPEC_v1.md).

Every feature is a pure function of the LAST W=126 DAILY bars ending at decision bar t (inclusive) — exactly what a
screenshot shows. Training rows and dashboard inference therefore use the same function (ERRATA E-11). Nothing outside
the window (no earlier history, no future bar) can influence a value.
"""
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

W = 126
EWMA_A = 0.06
LN2_4 = 4.0 * np.log(2.0)
EPS = 1e-8                       # variance floor = (1 bp)^2: flat/zero-return days do not create a -13.8 plateau
REL0 = 1e-10                     # "zero" test for dispersions of price levels (float std of a constant series is ~1e-17)
FEATURES = [
    # returns and overnight / intraday split
    "ret1", "ret5", "ret21", "overnight21", "intraday21", "gap",
    # volume
    "abn_vol", "vol_surge5",
    # position in range / mean-reversion block
    "close_loc5", "range_pos20", "z20", "dist_atr", "rsi2", "rsi3", "rsi14c",
    "down_streak", "up_streak", "vr5", "ar1_63", "trend20_z",
    # volatility (log daily vol): close-to-close, Parkinson, Garman-Klass at 1/5/22 bars, windowed EWMA
    "cc_1", "cc_5", "cc_22", "park_1", "park_5", "park_22", "gk_1", "gk_5", "gk_22", "ewma_win",
    # amplitude / scale covariates (what per-window image scaling hides)
    "atr_pct", "log_range_win", "maxdd_win", "mean_dvol_log",
]
COLS = ["x_" + f for f in FEATURES]    # stored / joined column names carry the x_ prefix (labels also have ret5, ret20, ...)
DEPLOY = [f for f in FEATURES if f != "mean_dvol_log"]   # mean_dvol_log needs the volume axis units: research/stratification only
MR_BLOCK = ["ret1", "ret5", "gap", "close_loc5", "range_pos20", "z20", "dist_atr", "rsi2", "rsi3", "down_streak",
            "vr5", "ar1_63", "overnight21"]


def _cutler_rsi(c, n):
    d = np.diff(c[:, -(n + 1):], axis=1)
    g = np.clip(d, 0, None).mean(1); l = np.clip(-d, 0, None).mean(1)
    s = g + l
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, 100.0 * (g / s), 50.0)


def _streak(neg, cap=5):
    """Number of consecutive True values ending at the last column (max cap). neg: (N, >=cap) bool."""
    s = np.zeros(neg.shape[0]); alive = np.ones(neg.shape[0], bool)
    for k in range(1, cap + 1):
        alive &= neg[:, -k]
        s += alive
    return s


def from_windows(O, H, L, C, V):
    """O,H,L,C,V: float arrays of shape (N, W) oldest -> newest. Returns DataFrame (N, len(FEATURES))."""
    N = C.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        lc, lo = np.log(C), np.log(O)
        r = lc[:, 1:] - lc[:, :-1]                                   # (N, W-1) daily log returns
        out = {}
        out["ret1"] = r[:, -1]
        out["ret5"] = lc[:, -1] - lc[:, -6]
        out["ret21"] = lc[:, -1] - lc[:, -22]
        out["overnight21"] = (lo[:, -21:] - lc[:, -22:-1]).sum(1)
        out["intraday21"] = (lc[:, -21:] - lo[:, -21:]).sum(1)
        out["gap"] = lo[:, -1] - lc[:, -2]
        v20 = V[:, -20:].mean(1)
        out["abn_vol"] = np.where(v20 > 0, V[:, -1] / v20, np.nan)
        out["vol_surge5"] = np.where(v20 > 0, V[:, -5:].mean(1) / v20, np.nan)
        hi5, lo5 = H[:, -5:].max(1), L[:, -5:].min(1)
        out["close_loc5"] = np.where(hi5 > lo5, (C[:, -1] - lo5) / (hi5 - lo5), np.nan)
        hi20, lo20 = H[:, -20:].max(1), L[:, -20:].min(1)
        out["range_pos20"] = np.where(hi20 > lo20, (C[:, -1] - lo20) / (hi20 - lo20), np.nan)
        m20 = C[:, -20:].mean(1); s20 = C[:, -20:].std(1, ddof=0)
        out["z20"] = np.where(s20 > REL0 * np.abs(m20), (C[:, -1] - m20) / s20, np.nan)
        pcl = C[:, :-1]                                              # Wilder ATR14 inside the window (v1.2): SMA seed over
        tr = np.maximum(H[:, 1:] - L[:, 1:], np.maximum(np.abs(H[:, 1:] - pcl), np.abs(L[:, 1:] - pcl)))   # TR_j, j=1..125
        atr = tr[:, :14].mean(1)                                     # the first 14 true ranges, then (13*A + TR)/14;
        for j in range(14, W - 1):                                   # equals the frozen full-history ATR14 to < 3e-4
            atr = (13.0 * atr + tr[:, j]) / 14.0
        out["dist_atr"] = np.where(atr > 0, (C[:, -1] - m20) / atr, np.nan)
        out["atr_pct"] = atr / C[:, -1]
        out["rsi2"], out["rsi3"], out["rsi14c"] = _cutler_rsi(C, 2), _cutler_rsi(C, 3), _cutler_rsi(C, 14)
        out["down_streak"] = _streak(r < 0)
        out["up_streak"] = _streak(r > 0)
        c64 = lc[:, -64:]                                            # 63 daily and 59 overlapping 5-day returns
        r1, r5 = c64[:, 1:] - c64[:, :-1], c64[:, 5:] - c64[:, :-5]
        v1 = r1.var(1, ddof=1)
        out["vr5"] = np.where(v1 > 1e-20, r5.var(1, ddof=1) / (5.0 * v1), np.nan)
        a, b = r[:, -63:-1], r[:, -62:]
        a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        out["ar1_63"] = np.where(den > 1e-18, (a * b).sum(1) / den, np.nan)
        x = np.arange(20.0); x -= x.mean()
        slope = (lc[:, -20:] * x).sum(1) / (x * x).sum()
        sd = r[:, -19:].std(1, ddof=1)
        out["trend20_z"] = np.where(sd > 1e-12, slope * 19.0 / (sd * np.sqrt(19.0)), np.nan)
        # volatility estimators (log of daily vol)
        hl = np.log(H / L); co = lc - lo
        park = hl ** 2 / LN2_4
        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        r2 = r ** 2
        for w in (1, 5, 22):
            out[f"cc_{w}"] = 0.5 * np.log(np.clip(r2[:, -w:].mean(1), EPS, None))
            out[f"park_{w}"] = 0.5 * np.log(np.clip(park[:, -w:].mean(1), EPS, None))
            out[f"gk_{w}"] = 0.5 * np.log(np.clip(gk[:, -w:].mean(1), EPS, None))
        wk = (1 - EWMA_A) ** np.arange(W - 1)[::-1]                  # weight of the newest return = 1
        out["ewma_win"] = 0.5 * np.log(np.clip((r2 * wk).sum(1) / wk.sum(), EPS, None))
        out["log_range_win"] = np.log(H.max(1) / L.min(1))
        peak = np.maximum.accumulate(C, axis=1)
        out["maxdd_win"] = np.log(C / peak).min(1)
        out["mean_dvol_log"] = np.log(np.clip((C[:, -20:] * V[:, -20:]).mean(1), 1.0, None))
    df = pd.DataFrame({"x_" + k: out[k].astype(np.float64) for k in FEATURES})
    # validity precondition (spec v1.1): every bar positive, finite and H >= max(O,C), L <= min(O,C), else all-NaN row
    tol = 1e-9 * C
    bad = ~(np.isfinite(O) & np.isfinite(H) & np.isfinite(L) & np.isfinite(C) & (L > 0) & (O > 0) & (C > 0)
            & (H >= np.maximum(O, C) - tol) & (L <= np.minimum(O, C) + tol)).all(axis=1)
    if bad.any(): df.loc[bad, :] = np.nan
    return df


def from_series(o, h, l, c, v, idx):
    """Features at end positions idx (0-based into the series; idx >= W-1) of one stock's daily series."""
    idx = np.asarray(idx, int)
    assert (idx >= W - 1).all() and (idx < len(c)).all()
    S = lambda a: swv(np.asarray(a, float), W)                         # row j = bars j .. j+W-1
    start = idx - (W - 1)
    O, H, L, C, V = (S(a)[start] for a in (o, h, l, c, v))
    return from_windows(O, H, L, C, V)
