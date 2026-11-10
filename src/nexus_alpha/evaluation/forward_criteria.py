"""Forward promotion criteria of the ranking candidates (docs/V3_RESEARCH.md; D-042, D-049, D-051).

A candidate (v3_fund, v3_alpha) becomes a shown signal (never traded) only if, on on-time rows after at least 120
scored sessions at 20D, both hold:
1. its mean daily rank IC (Spearman of rank_score with the realized return, across stocks on each session) is above 0;
2. its paired daily rank-IC difference against `ranker_v3` (v2's rank_score on the same sessions and stocks) is above 0.

Each condition gets a one-sided moving-block bootstrap p-value over sessions (blocks of 20 sessions, the horizon, so
overlapping 20D windows stay together). Both conditions must hold, so a candidate's p-value is the larger of its two.
The candidates are decided together with Holm's correction (D-051).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .iv_benchmark import holm

MIN_SESSIONS, BLOCK, MIN_STOCKS = 120, 20, 10


def daily_ic(rows: pd.DataFrame, score: str = "rank_score") -> pd.Series:
    out = {}
    for d, g in rows.groupby("asof_session"):
        g = g.dropna(subset=[score, "real_ret"])
        if len(g) >= MIN_STOCKS:
            out[d] = g[score].rank().corr(g.real_ret.rank())
    return pd.Series(out, dtype=float).sort_index()


def block_p_value(x: pd.Series, block: int = BLOCK, n_boot: int = 2000, seed: int = 11) -> float:
    """One-sided p-value for mean(x) > 0: the share of moving-block bootstrap means at or below 0."""
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 2:
        return float("nan")
    b = min(block, n)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / b))
    means = np.empty(n_boot)
    for i in range(n_boot):
        st = rng.integers(0, n - b + 1, k)
        means[i] = np.concatenate([v[s:s + b] for s in st])[:n].mean()
    return float(np.mean(means <= 0))


def evaluate(candidates: dict[str, pd.DataFrame], ranker: pd.DataFrame, min_sessions: int = MIN_SESSIONS) -> dict:
    """candidates: name -> scored 20D on-time rows (asof_session, ticker, rank_score, real_ret); ranker likewise."""
    out, pv = {}, {}
    ric = daily_ic(ranker)
    for name, rows in candidates.items():
        cic = daily_ic(rows)
        paired = rows.merge(ranker[["asof_session", "ticker", "rank_score"]], on=["asof_session", "ticker"],
                            suffixes=("", "_ranker"))
        pic = daily_ic(paired)
        rk = daily_ic(paired.assign(rank_score=paired.rank_score_ranker))
        diff = (pic - rk).dropna()
        p1, p2 = block_p_value(cic), block_p_value(diff)
        out[name] = {"sessions": int(len(cic)), "mean_ic": float(cic.mean()) if len(cic) else None,
                     "mean_ic_minus_ranker": float(diff.mean()) if len(diff) else None,
                     "p_ic_above_0": p1, "p_above_ranker": p2,
                     "ranker_mean_ic": float(ric.mean()) if len(ric) else None}
        if len(cic) >= min_sessions and np.isfinite(p1) and np.isfinite(p2):
            pv[name] = max(p1, p2)
    if pv and len(pv) == len(candidates):
        rej = holm(pv)
        for name in candidates:
            out[name]["decision"] = ("promote to a shown signal (Holm-corrected, D-051)" if rej[name]
                                     else "not promoted (Holm-corrected, D-051)")
    else:
        for name in candidates:
            out[name]["decision"] = (f"insufficient: {out[name]['sessions']} scored sessions at 20D "
                                     f"(needs {min_sessions} for every candidate)")
    return out
