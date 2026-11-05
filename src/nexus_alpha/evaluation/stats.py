"""Inference that respects time dependence (master spec 11.5).

* Month-block bootstrap: resample calendar months of the as-of date with replacement (overlapping 1D-1M labels and
  same-date cross-sections stay together), recompute the metric, percentile 95% CI. Paired: both models are scored on
  the same resampled rows.
* Holm step-down correction for a family of hypotheses.
* Newey-West (HAC) t-statistic for the mean of a time series of differences.
* Probabilistic Sharpe ratio and the deflated Sharpe ratio (Bailey and Lopez de Prado, 2014).
* Probability of backtest overfitting by combinatorially symmetric cross-validation (Bailey et al., 2017).
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps

N_BOOT = 1000
SEED = 20261105


def month_blocks(dates) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates))
    return (d.dt.year * 12 + d.dt.month).to_numpy()


def block_bootstrap(metric: Callable, arrays: tuple, blocks: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """metric(*arrays_subset) -> float. Returns point estimate, 95% percentile CI and the bootstrap SE."""
    uniq = np.unique(blocks)
    pos = {b: np.flatnonzero(blocks == b) for b in uniq}
    rng = np.random.default_rng(seed)
    point = float(metric(*arrays))
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        ix = np.concatenate([pos[b] for b in pick])
        vals[i] = metric(*(a[ix] for a in arrays))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"estimate": point, "ci95": [float(lo), float(hi)], "se": float(np.nanstd(vals, ddof=1)),
            "n_blocks": int(len(uniq)), "n_boot": n_boot,
            "p_two_sided_vs_0": float(min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean())))}


def paired_difference(metric: Callable, y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, blocks: np.ndarray,
                      n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap CI of metric(y, p_a) - metric(y, p_b) on identical resampled rows."""
    return block_bootstrap(lambda yy, a, b: metric(yy, a) - metric(yy, b), (y, p_a, p_b), blocks, n_boot, seed)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject_at_alpha": adj <= alpha}
    return out


def newey_west_t(x: np.ndarray, lags: int | None = None) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return {"mean": float(np.mean(x)) if n else np.nan, "t": np.nan, "n": n}
    lags = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    e = x - x.mean()
    s = (e ** 2).sum() / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        s += 2 * w * (e[L:] * e[:-L]).sum() / n
    se = np.sqrt(s / n)
    t = x.mean() / se if se > 0 else np.nan
    return {"mean": float(x.mean()), "t": float(t), "se": float(se), "n": n, "lags": lags,
            "p_two_sided": float(2 * (1 - sps.norm.cdf(abs(t)))) if np.isfinite(t) else np.nan}


def sharpe(returns: np.ndarray, periods_per_year: float = 252) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true (per-period) Sharpe exceeds sr_benchmark, given skew and kurtosis."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return np.nan
    sr = r.mean() / r.std(ddof=1)
    g3 = sps.skew(r)
    g4 = sps.kurtosis(r, fisher=False)
    den = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / den))


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_trials_var: float) -> dict:
    """DSR: PSR against the expected maximum Sharpe of n_trials independent trials with variance sr_trials_var
    (per-period units)."""
    emc = 0.5772156649
    if n_trials < 2:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(sr_trials_var) * ((1 - emc) * sps.norm.ppf(1 - 1 / n_trials)
                                        + emc * sps.norm.ppf(1 - 1 / (n_trials * np.e)))
    return {"sr0_per_period": float(sr0), "dsr": probabilistic_sharpe(returns, sr0), "n_trials": int(n_trials)}


def pbo_cscv(perf: np.ndarray, n_splits: int = 16) -> dict:
    """Probability of backtest overfitting. perf: (T, N) per-period performance of N configurations. The T periods are
    cut into n_splits contiguous blocks; every half/half combination is an in-sample/out-of-sample pair; PBO is the share
    of combinations where the in-sample best configuration ranks below the out-of-sample median."""
    perf = np.asarray(perf, float)
    T, N = perf.shape
    if N < 2:
        return {"pbo": np.nan, "reason": "fewer than 2 configurations"}
    blocks = np.array_split(np.arange(T), n_splits)
    logits = []
    for comb in itertools.combinations(range(n_splits), n_splits // 2):
        ins = np.concatenate([blocks[i] for i in comb])
        oos = np.concatenate([blocks[i] for i in range(n_splits) if i not in comb])
        mi, mo = np.nanmean(perf[ins], 0), np.nanmean(perf[oos], 0)
        best = int(np.nanargmax(mi))
        rank = (sps.rankdata(mo)[best]) / (N + 1)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": int(len(logits)), "n_configs": int(N),
            "median_logit": float(np.median(logits))}
