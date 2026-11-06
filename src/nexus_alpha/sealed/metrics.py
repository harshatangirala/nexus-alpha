"""Endpoints of docs/SEALED_EVALUATION_PLAN.md section 5.

Intervals: moving-block bootstrap over calendar months with blocks of max(1, ceil(h/21)) months (a group uses its longest
horizon's length), 1000 draws, seed 11 (the D-032 rule). Holm correction across the three horizon groups per model.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..evaluation import ladder as L

GROUPS_V1 = {"intraday": ["1D"], "swing": ["2D", "5D", "10D", "20D"], "long": ["1M"]}
GROUPS_V2 = {"intraday": ["1D"], "swing": [f"{h}D" for h in range(2, 21)], "long": ["3M", "6M", "1Y"]}


def _blocks(months: np.ndarray, L_months: int, rng, n_boot: int):
    um = np.unique(months)
    n = len(um)
    k = math.ceil(n / L_months)
    for _ in range(n_boot):
        st = rng.integers(0, n, k)
        yield um, ((st[:, None] + np.arange(L_months)[None, :]) % n).ravel()[:n]


def group_gain(per: list[tuple[np.ndarray, np.ndarray]], L_months: int, rng, n_boot: int = 1000) -> dict:
    """per: [(months, gain per row)] for the group's horizons. Mean over horizons of each horizon's mean gain."""
    um = np.unique(np.concatenate([m for m, _ in per]))
    S = np.stack([pd.Series(d).groupby(m).sum().reindex(um).fillna(0).to_numpy() for m, d in per])
    C = np.stack([pd.Series(d).groupby(m).size().reindex(um).fillna(0).to_numpy() for m, d in per])
    point = float(np.mean(S.sum(1) / np.maximum(C.sum(1), 1)))
    draws = [float(np.mean(S[:, ix].sum(1) / np.maximum(C[:, ix].sum(1), 1))) for _, ix in _blocks(um, L_months, rng, n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p_le0 = float((np.array(draws) <= 0).mean())
    return {"gain": point, "ci95": [float(lo), float(hi)], "p_one_sided": p_le0, "n_months": int(len(um))}


def auroc_ci(y: np.ndarray, P: np.ndarray, months: np.ndarray, L_months: int, rng, n_boot: int = 1000) -> dict:
    a = L.auroc_ud(y, P)
    um = np.unique(months)
    idx = {m: np.flatnonzero(months == m) for m in um}
    draws = []
    for _, ix in _blocks(months, L_months, rng, n_boot):
        r = np.concatenate([idx[um[i]] for i in ix])
        v = L.auroc_ud(y[r], P[r])
        if v == v:
            draws.append(v)
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if draws else [float("nan")] * 2
    return {"auroc_ud": a, "ci95": ci}


def ece(y: np.ndarray, P: np.ndarray, bins: int = 10) -> dict:
    """Class-wise expected calibration error (mean over the three classes) and a reliability table per class."""
    out, tables = [], {}
    for c, name in enumerate(("down", "side", "up")):
        p = P[:, c]
        b = np.minimum((p * bins).astype(int), bins - 1)
        df = pd.DataFrame({"b": b, "p": p, "y": (y == c).astype(float)}).groupby("b").agg(n=("y", "size"), p=("p", "mean"),
                                                                                     freq=("y", "mean"))
        out.append(float((df.n / len(y) * (df.p - df.freq).abs()).sum()))
        tables[name] = df.reset_index().round(4).to_dict(orient="records")
    return {"ece_classwise": float(np.mean(out)), "reliability": tables}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def evaluate(model: str, probs: np.ndarray, horizons: list[str], h_sessions: dict, labels: pd.DataFrame,
             prior: dict, groups: dict, quantiles: np.ndarray | None = None, seed: int = 11) -> dict:
    """labels: frame aligned to probs rows with y_<h>, z_<h>, asof_date. prior: {h: [p_down, p_side, p_up]}."""
    rng = np.random.default_rng(seed)
    month = (labels.asof_date.dt.year * 12 + labels.asof_date.dt.month).to_numpy()
    res, rows = {}, {}
    for j, h in enumerate(horizons):
        y = labels[f"y_{h}"].to_numpy()
        m = ~np.isnan(y)
        if m.sum() == 0:
            res[h] = {"n": 0}
            continue
        yy, P = y[m].astype(int), probs[m, j]
        ll = L.ll_rows(yy, P)
        llp = L.ll_rows(yy, np.tile(np.asarray(prior[h], float), (len(yy), 1)))
        Lm = max(1, math.ceil(h_sessions[h] / 21))
        r = {"n": int(m.sum()), "n_dates": int(labels.asof_date[m].nunique()), "log_loss": float(ll.mean()),
             "log_loss_prior": float(llp.mean()), "gain_vs_prior": group_gain([(month[m], llp - ll)], Lm, rng),
             "direction": auroc_ci(yy, P, month[m], Lm, rng), "calibration": ece(yy, P),
             "class_freq": [float((yy == c).mean()) for c in range(3)]}
        if quantiles is not None:
            z = labels[f"z_{h}"].to_numpy()[m]
            q = quantiles[m, j]
            ok = ~np.isnan(z)
            r["coverage_10_90"] = float(((z[ok] >= q[ok, 0]) & (z[ok] <= q[ok, -1])).mean()) if ok.any() else None
        res[h] = r
        rows[h] = (month[m], llp - ll)
    grp, pv = {}, {}
    for g, hs in groups.items():
        hs = [h for h in hs if h in rows]
        if not hs:
            continue
        Lm = max(max(1, math.ceil(h_sessions[h] / 21)) for h in hs)
        grp[g] = group_gain([rows[h] for h in hs], Lm, rng) | {"horizons": hs, "block_months": Lm}
        pv[g] = grp[g]["p_one_sided"]
    adj = holm(pv)
    for g in grp:
        grp[g]["p_holm"] = adj[g]
    return {"model": model, "by_horizon": res, "groups_primary": grp}


def compare(name_a: str, Pa: np.ndarray, ha: list[str], name_b: str, Pb: np.ndarray, hb: list[str], labels: pd.DataFrame,
            shared: list[str], h_sessions: dict, seed: int = 11) -> dict:
    """Log-loss gain of model b over model a on identical rows, per shared horizon."""
    rng = np.random.default_rng(seed)
    month = (labels.asof_date.dt.year * 12 + labels.asof_date.dt.month).to_numpy()
    out = {}
    for h in shared:
        y = labels[f"y_{h}"].to_numpy()
        m = ~np.isnan(y)
        yy = y[m].astype(int)
        d = L.ll_rows(yy, Pa[m, ha.index(h)]) - L.ll_rows(yy, Pb[m, hb.index(h)])
        out[h] = group_gain([(month[m], d)], max(1, math.ceil(h_sessions[h] / 21)), rng)
    return {"a": name_a, "b": name_b, "gain_of_b_over_a": out}
