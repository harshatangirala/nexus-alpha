"""Forward shadow test, scoring step: scores every logged, on-time forecast whose horizon has ended.

    python scripts/shadow/shadow_score.py --version v1.1

Writes shadow/<version>/scores.csv (one row per scored forecast) and shadow/<version>/summary.json (per horizon: n,
log loss vs the training class prior, up-vs-down AUROC, hit rate, with date-clustered bootstrap intervals once there
are at least 20 scored dates). Late rows are excluded. The chain is verified before anything is scored.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.targets import HORIZONS  # noqa: E402


def yahoo_daily(ticker: str, start: str) -> pd.DataFrame:
    import yfinance as yf
    from nexus_alpha.inference.pipeline import yahoo_symbol
    d = yf.download(yahoo_symbol(ticker), start=start, interval="1d", auto_adjust=False, actions=False, progress=False,
                    multi_level_index=False, threads=False)
    if d is None or not len(d):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    di = pd.DatetimeIndex(d.index)
    di = di.tz_localize(None) if di.tz is not None else di
    return pd.DataFrame({"date": di.normalize(), "open": d["Open"].values, "high": d["High"].values,
                         "low": d["Low"].values, "close": d["Close"].values}).dropna().reset_index(drop=True)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.1")
    a = ap.parse_args()
    probs = SH.verify_chain(a.version)
    if probs:
        sys.exit(f"chain check failed: {probs}")
    root = SH.SHADOW_ROOT / a.version
    files = sorted((root / "forecasts").glob("*.csv"))
    if not files:
        sys.exit("no forecasts logged yet")
    fc = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fc = fc[~fc.late.astype(bool)]
    first = (pd.Timestamp(fc.asof_session.min()) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    out = []
    for t, g in fc.groupby("ticker"):
        d = yahoo_daily(t, first)
        for asof, gg in g.groupby("asof_session"):
            lab = SH.realized_labels(d, pd.Timestamp(asof), HORIZONS)
            for _, r in gg.iterrows():
                if r.horizon in lab:
                    out.append({**r.to_dict(), **{f"real_{k}": v for k, v in lab[r.horizon].items()}})
    if not out:
        print("nothing has matured yet")
        return
    sc = pd.DataFrame(out)
    sc.to_csv(root / "scores.csv", index=False, float_format="%.6g")
    meta = json.loads((SH.FROZEN_DIR / f"{a.version}.json").read_text(encoding="utf-8"))
    prior = meta.get("train_class_prior", {})
    summ, rng = {}, np.random.default_rng(0)
    for h, g in sc.groupby("horizon"):
        P = g[["p_down", "p_side", "p_up"]].to_numpy()
        y = g.real_y.astype(int).to_numpy()
        ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))
        pr = np.array(prior.get(h, [1 / 3] * 3))
        llp = -np.log(pr[y])
        ud = y != 1
        res = {"n": int(len(y)), "n_dates": int(g.asof_session.nunique()), "log_loss": float(ll.mean()),
               "log_loss_prior": float(llp.mean()), "gain_vs_prior": float(llp.mean() - ll.mean()),
               "auroc_up_vs_down": auroc((y[ud] == 2).astype(int), (P[ud, 2] - P[ud, 0])),
               "hit_rate_argmax": float((P.argmax(1) == y).mean()),
               "class_freq": [float((y == c).mean()) for c in range(3)]}
        dates = g.asof_session.to_numpy()
        ud_dates = np.unique(dates)
        if len(ud_dates) >= 20:   # date-clustered bootstrap (forecasts on one date are correlated)
            idx = {d: np.where(dates == d)[0] for d in ud_dates}
            gains = []
            for _ in range(1000):
                pick = np.concatenate([idx[d] for d in rng.choice(ud_dates, len(ud_dates))])
                gains.append(llp[pick].mean() - ll[pick].mean())
            res["gain_ci95"] = [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))]
        else:
            res["gain_ci95"] = None
            res["note"] = "fewer than 20 scored dates: no interval yet"
        summ[h] = res
    (root / "summary.json").write_text(json.dumps({"version": a.version, "scored_rows": len(sc),
                                                   "first_asof": str(sc.asof_session.min()),
                                                   "last_asof": str(sc.asof_session.max()), "by_horizon": summ},
                                                  indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1)[:3000])


if __name__ == "__main__":
    main()
