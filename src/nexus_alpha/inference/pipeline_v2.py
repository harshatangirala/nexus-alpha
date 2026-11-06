"""v2 inference: every v2 horizon (1D-20D, 3M, 6M, 1Y) from the frozen E14 bundle.

Inputs are assembled exactly as in training:
* tabular columns: the v1 numeric block, `rel_*`, `mkt_*`, `gmd_*`, `fred_*` and text features as chosen by E11/E16;
* the fusion modalities: the v1 inputs plus the `ctx` block;
* the 63-bar sequences.
Historical mode reads the sealed stores (as-of dates before 2023-01-03). Live mode reads Yahoo (stock, SPY, ETFs, market
series), FRED's current vintages, and GMD's latest usable year.
The blend uses the frozen group weights; calibrators come from 2019; quantile ranges are shown only where validation
coverage passed; the direction-skill flags come from validation.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..targets_v2 import HORIZONS_V2

H = list(HORIZONS_V2)


def latest_v2_bundle() -> Path | None:
    c = sorted((paths.ARTIFACTS / "bundles").glob("*_e14_blend_freeze_v2_*"))
    return c[-1] if c else None


def decision_utc(asof_date: pd.Timestamp) -> pd.Timestamp:
    """09:00 New York on the next session (D-003), as in v1 and in training: the market calendar inside its range,
    otherwise the shadow test's rule (next business day, time-zone aware)."""
    from .. import shadow as SH
    from ..data.calendar import calendar, decision_time_utc
    if asof_date <= calendar().date.max() - pd.Timedelta(days=3):
        return decision_time_utc(pd.DatetimeIndex([asof_date]))[0]
    return SH.decision_utc_for(pd.Timestamp(asof_date))


class NexusPredictorV2:
    def __init__(self, bundle_dir: Path | None = None):
        import lightgbm as lgb
        self.dir = Path(bundle_dir or latest_v2_bundle())
        self.meta = json.loads((self.dir / "bundle_meta.json").read_text(encoding="utf-8"))
        self._ctx_cache: dict = {}
        self.weights = {h: self.meta["weights"][g] for g, hs in self.meta["groups"].items() for h in hs}
        comp = self.meta["components"]
        self.cols = comp["tabular_cols"]
        self.lgbm = {h: lgb.Booster(model_file=str(self.dir / f"lgbm_{h}.txt")) for h in H
                     if (self.dir / f"lgbm_{h}.txt").exists()}
        lp = self.dir / "logistic.pkl"
        self.logit = pickle.load(open(lp, "rb")) if lp.exists() else {}
        with open(self.dir / "calibrators.pkl", "rb") as f:
            self.cal = pickle.load(f)
        self.fusion = None
        if comp.get("fusion_seeds"):
            import torch
            from ..multimodal.fusion import FusionConfig, FusionModel
            with open(self.dir / "fusion_preprocessor.pkl", "rb") as f:
                self.fpre = pickle.load(f)
            self.fusion = []
            for s in comp["fusion_seeds"]:
                ck = torch.load(self.dir / f"fusion_seed{s['seed']}.pt", map_location="cpu", weights_only=False)
                m = FusionModel(FusionConfig(**ck["cfg"]))
                m.load_state_dict(ck["model"])
                self.fusion.append(m.eval())
        self.seq = {}
        if comp.get("sequence"):
            import torch
            from .. import seqmodels as SM
            sq = comp["sequence"]
            for arch, seeds in sq["seeds"].items():
                ms = []
                for s in seeds:
                    m = SM.TCN(len(sq["channels"]), len(H)) if arch == "tcn" else SM.GRUNet(len(sq["channels"]), len(H))
                    m.load_state_dict(torch.load(self.dir / f"{arch}_seed{s['seed']}.pt", map_location="cpu"))
                    ms.append(m.eval())
                self.seq[arch] = ms
            self.seq_mu, self.seq_sd = np.array(sq["channel_mean"]), np.array(sq["channel_sd"])
        self._v1 = None

    # ------------------------------------------------------------------ data
    def _v1pred(self):
        from .. import shadow as SH
        from .pipeline import NexusPredictor
        if self._v1 is None:
            # a private v1 predictor used only to assemble inputs; with a v2 fusion it carries the v2 refit preprocessor
            self._v1 = NexusPredictor(SH.verify_frozen("v1.1")[0])
            if self.fusion is not None:
                self._v1.b.pre = self.fpre["pre"]
        return self._v1

    def _bars(self, ticker, mode, asof):
        from ..cv import render as R
        from ..data import bars as B
        from ..data.calendar import calendar
        from .pipeline import fetch_live
        if mode == "historical":
            d = B.read_daily(ticker)
            early = set(calendar().date[calendar().early_close.astype(bool)])
            h = R.rth_hourly(B.read_hourly(ticker), early)
            k = int(d.date.searchsorted(pd.Timestamp(asof)))
            if k >= len(d) or d.date.iloc[k] != pd.Timestamp(asof):
                raise ValueError(f"{asof} is not a session for {ticker}")
            return d, h, k
        d, h = fetch_live(ticker)
        return d, h, len(d) - 1

    def _context(self, mode, asof_date):
        from ..macro import context as CX, fred as FR, gmd as G
        if mode == "historical":
            from ..data import etf_bars as EB
            etf = EB.closes([t for t in CX.ETF_LIST if t in EB.available()])
            spy = EB.read_daily("SPY")
            mk = CX.dev_date_features().loc[:asof_date].iloc[-1]
            fr = FR.asof_features(pd.DatetimeIndex([asof_date])).iloc[0] if (paths.ARTIFACTS / FR.STORE).exists() else pd.Series(dtype=float)
        else:
            start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
            etf = pd.concat({t: CX._yahoo_daily(t, start).set_index("date").close for t in CX.ETF_LIST}, axis=1)
            spy = CX._yahoo_daily("SPY", start)
            mk = CX.live_date_features().dropna(how="all").iloc[-1]
            fr = FR.live_features(asof_date) if FR.have_key() else pd.Series(dtype=float)
        gm = G.asof_features(pd.DatetimeIndex([asof_date])).iloc[0]
        return etf, spy, mk, gm, fr

    # ------------------------------------------------------------------ predict
    def predict(self, ticker: str, asof: str | None = None, mode: str = "historical", articles: list[dict] | None = None):
        from .. import seqmodels as SM
        from ..macro import features as MF
        from .pipeline import _articles_frame
        v1 = self._v1pred()
        daily, hourly, k = self._bars(ticker, mode, asof)
        asof_date = daily.date.iloc[k]
        key = (mode, pd.Timestamp(asof_date))
        if key not in self._ctx_cache:       # one context per session: identical for every ticker (500 per night)
            self._ctx_cache[key] = self._context(mode, asof_date)
        etf, spy, mk, gm, fr = self._ctx_cache[key]
        dec = decision_utc(asof_date)
        # v1 inputs (images, numeric, news, SEC off in live, rules) with the v2 fusion preprocessor's statistics
        row = {}
        v1.b.raw_num = None
        sec_ctx = v1._sec_context(ticker, dec, mode)
        x1, mask1, missing, evidence, _, _ = v1._inputs(ticker, daily, hourly, k, dec, articles, sec_ctx)
        row.update(v1.b.raw_num.iloc[0].to_dict())
        rel = MF.stock_relative(daily[["date", "close"]], etf)
        row.update(rel.iloc[k].to_dict())
        row.update(mk.to_dict())
        row.update({c: v for c, v in gm.items() if c != "gmd_data_year"})
        row.update(fr.to_dict())
        if articles and any(c.startswith(("news_", "fb_")) for c in self.cols):
            af, _ = _articles_frame(articles, ticker, dec)
            if len(af):
                from ..nlp.aggregate import dedup_and_cluster, news_features
                rep, raw_n = dedup_and_cluster(af)
                feats, rep = news_features(rep, raw_n, v1.b.news_tf)
                row.update(feats.iloc[0].to_dict())
                if any(c.startswith("fb_") for c in self.cols):
                    row.update(self._finbert(rep))
        X = pd.DataFrame([row]).reindex(columns=self.cols).astype(float)
        P = np.zeros((len(H), 3))
        parts = {}
        for j, h in enumerate(H):
            w = self.weights[h]
            if w.get("lgbm", 0) and h in self.lgbm:
                p = self.lgbm[h].predict(X)[0]
                P[j] += w["lgbm"] * p
                parts.setdefault("lgbm", {})[h] = p.tolist()
            if w.get("logistic", 0) and h in self.logit:
                m = self.logit[h]
                Z = ((X[m["cols"]].fillna(m["median"]) - m["mu"]) / m["sd"]).clip(-8, 8).to_numpy(np.float64)
                p = m["model"].predict_proba(Z)[0]
                P[j] += w["logistic"] * p
                parts.setdefault("logistic", {})[h] = p.tolist()
        Q = None
        if self.fusion is not None:
            import torch
            fp = self.fpre
            xx = {}
            for n in fp["names"]:
                if n == "ctx":
                    xx["ctx"] = self._ctx_vec(row, fp)
                else:
                    xx[n] = x1[n]
            mk_ = np.concatenate([mask1, np.ones((1, 1), np.float32)], 1) if "ctx" in fp["names"] else mask1
            with torch.no_grad():
                outs = [m({k_: torch.from_numpy(np.asarray(v, np.float32)) for k_, v in xx.items()},
                          torch.from_numpy(mk_)) for m in self.fusion]
            pf = np.mean([torch.softmax(o[0], -1).numpy()[0] for o in outs], 0)
            Q = np.sort(np.mean([o[1].numpy()[0] for o in outs], 0), axis=-1)
            for j, h in enumerate(H):
                P[j] += self.weights[h].get("fusion", 0) * pf[j]
        if self.seq:
            ch = SM.channels(daily, spy)
            Xs = SM.windows(ch, pd.DatetimeIndex([asof_date]))
            Xs = np.clip(np.nan_to_num((Xs - self.seq_mu) / self.seq_sd, nan=0.0), -8, 8).astype(np.float32)
            for arch, ms in self.seq.items():
                ps = np.mean([SM.predict(m, Xs)[0] for m in ms], 0)
                for j, h in enumerate(H):
                    P[j] += self.weights[h].get(arch, 0) * ps[j]
        from ..indicators.core import ewma_sigma
        sig = float(ewma_sigma(daily.close.iloc[: k + 1]).iloc[-1])
        out = {}
        for j, h in enumerate(H):
            p = self.cal[h].transform(P[j:j + 1])[0]
            p = p / p.sum()
            cov = self.meta.get("quantile_coverage", {}).get(h, {})
            show = Q is not None and cov.get("show_in_app", False) and np.isfinite(sig) and sig > 0
            sc = sig * np.sqrt(HORIZONS_V2[h])
            out[h] = {"p_down": float(p[0]), "p_side": float(p[1]), "p_up": float(p[2]),
                      "ret_q10": float(Q[j, 0] * sc) if show else None, "ret_q50": float(Q[j, 1] * sc) if show else None,
                      "ret_q90": float(Q[j, 2] * sc) if show else None,
                      "direction_skill_validated": bool(self.meta["direction_skill"].get(h, {}).get("validated", False))}
        return {"ticker": ticker, "asof_session": str(asof_date.date()), "mode": mode, "bundle_id": self.meta["bundle_id"],
                "reference_close": float(daily.close.iloc[k]), "forecasts": out, "missing": missing,
                "features_missing_share": float(X.isna().mean(axis=1).iloc[0]), "components": parts}

    def _ctx_vec(self, row: dict, fp: dict) -> np.ndarray:
        cols = fp["ctx_cols"]
        num = [c for c in cols if c != "rel_sector_id"]
        mu, sd = fp["ctx_mu"], fp["ctx_sd"]
        z = np.clip(np.nan_to_num((np.array([row.get(c, np.nan) for c in num], float) - mu) / sd, nan=0.0), -8, 8)
        if "rel_sector_id" in cols:
            sid = row.get("rel_sector_id", np.nan)
            z = np.concatenate([z, [(1.0 if sid == k else 0.0) for k in range(9)]])
        return z[None, :].astype(np.float32)

    def _finbert(self, rep: pd.DataFrame) -> dict:
        import sys
        sys.path.insert(0, str(paths.REPO_ROOT / "scripts" / "data"))
        import build_finbert_features as FB
        pca = np.load(paths.ARTIFACTS / "features" / "finbert_pca.npz")
        heads = rep.headline.fillna("").tolist()
        P, E, info = FB.encode(heads, batch=32, threads=4)
        lab = info["id2label"]
        ip = [k for k, v in lab.items() if v == "positive"][0]
        ineg = [k for k, v in lab.items() if v == "negative"][0]
        w = rep.get("n_cluster", pd.Series(1, index=rep.index)).fillna(1).clip(lower=1).to_numpy(float)
        tone = P[:, ip] - P[:, ineg]
        mean_emb = (E.astype(float) * w[:, None]).sum(0) / w.sum()
        z = (mean_emb - pca["mean"]) @ pca["components"].T
        return {"fb_tone_mean": float(np.average(tone, weights=w)), "fb_tone_min": float(tone.min()),
                "fb_tone_max": float(tone.max()), "fb_neg_share": float((P[:, ineg] > 0.5).mean()), "fb_n": float(len(heads)),
                "fb_present": 1.0, **{f"fb_pc{i}": float(z[i]) for i in range(16)}}
