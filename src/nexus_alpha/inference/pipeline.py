"""End-to-end inference: one ticker, one as-of session -> a validated InferenceResult (master spec 16).

Modes
* historical: bars from the sealed TriFrame store (as-of <= 2022-12-30). The result states whether the date lies in the
  training, validation, calibration or test block, because a forecast for a training date is in-sample.
* live: daily (3 years) and hourly (60 days) bars from Yahoo Finance, split-adjusted and not dividend-adjusted like the
  training data; as-of = the last completed session. The model was trained on 2003-2015 data, so a live forecast is an
  extrapolation; SEC filings are not fetched live (marked missing); news comes only from articles the user supplies,
  labelled as inference-time inputs.
Arbitrary 2023-2026 historical dates are not offered: that period is sealed.

Nothing here reads prices from chart pixels. Uploaded screenshots are shown to the user but never parsed for levels.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import ARCHITECTURE_VERSION, paths
from ..cv import backbones as BB
from ..cv import render as R
from ..data import bars as B
from ..data.calendar import calendar, decision_time_utc
from ..indicators import triframe_visible as FV
from ..indicators.core import ewma_sigma, indicators, structured_features
from ..multimodal.fusion import FusionConfig, FusionModel
from ..rules import batch as RB
from ..seal import SEAL_DATE
from ..splits import assign_blocks
from ..targets import HORIZONS
from ..trading.plan import RiskConfig, plan_trade
from .schema import HorizonForecast, InferenceResult, Instrument, ModelInfo

BUNDLES = paths.ARTIFACTS / "bundles"


def _is_v1_stock_bundle(p: Path) -> bool:
    """v1 stock bundles (E03/E08). v2 (E14) and market (E19) bundles have their own loaders."""
    m = p / "bundle_meta.json"
    if not m.exists() or "_e14_" in p.name or "_e19_" in p.name:
        return False
    return json.loads(m.read_text(encoding="utf-8")).get("kind", "v1") == "v1"


def latest_bundle() -> Path | None:
    c = sorted(p for p in BUNDLES.glob("*") if _is_v1_stock_bundle(p)) if BUNDLES.exists() else []
    return c[-1] if c else None


class Bundle:
    """Loads an inference bundle. v1 bundles hold one fusion model; v1.1 bundles (E08) may hold several fusion seeds
    (averaged), one LightGBM per horizon on the raw numeric features, and a blend weight; calibration is applied last."""

    def __init__(self, path: Path):
        import torch
        self.path = Path(path)
        self.meta = json.loads((self.path / "bundle_meta.json").read_text(encoding="utf-8"))
        comp = self.meta.get("components") or {}
        self.w_lgbm = float(comp.get("lgbm_weight", 0.0))
        self.w_log = float(comp.get("logistic_weight", 0.0))
        self.w_fus = float(comp.get("fusion_weight", 1.0 - self.w_lgbm - self.w_log))
        self.logistic = None
        if comp.get("logistic"):
            with open(self.path / comp["logistic"]["file"], "rb") as f:
                self.logistic = pickle.load(f)
        self.models = []
        seeds = comp.get("fusion_seeds") or ([{"file": "fusion.pt"}] if (self.path / "fusion.pt").exists() else [])
        for sd in seeds:
            st = torch.load(self.path / sd["file"], map_location="cpu", weights_only=False)
            m = FusionModel(FusionConfig(**st["cfg"])).eval()
            m.load_state_dict(st["model"])
            self.models.append(m)
        self.cfg = self.models[0].cfg if self.models else None
        self.lgbm = {}
        if comp.get("lgbm"):
            import lightgbm as lgb
            self.lgbm = {h: lgb.Booster(model_file=str(self.path / v["file"])) for h, v in comp["lgbm"].items()}
        with open(self.path / "preprocessor.pkl", "rb") as f:
            self.pre = pickle.load(f)
        with open(self.path / "calibrators.pkl", "rb") as f:
            self.calibrators = pickle.load(f)
        self.news_tf = self._opt("news_tfidf_svd.pkl")
        self.sec_tf = self._opt("sec_tfidf_svd.pkl")
        self.names = list(self.cfg.modality_dims) if (self.cfg and self.w_fus > 0) else ["num"]
        self.raw_num = None                              # set by the pipeline for the LightGBM component

    def _opt(self, fn):
        p = self.path / fn
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def probs(self, x: dict, mask: np.ndarray, calibrated: bool = True) -> np.ndarray:
        import torch
        p_f, q = None, None
        if self.models:
            ps, qs = [], []
            with torch.no_grad():
                for m in self.models:
                    lg, qq, _ = m({k: torch.from_numpy(v) for k, v in x.items()}, torch.from_numpy(mask))
                    ps.append(torch.softmax(lg, -1).numpy()[0])
                    qs.append(qq.numpy()[0])
            # quantiles averaged over seeds, then sorted (monotone rearrangement: the heads are unconstrained and
            # sorting can only reduce pinball loss, Chernozhukov, Fernandez-Val and Galichon 2010)
            p_f, q = np.mean(ps, 0), np.sort(np.mean(qs, 0), axis=-1)
        parts, weights = [], []
        if p_f is not None and self.w_fus > 0:
            parts.append(p_f); weights.append(self.w_fus)
        if self.lgbm and self.w_lgbm > 0:
            if self.raw_num is None:
                raise ValueError("raw numeric features are required for the LightGBM component")
            parts.append(np.stack([self.lgbm[h].predict(self.raw_num)[0] for h in HORIZONS])); weights.append(self.w_lgbm)
        if self.logistic is not None and self.w_log > 0:
            parts.append(np.stack([self.logistic[h].predict_proba(x["num"].astype(np.float64))[0] for h in HORIZONS]))
            weights.append(self.w_log)
        w = np.array(weights) / np.sum(weights)
        p = sum(wi * pi for wi, pi in zip(w, parts))
        if calibrated:
            p = np.stack([self.calibrators[h].transform(p[j:j + 1])[0] for j, h in enumerate(HORIZONS)])
        return p, q


def yahoo_symbol(ticker: str) -> str:
    """Universe tickers use a dot for share classes (BRK.B); Yahoo uses a dash (BRK-B) (audit finding A-23)."""
    return ticker.replace(".", "-")


def fetch_live(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Live Yahoo bars. If NEXUS_LIVE_CACHE names a directory (set by the shadow forecaster for one New York evening),
    the bars are cached there, so v1.1 and v2 forecast from identical bars with half the Yahoo requests. Errors are
    never cached."""
    import os
    cache = os.environ.get("NEXUS_LIVE_CACHE")
    if cache:
        fd, fh = Path(cache) / f"{ticker}_daily.pkl", Path(cache) / f"{ticker}_hourly.pkl"   # exact round trip
        if fd.exists() and fh.exists():
            return pd.read_pickle(fd), pd.read_pickle(fh)
    daily, hourly = _fetch_live_uncached(ticker)
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        daily.to_pickle(fd)
        hourly.to_pickle(fh)
    return daily, hourly


def _fetch_live_uncached(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    ysym = yahoo_symbol(ticker)
    # Explicit start date: "3y" is not a valid Yahoo range and returns no rows (audit finding A-22).
    start = (pd.Timestamp.now(tz="America/New_York").normalize() - pd.Timedelta(days=3 * 365 + 7)).strftime("%Y-%m-%d")
    d = yf.download(ysym, start=start, interval="1d", auto_adjust=False, actions=False, progress=False,
                    multi_level_index=False, threads=False)
    if d is None or len(d) == 0:
        raise ValueError(f"no daily data for {ticker}")
    di = pd.DatetimeIndex(d.index)
    di = di.tz_localize(None) if di.tz is not None else di
    daily = pd.DataFrame({"date": di.normalize(), "open": d["Open"].values, "high": d["High"].values,
                          "low": d["Low"].values, "close": d["Close"].values, "volume": d["Volume"].values}).dropna()
    h = yf.download(ysym, period="60d", interval="60m", auto_adjust=False, actions=False, prepost=False,
                    progress=False, multi_level_index=False, threads=False)
    if h is None or len(h) == 0:
        hourly = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        hi = pd.DatetimeIndex(h.index)
        hi = hi.tz_localize("America/New_York") if hi.tz is None else hi
        hourly = pd.DataFrame({"timestamp": hi.tz_convert("UTC"), "open": h["Open"].values, "high": h["High"].values,
                               "low": h["Low"].values, "close": h["Close"].values, "volume": h["Volume"].values}).dropna()
    now = pd.Timestamp.now(tz="America/New_York")
    if len(daily) and daily.date.iloc[-1] == now.tz_localize(None).normalize() and (now.hour, now.minute) < (16, 5):
        daily = daily.iloc[:-1]                                     # today's session is still forming
    daily = daily[(daily[["open", "high", "low", "close"]] > 0).all(axis=1)].reset_index(drop=True)
    if len(daily) < 130:      # the visible-window features need a 126-session window (data contract)
        raise ValueError(f"only {len(daily)} usable daily bars from Yahoo for {ticker}: insufficient history")
    return daily, hourly


def _articles_frame(articles: list[dict], ticker: str, decision_utc: pd.Timestamp) -> tuple[pd.DataFrame, list[dict]]:
    """User articles -> rows for the shared aggregation. Each needs headline (or text) and published_utc."""
    from ..data import news as N
    rows, listing = [], []
    for i, a in enumerate(articles or []):
        txt = (a.get("text") or "").strip()
        head = (a.get("headline") or N.headline(txt)).strip()[:300]
        pub = pd.to_datetime(a.get("published_utc"), utc=True, errors="coerce")
        precision = "minute" if a.get("time_known", True) else "day"
        _, avail, eff = N.availability(pd.Series([pub]), pd.Series([precision]))
        ok = pd.notna(pub) and avail.iloc[0] <= decision_utc
        listing.append({"index": i, "headline": head, "source": a.get("source"), "published_utc": str(pub),
                        "available_utc": str(avail.iloc[0]), "used": bool(ok),
                        "reason": "available before the decision time" if ok else
                        ("no valid publication time" if pd.isna(pub) else "published after the decision time: excluded"),
                        "ticker_match": a.get("ticker_match", "user-asserted"), "input_type": "inference-time (user supplied)"})
        if ok:
            rows.append({"sample_id": "live", "available_utc": avail.iloc[0], "decision_utc": decision_utc,
                         "headline": head, "lead": txt[:1500], "text_hash": N.text_hash(head + " " + txt),
                         "n_tickers_tagged": 1, "effective_precision": eff.iloc[0]})
    return pd.DataFrame(rows), listing


class NexusPredictor:
    def __init__(self, bundle_path: Path | None = None, kb=None):
        p = bundle_path or latest_bundle()
        if p is None:
            raise FileNotFoundError("No inference bundle found under artifacts/bundles. Run scripts/experiments/e03_fusion.py.")
        self.b = Bundle(p)
        self.kb = kb

    # ---------------------------------------------------------------- inputs
    def _inputs(self, ticker, daily, hourly, k, decision_utc, articles, sec_ctx):
        pre, names = self.b.pre, self.b.names
        x, mask, missing, evidence = {}, [], [], {}
        sb = B.StockBars(ticker, daily=daily, hourly=hourly)
        trip = R.triplet_at(sb, k)
        evidence["charts"] = {"n_valid_bars": trip["n_valid"], "renderer": R.VERSION}
        emb = {}
        img_views = [n.split("_")[1] for n in names if n.startswith("img_")]
        for v in img_views:
            emb[v] = BB.embed(pre.backbone, [trip[v]])
        Z = pre.vpca.transform_embeddings(emb) if img_views else {}
        for n in names:
            if n.startswith("img_"):
                v = n.split("_")[1]
                ok = trip["n_valid"][v] >= {"W": 42, "D": 51, "H": 56}[v]       # >= 40% of the window (data contract)
                x[n] = pre._std(n, Z[v], None) * float(ok)
                mask.append(float(ok))
                if not ok:
                    missing.append(f"{v} chart has only {trip['n_valid'][v]} bars: timeframe treated as missing")
            elif n == "num":
                f = structured_features(daily, hourly, pd.DatetimeIndex([daily.date.iloc[k]]))
                xv = FV.from_series(daily.open, daily.high, daily.low, daily.close, daily.volume, [k])
                row = pd.concat([f.reset_index(drop=True), xv.reset_index(drop=True)], axis=1)
                x[n] = pre._std("num", row.reindex(columns=pre.num_cols).to_numpy(np.float64), None)
                self.b.raw_num = row.reindex(columns=pre.num_cols).astype(float)
                mask.append(1.0)
                evidence["structured_features"] = {c: (None if pd.isna(row[c].iloc[0]) else float(row[c].iloc[0]))
                                                   for c in row.columns if c in pre.num_cols}
                if pd.isna(row.filter(like="h_").iloc[0]).all():
                    missing.append("no hourly bars for the as-of session: hourly indicators missing")
            elif n == "news":
                feats, listing = None, []
                if articles and self.b.news_tf is not None:
                    from ..nlp.aggregate import dedup_and_cluster, news_features
                    af, listing = _articles_frame(articles, ticker, decision_utc)
                    if len(af):
                        rep, raw_n = dedup_and_cluster(af)
                        feats, rep = news_features(rep, raw_n, self.b.news_tf)
                        for item in listing:
                            if item["used"]:
                                m = rep[rep.headline == item["headline"]]
                                if len(m):
                                    item["lm_net_tone"] = float(m.news_lm_net_tone.iloc[0])
                evidence["news"] = listing
                if feats is None or not len(feats):
                    x[n] = np.zeros((1, len(pre.news_cols)), np.float32)
                    mask.append(0.0)
                    missing.append("no news available before the decision time (missing, not neutral)")
                else:
                    x[n] = pre._std("news", feats.reindex(columns=pre.news_cols).to_numpy(np.float64), None)
                    mask.append(1.0)
            elif n == "sec":
                if sec_ctx is not None and sec_ctx.get("features") is not None:
                    x[n] = pre._std("sec", sec_ctx["features"].reindex(columns=pre.sec_cols).to_numpy(np.float64), None)
                    mask.append(1.0)
                    evidence["sec_filings"] = sec_ctx.get("listing", [])
                else:
                    x[n] = np.zeros((1, len(pre.sec_cols)), np.float32)
                    mask.append(0.0)
                    evidence["sec_filings"] = []
                    missing.append((sec_ctx or {}).get("reason", "SEC filings unavailable"))
            elif n == "rules":
                ev = RB.evaluate_ticker(ticker, [daily.date.iloc[k]], daily=daily, hourly=hourly)
                x[n] = pre._std("rules", ev.reindex(columns=pre.rule_cols).to_numpy(np.float64), None)
                mask.append(1.0)
        return x, np.array([mask], np.float32), missing, evidence, trip, emb

    def _sec_context(self, ticker, decision_utc, mode):
        if mode == "live":
            return {"features": None, "reason": "SEC filings are not fetched in live mode (no EDGAR access configured): missing"}
        from ..data import sec as S
        from ..nlp.aggregate import sec_text_features
        import pyarrow.parquet as pq
        ev = S.load_events(dev_only=True)
        e = ev[ev.ticker == ticker]
        if e.empty:
            return {"features": None, "reason": f"{ticker} is outside the SEC study's 500-firm coverage: missing"}
        s = pd.DataFrame({"sample_id": ["live"], "ticker": [ticker], "decision_utc": [decision_utc]})
        st = S.sec_features(s, ev)
        if float(st.sec_covered.iloc[0]) == 0:
            return {"features": None, "reason": "decision date outside SEC coverage: missing"}
        latest = S.latest_event_ids(s, ev)
        ids = list(latest.drop(columns="sample_id").stack().dropna()) if len(latest) else []
        txt = pd.Series(dtype=object)
        if ids:
            path = next((paths.SEC_FULL500_RUN / "S09").glob("*/event_text.parquet"))
            txt = pq.read_table(path, filters=[("event_id", "in", ids)]).to_pandas().set_index("event_id").text.fillna("").str.slice(0, 3000)
        tfeat = sec_text_features(latest if len(latest) else pd.DataFrame({"sample_id": ["live"]}), txt, self.b.sec_tf) \
            if self.b.sec_tf is not None else pd.DataFrame(index=["live"])
        feats = pd.concat([st, tfeat], axis=1)
        recent = e[(e.available_utc <= decision_utc) & (e.available_utc > decision_utc - pd.Timedelta(days=120))].tail(8)
        listing = [{"event_id": r.event_id, "type": r.event_type, "forms": r.forms, "items": r["items"],
                    "acceptance_time": str(r.acceptance_time), "available_utc": str(r.available_utc),
                    "timestamp_quality": r.ts_quality, "accessions": r.accessions,
                    "used_for_text": r.event_id in ids} for _, r in recent.iloc[::-1].iterrows()]
        return {"features": feats, "listing": listing}

    # ---------------------------------------------------------------- main entry
    def predict(self, ticker: str, asof: str | None = None, mode: str = "historical", articles: list[dict] | None = None,
                risk: RiskConfig | None = None, explain: bool = True, occlusion: bool = False) -> InferenceResult:
        risk = risk or RiskConfig()
        limitations = ["Research and paper-trading decision support only; not investment advice; no orders are placed."]
        if mode == "historical":
            daily = B.read_daily(ticker)
            cal = calendar()
            early = set(cal.date[cal.early_close.astype(bool)])
            hourly = R.rth_hourly(B.read_hourly(ticker), early)
            a = pd.Timestamp(asof) if asof else daily.date.iloc[-1]
            if a >= SEAL_DATE:
                raise ValueError("as-of dates on or after 2023-01-03 are sealed")
            k = int(daily.date.searchsorted(a))
            if k >= len(daily) or daily.date.iloc[k] != a:
                raise ValueError(f"{a.date()} is not a trading session for {ticker}")
            block = assign_blocks(pd.Series([a])).iloc[0]
            refit = "refit 2003-2018" in self.b.meta.get("trained_through_block", "")
            insample = ("train", "inner", "valid", None) if refit else ("train", None)
            span = "2003-2018" if refit else "2003-2015"
            limitations.append(f"Historical as-of date in block '{block or 'outside the development blocks'}'"
                               + (f": IN-SAMPLE for the model (trained on {span})." if block in insample else "."))
        elif mode == "live":
            daily, hourly = fetch_live(ticker)
            k = len(daily) - 1
            span = "2003-2018" if "refit 2003-2018" in self.b.meta.get("trained_through_block", "") else "2003-2015"
            limitations.append(f"Live forecast: the model was trained on {span} data, calibrated on 2019; it has not been "
                               "validated on post-2022 data. Live bars come from Yahoo Finance and can differ slightly between requests.")
        else:
            raise ValueError("mode must be 'historical' or 'live'")
        asof_date = daily.date.iloc[k]
        if asof_date <= calendar().date.max() - pd.Timedelta(days=3):
            decision_utc = decision_time_utc(pd.DatetimeIndex([asof_date]))[0]
        else:   # beyond the market calendar: next business day, 09:00 New York, time-zone aware (audit A-30)
            from .. import shadow as _SH
            decision_utc = _SH.decision_utc_for(pd.Timestamp(asof_date))
        data_reasons = []
        if k < 252:
            data_reasons.append("fewer than 252 daily bars: indicators and structure are not warm")
        sec_ctx = self._sec_context(ticker, decision_utc, mode)
        x, mask, missing, evidence, trip, emb = self._inputs(ticker, daily, hourly, k, decision_utc, articles, sec_ctx)
        P, Q = self.b.probs(x, mask)
        skill = self.b.meta.get("direction_skill", {})
        # Return range from the quantile heads (volatility-scaled z, so multiply by sigma_t * sqrt(h)); shown only for
        # horizons whose 10-90% band covered 0.80 +/- 0.08 on validation (scripts/verify/quantile_coverage.py).
        qcov = self.b.meta.get("quantile_coverage", {})
        sig = float(ewma_sigma(daily.close.iloc[: k + 1]).iloc[-1])
        forecasts = {}
        for j, h in enumerate(HORIZONS):
            p = P[j] / P[j].sum()
            show_q = Q is not None and bool(qcov.get(h, {}).get("show_in_app", False)) and np.isfinite(sig) and sig > 0
            scale = sig * np.sqrt(HORIZONS[h])
            forecasts[h] = HorizonForecast(up_probability=float(p[2]), down_probability=float(p[0]),
                                           sideways_probability=float(p[1]),
                                           predicted_direction=["DOWN", "SIDEWAYS", "UP"][int(np.argmax(p))],
                                           expected_return=float(Q[j, 1] * scale) if show_q else None,
                                           return_q10=float(Q[j, 0] * scale) if show_q else None,
                                           return_q90=float(Q[j, 2] * scale) if show_q else None,
                                           uncertainty=float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(3)),
                                           direction_skill_validated=bool(skill.get(h, {}).get("validated", False)))
        # rule events and indicator evidence
        ev = RB.evaluate_ticker(ticker, [asof_date], daily=daily, hourly=hourly).iloc[0]
        setups = json.loads(ev.setups_json)
        trig = {"LONG": json.loads(ev.trigger_long_json), "SHORT": json.loads(ev.trigger_short_json)}
        weekly = {"regime": ev.regime, "permission": ev.permission}
        ind = indicators(daily).iloc[k]
        evidence["indicators"] = {"asof_session": str(asof_date.date()), "timeframe": "1D",
                                  **{c: (None if pd.isna(ind[c]) else float(ind[c])) for c in
                                     ("sma20", "sma50", "macd", "macd_sig", "macd_hist", "rsi14", "bb_up", "bb_mid",
                                      "bb_lo", "atr14", "vol_avg20", "vol_ratio")}, "close": float(daily.close.iloc[k])}
        evidence["rules"] = {"weekly_regime": ev.regime, "weekly_permission": ev.permission, "daily_setups": setups,
                             "hourly_triggers": trig}
        # diagnostics
        if explain:
            from ..explainability.attribution import chart_occlusion, leave_one_modality_out
            score = lambda xx, mm: self.b.probs(xx, mm)[0]
            evidence["modality_diagnostics"] = {"method": "leave-one-modality-out (model diagnostic, not causal)",
                                                "horizons": list(HORIZONS),
                                                "by_modality": leave_one_modality_out(score, x, mask, self.b.names)}
            if occlusion:
                pre = self.b.pre

                def embed_view(v, img):
                    return BB.embed(pre.backbone, [img])

                def score_from_views(e):
                    z = pre.vpca.transform_embeddings({v: e[v] for v in ("W", "D", "H")})
                    xx = dict(x)
                    for v in ("W", "D", "H"):
                        if f"img_{v}" in xx:
                            xx[f"img_{v}"] = pre._std(f"img_{v}", z[v], None) * mask[0, self.b.names.index(f"img_{v}")]
                    return self.b.probs(xx, mask)[0]
                evidence["chart_occlusion"] = chart_occlusion({v: trip[v] for v in ("W", "D", "H")}, embed_view, score_from_views)
        if self.kb is not None:
            from ..rag.explain import citations_for
            evidence["rag_sources"] = citations_for(self.kb, ev, setups, trig)
        # confluence (supporting vs conflicting, per the 5D horizon as the reference)
        sup, con = [], []
        p5 = forecasts["5D"]
        lean = "UP" if p5.up_probability > p5.down_probability else "DOWN"
        for s in setups:
            (sup if (s["direction"] == "LONG") == (lean == "UP") else con).append(f"daily setup {s['family']} ({', '.join(s['rule_ids'])})")
        if ev.regime in ("BULL_TREND", "BEAR_TREND"):
            (sup if (ev.regime == "BULL_TREND") == (lean == "UP") else con).append(f"weekly regime {ev.regime}")
        for side, t in trig.items():
            if t.get("trigger", "NO_TRIGGER") != "NO_TRIGGER":
                (sup if (side == "LONG") == (lean == "UP") else con).append(f"hourly {t['trigger']} ({side.lower()})")
        confluence = {"reference_horizon": "5D", "model_lean": lean, "supporting_signals": sup, "conflicting_signals": con,
                      "missing_information": missing,
                      "summary": (f"Calibrated 5D probabilities: UP {p5.up_probability:.2f}, DOWN {p5.down_probability:.2f}, "
                                  f"SIDEWAYS {p5.sideways_probability:.2f}. "
                                  + ("Direction skill at 5D is NOT validated, so the lean is informational only."
                                     if not p5.direction_skill_validated else "Direction skill at 5D passed validation."))}
        plans = {}
        for j, (h, n) in enumerate(HORIZONS.items()):
            f = forecasts[h]
            plans[h] = plan_trade(h, n, {"up": f.up_probability, "down": f.down_probability, "side": f.sideways_probability},
                                  calibrated=True, skill_validated=bool(f.direction_skill_validated), levels_available=True,
                                  weekly=weekly, setups=setups, trigger_by_side=trig, atr=float(ev.atr14),
                                  close=float(ev.close), risk=risk, data_gate_reasons=data_reasons)
        if any(p.decision in ("LONG", "SHORT") for p in plans.values()):
            rb = paths.REPO_ROOT / "reports" / "backtest"
            bt, tag = (rb / "e04_backtest_v1.1.json", "E04 v1.1") if (rb / "e04_backtest_v1.1.json").exists()                 else (rb / "e04_backtest.json", "E04 v1")
            if bt.exists():
                b = json.loads(bt.read_text(encoding="utf-8"))
                g = {k.split("|")[1]: v["sharpe"] for k, v in b["test_final"].items()
                     if k.startswith("nexus_gated") and k.endswith("|base")}
                limitations.append(f"Backtest evidence ({tag}, 2020-2022, base costs): this gated policy had Sharpe "
                                   + ", ".join(f"{h} {s:.2f}" for h, s in g.items())
                                   + f" against {b['benchmark_ew_survivors']['test']['sharpe']:.2f} for equal-weight "
                                   "buy-and-hold, with confidence intervals including zero. Treat any LONG/SHORT plan "
                                   "as an unvalidated research candidate.")
        dq = "validated" if not missing and not data_reasons else ("insufficient" if data_reasons else "partial")
        res = InferenceResult(
            instrument=Instrument(symbol=ticker, forecast_timestamp=decision_utc.isoformat(), asof_session=str(asof_date.date()),
                                  reference_price=float(daily.close.iloc[k]), reference_price_source=
                                  ("sealed TriFrame bars" if mode == "historical" else "Yahoo Finance") + " (as-of close)",
                                  data_quality=dq),
            model=ModelInfo(architecture_version=self.b.meta.get("architecture_version", ARCHITECTURE_VERSION),
                            checkpoint_ids=[self.b.meta["bundle_id"], self.b.pre.backbone, self.b.meta.get("recipe", "fusion")],
                            calibration_status="calibrated", bundle_id=self.b.meta["bundle_id"],
                            trained_through="2018-12-31" if "refit 2003-2018" in self.b.meta.get("trained_through_block", "") else "2015-12-31"),
            forecasts=forecasts, evidence=evidence, confluence=confluence, trade_plans=plans, limitations=limitations)
        return res
