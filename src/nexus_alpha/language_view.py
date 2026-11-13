"""Data layer of the dashboard's "Vision + NLP" view (E28, D-053). Display only: nothing here changes a forecast.

Reads the documented E28 run (experiments/), the SEC price-impact expert's per-filing predictions and the ChronoBERT
filing vectors (artifacts/features/). Every lookup is point in time: a filing is shown or compared only with filings
that were public before it.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths

E28_RUN = "20261113T035556_e28_vision_nlp_fusion_b1f899"   # experiments/ folder of the documented E28 run, pinned (never "the newest")
E28_FIRST_RUN = "20261113T025828_e28_vision_nlp_fusion_b1f899"   # superseded by D-054 (SEC expert read timestamps)
FEAT = paths.ARTIFACTS / "features"
GROUP = {"img_W": "vision", "img_D": "vision", "img_H": "vision", "num": "indicators", "news": "language",
         "sec": "language", "news_tx": "language", "sec_tx": "language", "sec_impact": "language", "rules": "indicators"}
LABEL = {"img_W": "weekly chart", "img_D": "daily chart", "img_H": "hourly chart", "num": "indicators",
         "news": "news (TF-IDF, tone)", "sec": "SEC (TF-IDF, tone, items)", "news_tx": "news (FinBERT)",
         "sec_tx": "SEC filings (ChronoBERT)", "sec_impact": "SEC price-impact expert"}
ITEM = {"2.02": "results", "5.02": "officers", "1.01": "material agreement", "8.01": "other events",
        "7.01": "Reg FD", "2.01": "acquisition or disposal", "2.06": "impairment", "4.02": "non-reliance",
        "5.07": "shareholder vote", "1.02": "agreement ended"}


def e28(run: str | None = None) -> dict | None:
    d = paths.REPO_ROOT / "experiments" / (run or E28_RUN)
    p = d / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    g = d / "gates.json"
    m["gates_file"] = json.loads(g.read_text(encoding="utf-8")) if g.exists() else m.get("gates", {})
    if (run or E28_RUN) == E28_RUN:
        m["run_note"] = (f"{E28_RUN} is the result of record (D-054). The first run, {E28_FIRST_RUN}, used three "
                         "timestamp columns in the SEC expert by mistake; it was also NOT KEPT (MM vs BASE -0.00030)")
    return m


def gate_table(m: dict, config: str = "MM", split: str = "valid") -> pd.DataFrame:
    g = (m.get("gates_file") or m.get("gates") or {}).get(config)
    if not g:
        return pd.DataFrame(columns=["block", "label", "group", "gate"])
    return pd.DataFrame({"block": g["names"], "label": [LABEL.get(n, n) for n in g["names"]],
                         "group": [GROUP.get(n, "other") for n in g["names"]], "gate": g[split]})


def config_table(m: dict) -> pd.DataFrame:
    r = m["results"]
    base_v, base_t = r["BASE"]["valid_mean_log_loss"], r["BASE"]["test_mean_log_loss"]
    rows = [{"config": k, "blocks": ", ".join(LABEL.get(b, b) for b in v["modalities"]),
             "valid": v["valid_mean_log_loss"], "test": v["test_mean_log_loss"],
             "valid gain vs BASE": base_v - v["valid_mean_log_loss"], "test gain vs BASE": base_t - v["test_mean_log_loss"]}
            for k, v in r.items()]
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def expert_meta() -> dict | None:
    p = FEAT / "sec_impact" / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@lru_cache(maxsize=1)
def event_predictions() -> pd.DataFrame | None:
    p = FEAT / "sec_impact" / "event_predictions.parquet"
    if not p.exists():
        return None
    e = pd.read_parquet(p)
    e["available_utc"] = pd.to_datetime(e.available_utc, utc=True)
    for h in (1, 5):
        e[f"p_move_h{h}"] = 1 - e[f"pside_h{h}"]
        e[f"lean_h{h}"] = e[f"pup_h{h}"] - e[f"pdown_h{h}"]
    return e


def filing_timeline(ticker: str, asof: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """The expert's view of one stock's filings, up to `asof` (inclusive) when given."""
    e = event_predictions()
    if e is None:
        return pd.DataFrame()
    t = e[e.ticker == ticker.upper()]
    if asof is not None:
        cut = pd.Timestamp(asof)
        cut = cut.tz_localize("UTC") if cut.tzinfo is None else cut.tz_convert("UTC")
        t = t[t.available_utc <= cut + pd.Timedelta(days=1)]
    out = t[["event_type", "available_utc", "p_move_h1", "p_move_h5", "lean_h1", "lean_h5"]].copy()
    out["kind"] = out.event_type.map({"8K": "8-K", "periodic_10Q": "10-Q", "periodic_10K": "10-K"}).fillna(out.event_type)
    return out.sort_values("available_utc")


@lru_cache(maxsize=1)
def _centre() -> np.ndarray | float:
    """Mean ChronoBERT vector of the pre-2016 filings (the PCA centre). Mean-pooled BERT vectors share one dominant
    direction (median cosine to the mean is about 0.99), so similarity is measured after removing it."""
    p = FEAT / "sec_tx_pca.npz"
    return np.load(p)["mean"].astype(np.float32) if p.exists() else 0.0


@lru_cache(maxsize=1)
def _vectors():
    m = FEAT / "sec_tx" / "events.parquet"
    v = FEAT / "sec_tx" / "event_emb.npy"
    if not (m.exists() and v.exists()):
        return None
    meta = pd.read_parquet(m)
    meta["available_utc"] = pd.to_datetime(meta.available_utc, utc=True)
    E = np.load(v).astype(np.float32)
    ok = np.isfinite(E).all(1)
    E = E - _centre()
    E[~ok] = 0
    norm = np.linalg.norm(E, axis=1)
    norm[norm == 0] = 1
    return meta, E / norm[:, None], ok


def encoded_filings(ticker: str) -> pd.DataFrame:
    v = _vectors()
    if v is None:
        return pd.DataFrame()
    meta, _, ok = v
    t = meta[(meta.ticker == ticker.upper()) & ok]
    return t.assign(kind=t.event_type.map({"8K": "8-K", "periodic_10Q": "10-Q", "periodic_10K": "10-K"})) \
        .sort_values("available_utc", ascending=False)


def _top_one_per_firm(ix: np.ndarray, sim: np.ndarray, tickers: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Best match per firm, then the k best firms (a firm's own quarterly filings are near-duplicates of each other)."""
    order = np.argsort(-sim)
    seen, keep = set(), []
    for j in order:
        t = tickers[ix[j]]
        if t not in seen:
            seen.add(t)
            keep.append(j)
            if len(keep) == k:
                break
    keep = np.array(keep, dtype=int)
    return ix[keep], sim[keep]


def similar_filings(event_id: str, k: int = 5, other_firms_only: bool = True, one_per_firm: bool = True) -> pd.DataFrame:
    """The k filings most similar to `event_id` (cosine of ChronoBERT vectors) among filings public BEFORE it; by
    default the best filing of each firm, so one firm's near-identical quarterly filings do not fill the list."""
    v = _vectors()
    if v is None:
        return pd.DataFrame()
    meta, E, ok = v
    i = int(np.flatnonzero(meta.event_id.to_numpy() == event_id)[0])
    cand = ok & (meta.available_utc < meta.available_utc.iloc[i]).to_numpy()
    if other_firms_only:
        cand &= (meta.ticker != meta.ticker.iloc[i]).to_numpy()
    ix = np.flatnonzero(cand)
    if not len(ix):
        return pd.DataFrame()
    sim = E[ix] @ E[i]
    if one_per_firm:
        top, best = _top_one_per_firm(ix, sim, meta.ticker.to_numpy(), k)
    else:
        o = np.argsort(-sim)[:k]
        top, best = ix[o], sim[o]
    out = meta.iloc[top][["event_id", "ticker", "event_type", "available_utc"]].copy()
    out["cosine"] = best
    e = event_predictions()
    if e is not None:
        out["p_move_h1"] = e.p_move_h1.reindex(out.event_id).to_numpy()
    return out.reset_index(drop=True)


def filing_text(event_ids: list[str], chars: int = 600) -> dict[str, str]:
    """Cleaned text (as encoded) of a few filings, read from the SEC study's event text (firm names masked)."""
    import pyarrow.parquet as pq
    from .nlp.transformer_text import clean_filing_text
    try:
        path = next((paths.SEC_FULL500_RUN / "S09").glob("*/event_text.parquet"))
    except StopIteration:
        return {}
    t = pq.read_table(path, filters=[("event_id", "in", list(event_ids))]).to_pandas()
    return {r.event_id: clean_filing_text(r.text)[:chars] for r in t.itertuples()}


@lru_cache(maxsize=1)
def _encoder():
    from .nlp import transformer_text as TT
    return TT.load("cpu")


@lru_cache(maxsize=1)
def _labels() -> pd.DataFrame | None:
    p = FEAT / "sec_impact" / "event_labels.parquet"
    return pd.read_parquet(p) if p.exists() else None


def encode_text(text: str) -> np.ndarray | None:
    """ChronoBERT vector of any text, cleaned and truncated exactly as the filings were (CPU; the first call loads the
    model, which can take a minute)."""
    from .nlp import transformer_text as TT
    t = TT.clean_filing_text(text)
    if not t:
        return None
    tok, mdl = _encoder()
    return TT.encode([t], tok, mdl, batch=1, log=lambda s: None)[0].astype(np.float32)


def analogues(vec: np.ndarray, k: int = 12) -> tuple[pd.DataFrame, dict]:
    """The k historical filings nearest to `vec` (the best one of each firm), with the expert's P(move) and, where read (exits before 2019), what
    each did: moved (UP or DOWN) or stayed SIDEWAYS over the next session. Descriptive evidence, not a forecast."""
    v = _vectors()
    if v is None or vec is None:
        return pd.DataFrame(), {}
    meta, E, ok = v
    q = vec - _centre()
    q = q / (np.linalg.norm(q) or 1.0)
    ix = np.flatnonzero(ok)
    top, best = _top_one_per_firm(ix, E[ix] @ q, meta.ticker.to_numpy(), k)   # firms, not near-duplicate filings
    out = meta.iloc[top][["event_id", "ticker", "event_type", "available_utc"]].copy()
    out["cosine"] = best
    e = event_predictions()
    if e is not None:
        out["p_move_h1"] = e.p_move_h1.reindex(out.event_id).to_numpy()
    lab = _labels()
    if lab is not None:
        y = lab.y_h1.reindex(out.event_id).to_numpy()
        out["moved_next_session"] = np.where(np.isnan(y), np.nan, (y != 1).astype(float))
    moved = out.get("moved_next_session", pd.Series(dtype=float)).dropna()
    base = None
    if lab is not None:
        yl = lab.y_h1.dropna()
        base = float((yl != 1).mean())
    summary = {"k": int(k), "mean_cosine": float(out.cosine.mean()),
               "expert_p_move_mean": float(out.p_move_h1.mean()) if "p_move_h1" in out else None,
               "analogues_with_outcome": int(len(moved)), "share_moved": float(moved.mean()) if len(moved) else None,
               "base_rate_moved": base}
    return out.reset_index(drop=True), summary
