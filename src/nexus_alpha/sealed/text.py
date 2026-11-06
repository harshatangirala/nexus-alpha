"""News and SEC features for evaluation samples, with the frozen TF-IDF/SVD models of the v1.1 bundle.

Same procedure as scripts/data/build_text_features.py:
* news: articles of the sample's ticker available in the 5 days before the decision time, de-duplicated and clustered
  per sample, features from the cluster representatives;
* SEC: structured filing features and the text features of the latest 8-K (30 days) and 10-Q/10-K (120 days).

Beyond the seal (`beyond_seal=True`) the news index and SEC events are loaded with dev_only=False, which requires the
evaluation gate to be open. Article text for articles not fetched during development is streamed from the corpus into
artifacts/news/text_sealed/ (the development text cache is never modified).
"""
from __future__ import annotations

import glob
import json

import pandas as pd
import pyarrow.parquet as pq

from .. import paths

TEXT_DIRS = ("text", "text_sealed")


def _texts() -> pd.DataFrame:
    files = [f for d in TEXT_DIRS for f in glob.glob(str(paths.ARTIFACTS / "news" / d / "*.parquet"))]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("article_key")


def fetch_missing_texts(sel: pd.DataFrame, have: set, log=print) -> int:
    """Stream the text of selected articles not yet cached, with the development fetcher's per-article processing."""
    import sys
    sys.path.insert(0, str(paths.REPO_ROOT / "scripts" / "data"))
    import fetch_news_text as FT
    need = sel[~sel.article_key.isin(have)].drop_duplicates("article_key")
    if need.empty:
        return 0
    info = json.loads((paths.REPO_ROOT / "data_manifests" / "news_dataset_info.json").read_text(encoding="utf-8"))
    out_dir = paths.ARTIFACTS / "news" / "text_sealed"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for (subset, shard), g in need.groupby(["subset", "shard"]):
        rows = g.groupby("row_group").row_in_group.apply(list).to_dict()
        target = out_dir / f"{subset}__{int(shard):03d}.parquet"
        if target.exists():
            continue
        res = FT.fetch_shard((info["sha"], subset, int(shard), rows), out_dir=out_dir)
        log(f"news text {subset}.{int(shard):03d}: {res[2]} {res[3]} rows")
        n += int(res[3] or 0)
    return n


def news_for(samples: pd.DataFrame, news_tf, beyond_seal: bool = False, fetch: bool = False, log=print):
    """samples: sample_id, ticker, decision_utc. Returns (features indexed by sample_id, selected pairs)."""
    from ..data import news_select as NS
    from ..nlp.aggregate import dedup_and_cluster, news_features
    idx = NS.load_index(dev_only=not beyond_seal)
    sel = NS.select_for_samples(samples[["sample_id", "ticker", "decision_utc"]], idx)
    texts = _texts()
    if fetch:
        if fetch_missing_texts(sel, set(texts.article_key), log):
            texts = _texts()
    no_text = sel[~sel.article_key.isin(set(texts.article_key))].groupby("sample_id").size()
    pairs = sel.merge(texts, on="article_key", how="inner")
    pairs = pairs.drop(columns=[c for c in ("decision_utc",) if c in pairs]).merge(
        samples[["sample_id", "decision_utc"]], on="sample_id")
    pairs["available_utc"] = pd.to_datetime(pairs.available_utc, utc=True)
    pairs["decision_utc"] = pd.to_datetime(pairs.decision_utc, utc=True)
    out = pd.DataFrame(index=pd.Index(samples.sample_id, name="sample_id"))
    if len(pairs):
        rep, raw_n = dedup_and_cluster(pairs)
        feats, rep = news_features(rep, raw_n, news_tf)
        out = out.join(feats)
    if "news_present" not in out:
        out["news_present"] = 0.0
    out["news_present"] = out.news_present.fillna(0.0)
    out["news_n_selected_no_text"] = no_text.reindex(out.index).fillna(0.0).values
    return out, sel


def sec_for(samples: pd.DataFrame, sec_tf, beyond_seal: bool = False, log=print) -> pd.DataFrame:
    from ..data import sec as S
    from ..nlp.aggregate import sec_text_features
    ev = S.load_events(dev_only=not beyond_seal)
    s = samples[["sample_id", "ticker", "decision_utc"]].copy()
    s["decision_utc"] = pd.to_datetime(s.decision_utc, utc=True)
    structured = S.sec_features(s, ev)
    latest = S.latest_event_ids(s, ev)
    need = set(latest.drop(columns="sample_id").stack().dropna())
    path = next((paths.SEC_FULL500_RUN / "S09").glob("*/event_text.parquet"))
    txt = pq.read_table(path, filters=[("event_id", "in", list(need))]).to_pandas().set_index("event_id").text
    txt = txt.fillna("").str.slice(0, 3000)
    log(f"SEC texts loaded: {len(txt):,} of {len(need):,} needed")
    textf = sec_text_features(latest, txt, sec_tf)
    out = pd.concat([structured, textf], axis=1).reindex(samples.sample_id)
    for kind in ("8k", "periodic"):
        out[f"sec_{kind}_present"] = out[f"sec_{kind}_present"].fillna(0.0)
    out["sec_covered"] = out.sec_covered.fillna(0.0)
    out.index.name = "sample_id"
    return out
