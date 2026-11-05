"""Per-sample text aggregation shared by training (scripts/data/build_text_features.py) and inference, so both compute
identical features.

News input: one row per article already filtered to the sample's window, with columns sample_id, available_utc,
decision_utc, headline, lead, text_hash, n_tickers_tagged, effective_precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import news as N
from ..data.news_select import MAX_ARTICLES
from .text_features import lm_tone_frame

NEWS_TONES = ["news_lm_negative", "news_lm_positive", "news_lm_uncertainty", "news_lm_litigious", "news_lm_net_tone"]
SEC_TONES = ["lm_negative", "lm_positive", "lm_uncertainty", "lm_net_tone"]


def dedup_and_cluster(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    pairs = pairs.sort_values(["sample_id", "available_utc"])
    raw_n = pairs.groupby("sample_id").size()
    pairs = pairs.drop_duplicates(["sample_id", "text_hash"], keep="first")
    reps = []
    for _, g in pairs.groupby("sample_id", sort=False):
        if len(g) == 1:
            reps.append(g.assign(cluster=0, n_cluster=1))
            continue
        cid = N.cluster_near_duplicates(g.headline.tolist(), g.available_utc.tolist())
        g = g.assign(cluster=cid)
        sizes = g.cluster.value_counts()
        rep = g.drop_duplicates("cluster", keep="first").copy()
        rep["n_cluster"] = rep.cluster.map(sizes).values
        reps.append(rep)
    rep = pd.concat(reps, ignore_index=True) if reps else pairs.assign(cluster=[], n_cluster=[])
    rep = rep.sort_values(["sample_id", "available_utc"]).groupby("sample_id").tail(MAX_ARTICLES)
    rep["doc"] = rep.headline.fillna("") + ". " + rep.lead.fillna("")
    return rep.reset_index(drop=True), raw_n


def news_features(rep: pd.DataFrame, raw_n: pd.Series, tf) -> pd.DataFrame:
    V = pd.DataFrame(tf.transform(rep.doc), columns=[f"news_svd_{i}" for i in range(tf.svd.n_components)], index=rep.index)
    T = lm_tone_frame(rep.doc).add_prefix("news_")
    rep = pd.concat([rep, V, T], axis=1)
    rep["hours_before"] = (pd.to_datetime(rep.decision_utc, utc=True) - pd.to_datetime(rep.available_utc, utc=True)).dt.total_seconds() / 3600
    g = rep.groupby("sample_id")
    feats = pd.concat([
        g[list(V.columns)].mean(), g[NEWS_TONES].mean(),
        g.news_lm_net_tone.min().rename("news_lm_net_tone_min"),
        g.size().rename("news_n_clusters"), raw_n.rename("news_n_raw"),
        g.hours_before.min().rename("news_hours_since_latest"),
        g.n_tickers_tagged.apply(lambda s: (s == 1).mean()).rename("news_single_ticker_share"),
        g.effective_precision.apply(lambda s: (s == "minute").mean()).rename("news_minute_precision_share"),
    ], axis=1)
    feats["news_present"] = 1.0
    return feats, rep


def sec_text_features(latest: pd.DataFrame, texts: pd.Series, tf) -> pd.DataFrame:
    """latest: sample_id, event_8k, event_periodic (event ids or NaN); texts: event_id -> first 3,000 characters."""
    ids = sorted(set(latest.drop(columns="sample_id").stack().dropna()))
    vec = pd.DataFrame(tf.transform(texts.reindex(ids).fillna("")), index=ids)
    tone = lm_tone_frame(texts.reindex(ids).fillna(""))
    tone.index = ids
    parts = []
    L = latest.set_index("sample_id")
    for kind in ("8k", "periodic"):
        col = f"event_{kind}"
        m = L[col] if col in L else pd.Series(np.nan, index=L.index, dtype=object)
        V = vec.reindex(m.values).set_axis(m.index)
        V.columns = [f"sec_{kind}_svd_{i}" for i in range(V.shape[1])]
        Tn = tone.reindex(m.values).set_axis(m.index)[SEC_TONES]
        Tn.columns = [f"sec_{kind}_{c}" for c in Tn.columns]
        parts += [V, Tn, pd.Series(m.notna().astype(float).values, index=m.index, name=f"sec_{kind}_present")]
    return pd.concat(parts, axis=1)
