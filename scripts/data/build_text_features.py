"""Per-sample news and SEC features for the s500 frame (CPU profile: classical text representations).

News (per sample, articles available in the 5 days before the decision time):
  exact duplicates (normalised text hash) dropped; near-duplicate headlines (3-word shingle Jaccard >= 0.8 within 2 days)
  clustered so syndicated copies count once; the 7 most recent cluster representatives kept. Features: counts (raw,
  clusters), hours since the latest article, share of single-ticker tags (match specificity), share with minute-precision
  timestamps, mean/min Loughran-McDonald tone, mean TF-IDF->SVD(64) vector (fitted on articles available before
  2016-01-01 only). news_present = 1 if at least one article, else every news feature is NaN.
SEC: structured filing features (nexus_alpha.data.sec) + TF-IDF->SVD(32) and LM tone of the latest 8-K (30 days) and of
  the latest 10-Q/10-K (120 days), fitted on events available before 2016-01-01 only.

Outputs: artifacts/features/news_features.parquet, artifacts/features/sec_features.parquet
Run: .venv/Scripts/python -I scripts/data/build_text_features.py
"""
from __future__ import annotations

import glob
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nexus_alpha import paths
from nexus_alpha.data import news as N
from nexus_alpha.data import sec as S
from nexus_alpha.data.news_select import MAX_ARTICLES
from nexus_alpha.nlp.text_features import TfidfSvd, lm_tone_frame

FIT_BEFORE = pd.Timestamp("2016-01-01", tz="UTC")


def build_news(fr: pd.DataFrame, log=print) -> pd.DataFrame:
    pairs = pd.read_parquet(paths.ARTIFACTS / "news" / "s500_selected_pairs.parquet",
                            columns=["sample_id", "article_key", "available_utc", "n_tickers_tagged", "effective_precision"])
    texts = pd.concat([pd.read_parquet(p) for p in glob.glob(str(paths.ARTIFACTS / "news" / "text" / "*.parquet"))],
                      ignore_index=True).drop_duplicates("article_key")
    pairs = pairs.merge(texts, on="article_key", how="inner")
    log(f"news pairs with text: {len(pairs):,} (articles {pairs.article_key.nunique():,})")
    pairs = pairs.merge(fr[["sample_id", "decision_utc"]], on="sample_id")
    pairs["available_utc"] = pd.to_datetime(pairs.available_utc, utc=True)
    pairs = pairs.sort_values(["sample_id", "available_utc"])
    # exact duplicates within a sample
    pairs = pairs.drop_duplicates(["sample_id", "text_hash"], keep="first")
    # near-duplicate clusters within a sample
    reps = []
    for sid, g in pairs.groupby("sample_id", sort=False):
        if len(g) == 1:
            reps.append(g.assign(cluster=0, n_cluster=1))
            continue
        cid = N.cluster_near_duplicates(g.headline.tolist(), g.available_utc.tolist())
        g = g.assign(cluster=cid)
        sizes = g.cluster.value_counts()
        rep = g.drop_duplicates("cluster", keep="first").copy()
        rep["n_cluster"] = rep.cluster.map(sizes).values
        reps.append(rep)
    rep = pd.concat(reps, ignore_index=True)
    raw_n = pairs.groupby("sample_id").size()
    rep = rep.sort_values(["sample_id", "available_utc"]).groupby("sample_id").tail(MAX_ARTICLES)
    rep["doc"] = rep.headline.fillna("") + ". " + rep.lead.fillna("")
    fit_docs = rep.loc[rep.available_utc < FIT_BEFORE, "doc"]
    fit_docs = fit_docs.sample(min(len(fit_docs), 150_000), random_state=7)
    tf = TfidfSvd(64).fit(fit_docs)
    log(f"news TF-IDF fitted on {len(fit_docs):,} pre-2016 articles; SVD explained {tf.explained:.3f}")
    V = pd.DataFrame(tf.transform(rep.doc), columns=[f"news_svd_{i}" for i in range(64)], index=rep.index)
    T = lm_tone_frame(rep.doc).add_prefix("news_")
    rep = pd.concat([rep, V, T], axis=1)
    rep["hours_before"] = (pd.to_datetime(rep.decision_utc, utc=True) - rep.available_utc).dt.total_seconds() / 3600
    g = rep.groupby("sample_id")
    feats = pd.concat([
        g[[f"news_svd_{i}" for i in range(64)]].mean(),
        g[["news_lm_negative", "news_lm_positive", "news_lm_uncertainty", "news_lm_litigious", "news_lm_net_tone"]].mean(),
        g.news_lm_net_tone.min().rename("news_lm_net_tone_min"),
        g.size().rename("news_n_clusters"),
        raw_n.rename("news_n_raw"),
        g.hours_before.min().rename("news_hours_since_latest"),
        g.n_tickers_tagged.apply(lambda s: (s == 1).mean()).rename("news_single_ticker_share"),
        g.effective_precision.apply(lambda s: (s == "minute").mean()).rename("news_minute_precision_share"),
    ], axis=1)
    feats["news_present"] = 1.0
    out = fr[["sample_id"]].merge(feats, left_on="sample_id", right_index=True, how="left").set_index("sample_id")
    out["news_present"] = out.news_present.fillna(0.0)
    out.attrs["tfidf_explained"] = tf.explained
    return out, rep[["sample_id", "article_key", "available_utc", "headline", "n_cluster", "news_lm_net_tone"]]


def build_sec(fr: pd.DataFrame, log=print) -> pd.DataFrame:
    ev = S.load_events(dev_only=True)
    structured = S.sec_features(fr[["sample_id", "ticker", "decision_utc"]], ev)
    latest = S.latest_event_ids(fr[["sample_id", "ticker", "decision_utc"]], ev)
    need = set(latest.drop(columns="sample_id").stack().dropna())
    fit_ids = set(ev.loc[ev.available_utc < FIT_BEFORE, "event_id"].sample(min(60_000, int((ev.available_utc < FIT_BEFORE).sum())), random_state=7))
    path = next((paths.SEC_FULL500_RUN / "S09").glob("*/event_text.parquet"))
    txt = pq.read_table(path, filters=[("event_id", "in", list(need | fit_ids))]).to_pandas().set_index("event_id").text
    txt = txt.fillna("").str.slice(0, 3000)
    log(f"SEC texts loaded: {len(txt):,} (needed {len(need):,}, fit {len(fit_ids):,})")
    tf = TfidfSvd(32).fit(txt.reindex(list(fit_ids)).dropna())
    log(f"SEC TF-IDF SVD explained {tf.explained:.3f}")
    ids = sorted(need)
    vec = pd.DataFrame(tf.transform(txt.reindex(ids).fillna("")), index=ids)
    tone = lm_tone_frame(txt.reindex(ids).fillna(""))
    tone.index = ids
    parts = [structured]
    for kind in ("8k", "periodic"):
        col = f"event_{kind}"
        m = latest.set_index("sample_id")[col] if col in latest else pd.Series(dtype=object)
        V = vec.reindex(m.values).set_axis(m.index)
        V.columns = [f"sec_{kind}_svd_{i}" for i in range(32)]
        Tn = tone.reindex(m.values).set_axis(m.index)[["lm_negative", "lm_positive", "lm_uncertainty", "lm_net_tone"]]
        Tn.columns = [f"sec_{kind}_{c}" for c in Tn.columns]
        P = pd.Series(m.notna().astype(float).values, index=m.index, name=f"sec_{kind}_present")
        parts += [V, Tn, P]
    out = pd.concat(parts, axis=1).reindex(fr.sample_id)
    for kind in ("8k", "periodic"):
        out[f"sec_{kind}_present"] = out[f"sec_{kind}_present"].fillna(0.0)
    out["sec_covered"] = out.sec_covered.fillna(0.0)
    out.attrs["tfidf_explained"] = tf.explained
    return out


def main() -> int:
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "decision_utc", "block"])
    out = paths.ARTIFACTS / "features"
    out.mkdir(parents=True, exist_ok=True)
    news, reps = build_news(fr)
    news.to_parquet(out / "news_features.parquet")
    reps.to_parquet(out / "news_representatives.parquet", index=False)
    print(f"news features: {news.news_present.mean():.3f} of samples have news ({time.time() - t0:.0f} s)", flush=True)
    sec = build_sec(fr)
    sec.to_parquet(out / "sec_features.parquet")
    summ = {"news_present_share": float(news.news_present.mean()), "news_tfidf_explained": news.attrs.get("tfidf_explained"),
            "sec_covered_share": float(sec.sec_covered.mean()), "sec_8k_present_share": float(sec.sec_8k_present.mean()),
            "sec_periodic_present_share": float(sec.sec_periodic_present.mean()),
            "sec_tfidf_explained": sec.attrs.get("tfidf_explained"), "elapsed_s": round(time.time() - t0, 1)}
    (paths.REPO_ROOT / "data_manifests" / "text_features_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
