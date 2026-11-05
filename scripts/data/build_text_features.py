"""Per-sample news and SEC features for the s500 frame (CPU profile: classical text representations).

News (per sample, articles available in the 5 days before the decision time): exact duplicates (normalised text hash)
dropped; near-duplicate headlines (3-word shingle Jaccard >= 0.8 within 2 days) clustered so syndicated copies count
once; the 7 most recent cluster representatives kept. Features: counts (raw, clusters), hours since the latest article,
share of single-ticker tags (match specificity), share with minute-precision timestamps, mean/min Loughran-McDonald tone,
mean TF-IDF->SVD(64) vector fitted on articles available before 2016-01-01 only. news_present = 1 if at least one
article WITH TEXT, else NaN features. Articles whose text could not be fetched are counted in news_n_selected_no_text.
SEC: structured filing features (nexus_alpha.data.sec) + TF-IDF->SVD(32) and LM tone of the latest 8-K (30 days) and of
the latest 10-Q/10-K (120 days), fitted on events available before 2016-01-01 only.
The aggregation code is shared with inference (nexus_alpha.nlp.aggregate).

Outputs: artifacts/features/{news_features,sec_features}.parquet, news/sec TF-IDF models (.pkl) for the bundle.
Run: .venv/Scripts/python -I scripts/data/build_text_features.py
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

from nexus_alpha import paths
from nexus_alpha.data import sec as S
from nexus_alpha.nlp.aggregate import dedup_and_cluster, news_features, sec_text_features
from nexus_alpha.nlp.text_features import TfidfSvd

FIT_BEFORE = pd.Timestamp("2016-01-01", tz="UTC")
OUT = paths.ARTIFACTS / "features"


def build_news(fr: pd.DataFrame, log=print):
    pairs = pd.read_parquet(paths.ARTIFACTS / "news" / "s500_selected_pairs.parquet",
                            columns=["sample_id", "article_key", "available_utc", "n_tickers_tagged", "effective_precision"])
    texts = pd.concat([pd.read_parquet(p) for p in glob.glob(str(paths.ARTIFACTS / "news" / "text" / "*.parquet"))],
                      ignore_index=True).drop_duplicates("article_key")
    no_text = pairs[~pairs.article_key.isin(set(texts.article_key))].groupby("sample_id").size()
    pairs = pairs.merge(texts, on="article_key", how="inner").merge(fr[["sample_id", "decision_utc"]], on="sample_id")
    pairs["available_utc"] = pd.to_datetime(pairs.available_utc, utc=True)
    log(f"news pairs with text: {len(pairs):,} (articles {pairs.article_key.nunique():,}); pairs without text in "
        f"{len(no_text):,} samples")
    rep, raw_n = dedup_and_cluster(pairs)
    fit_docs = rep.loc[rep.available_utc < FIT_BEFORE, "doc"]
    fit_docs = fit_docs.sample(min(len(fit_docs), 150_000), random_state=7)
    tf = TfidfSvd(64).fit(fit_docs)
    log(f"news TF-IDF fitted on {len(fit_docs):,} pre-2016 articles; SVD explained {tf.explained:.3f}")
    feats, rep = news_features(rep, raw_n, tf)
    out = fr[["sample_id"]].merge(feats, left_on="sample_id", right_index=True, how="left").set_index("sample_id")
    out["news_present"] = out.news_present.fillna(0.0)
    out["news_n_selected_no_text"] = no_text.reindex(out.index).fillna(0.0).values
    with open(OUT / "news_tfidf_svd.pkl", "wb") as f:
        pickle.dump(tf, f)
    return out, rep[["sample_id", "article_key", "available_utc", "headline", "n_cluster", "news_lm_net_tone"]], tf


def build_sec(fr: pd.DataFrame, log=print):
    ev = S.load_events(dev_only=True)
    structured = S.sec_features(fr[["sample_id", "ticker", "decision_utc"]], ev)
    latest = S.latest_event_ids(fr[["sample_id", "ticker", "decision_utc"]], ev)
    need = set(latest.drop(columns="sample_id").stack().dropna())
    pre = ev[ev.available_utc < FIT_BEFORE]
    fit_ids = set(pre.event_id.sample(min(60_000, len(pre)), random_state=7))
    path = next((paths.SEC_FULL500_RUN / "S09").glob("*/event_text.parquet"))
    txt = pq.read_table(path, filters=[("event_id", "in", list(need | fit_ids))]).to_pandas().set_index("event_id").text
    txt = txt.fillna("").str.slice(0, 3000)
    log(f"SEC texts loaded: {len(txt):,} (needed {len(need):,}, fit {len(fit_ids):,})")
    tf = TfidfSvd(32).fit(txt.reindex(list(fit_ids)).dropna())
    log(f"SEC TF-IDF SVD explained {tf.explained:.3f}")
    with open(OUT / "sec_tfidf_svd.pkl", "wb") as f:
        pickle.dump(tf, f)
    textf = sec_text_features(latest, txt, tf)
    out = pd.concat([structured, textf], axis=1).reindex(fr.sample_id)
    for kind in ("8k", "periodic"):
        out[f"sec_{kind}_present"] = out[f"sec_{kind}_present"].fillna(0.0)
    out["sec_covered"] = out.sec_covered.fillna(0.0)
    return out, tf


def main() -> int:
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "decision_utc", "block"])
    OUT.mkdir(parents=True, exist_ok=True)
    news, reps, ntf = build_news(fr)
    news.to_parquet(OUT / "news_features.parquet")
    reps.to_parquet(OUT / "news_representatives.parquet", index=False)
    print(f"news features: {news.news_present.mean():.3f} of samples have news ({time.time() - t0:.0f} s)", flush=True)
    sec, stf = build_sec(fr)
    sec.to_parquet(OUT / "sec_features.parquet")
    summ = {"news_present_share": float(news.news_present.mean()), "news_tfidf_explained": ntf.explained,
            "samples_with_untexted_articles": int((news.news_n_selected_no_text > 0).sum()),
            "sec_covered_share": float(sec.sec_covered.mean()), "sec_8k_present_share": float(sec.sec_8k_present.mean()),
            "sec_periodic_present_share": float(sec.sec_periodic_present.mean()), "sec_tfidf_explained": stf.explained,
            "elapsed_s": round(time.time() - t0, 1)}
    (paths.REPO_ROOT / "data_manifests" / "text_features_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
