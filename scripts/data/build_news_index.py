"""Build the news index (no article text) and the corpus date-coverage audit.

* Ticker-tagged subsets: reads `date` and `extra_fields` row group by row group, keeps articles tagged with at least one
  universe ticker, and records each article's exact location (shard, row group, row) so text can be fetched later
  for selected rows only. One output row per (article, universe ticker).
* All subsets: reads the `date` column only, for the true min/max publication date and per-year counts.

Outputs: artifacts/news/news_index.parquet, data_manifests/news_date_coverage.json
Run: .venv/Scripts/python -I scripts/data/build_news_index.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATASET = "Brianferrell787/financial-news-multisource"


def scan_shard(args):
    rev, path, tagged, universe = args
    from huggingface_hub import HfFileSystem
    from nexus_alpha.data import news as N
    fs = HfFileSystem()
    subset = path.split("/")[1]
    shard = int(path.rsplit(".", 2)[-2])
    stats = {"subset": subset, "path": path, "rows": 0, "years": {}, "date_min": None, "date_max": None,
             "unparseable_dates": 0}
    parts = []
    with fs.open(f"datasets/{DATASET}@{rev}/{path}", "rb", block_size=8 << 20) as fh:
        pf = pq.ParquetFile(fh)
        for g in range(pf.metadata.num_row_groups):
            cols = ["date", "extra_fields"] if tagged else ["date"]
            t = pf.read_row_group(g, columns=cols)
            dates = pd.to_datetime(pd.Series(t.column("date").to_pylist()), utc=True, errors="coerce")
            stats["rows"] += len(dates)
            stats["unparseable_dates"] += int(dates.isna().sum())
            ok = dates.dropna()
            if len(ok):
                mn, mx = ok.min(), ok.max()
                stats["date_min"] = mn if stats["date_min"] is None else min(stats["date_min"], mn)
                stats["date_max"] = mx if stats["date_max"] is None else max(stats["date_max"], mx)
                for y, c in ok.dt.year.value_counts().items():
                    stats["years"][int(y)] = stats["years"].get(int(y), 0) + int(c)
            if not tagged:
                continue
            extras = t.column("extra_fields").to_pylist()
            rows = []
            for i, x in enumerate(extras):
                e = N.parse_extra(x)
                tk = N.tickers_of(e)
                hit = [k for k in tk if k in universe]
                if not hit:
                    continue
                for k in hit:
                    rows.append((g, i, k, len(tk), e.get("time_precision"), e.get("date_trading"), e.get("url"),
                                 str(e.get("source") or e.get("publisher") or "")[:80], e.get("text_type"),
                                 e.get("dataset"), e.get("tz_hint")))
            if rows:
                p = pd.DataFrame(rows, columns=["row_group", "row_in_group", "ticker", "n_tickers_tagged",
                                                "time_precision", "date_trading", "url", "publisher", "text_type",
                                                "dataset_field", "tz_hint"])
                p["date_utc"] = dates.iloc[p.row_in_group.values].values
                parts.append(p)
    out = pd.concat(parts, ignore_index=True) if parts else None
    if out is not None:
        out.insert(0, "shard", shard)
        out.insert(0, "subset", subset)
    stats["date_min"] = str(stats["date_min"]) if stats["date_min"] is not None else None
    stats["date_max"] = str(stats["date_max"]) if stats["date_max"] is not None else None
    return stats, out


def main() -> int:
    from nexus_alpha.data import bars as B
    from nexus_alpha.data import news as N
    t0 = time.time()
    info = json.loads((REPO / "data_manifests" / "news_dataset_info.json").read_text(encoding="utf-8"))
    rev = info["sha"]
    universe = set(B.available_tickers())
    shards = sorted(f["path"] for f in info["files"] if f["path"].startswith("data/") and f["path"].endswith(".parquet"))
    jobs = [(rev, p, p.split("/")[1] in N.TICKER_TAGGED_SUBSETS, universe) for p in shards]
    # big tagged shards first so the pool stays busy
    sizes = {f["path"]: f["bytes"] or 0 for f in info["files"]}
    jobs.sort(key=lambda j: (not j[2], -sizes.get(j[1], 0)))
    stats, parts = [], []
    with cf.ProcessPoolExecutor(max_workers=6) as ex:
        for k, (s, p) in enumerate(ex.map(scan_shard, jobs)):
            stats.append(s)
            if p is not None:
                parts.append(p)
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(jobs)} shards, {time.time() - t0:.0f} s", flush=True)
    idx = pd.concat(parts, ignore_index=True)
    pub, avail, eff = N.availability(idx.date_utc, idx.time_precision)
    idx["published_utc"] = pub
    idx["available_utc"] = avail
    idx["effective_precision"] = eff.values
    idx["article_key"] = (idx.subset + "/" + idx.shard.astype(str) + "/" + idx.row_group.astype(str) + "/"
                          + idx.row_in_group.astype(str))
    out_dir = REPO / "artifacts" / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(out_dir / "news_index.parquet", index=False)

    cov = {"dataset": DATASET, "revision": rev, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "subsets": {}, "elapsed_s": None}
    st = pd.DataFrame(stats)
    for sub, g in st.groupby("subset"):
        yrs = {}
        for d in g.years:
            for y, c in d.items():
                yrs[y] = yrs.get(y, 0) + c
        cov["subsets"][sub] = {"rows": int(g.rows.sum()), "date_min": min(x for x in g.date_min if x),
                               "date_max": max(x for x in g.date_max if x),
                               "unparseable_dates": int(g.unparseable_dates.sum()),
                               "rows_by_year": {int(k): int(v) for k, v in sorted(yrs.items())}}
    allmax = max(v["date_max"] for v in cov["subsets"].values())
    allmin = min(v["date_min"] for v in cov["subsets"].values())
    cov["corpus_date_min"], cov["corpus_date_max"] = allmin, allmax
    a = idx.drop_duplicates("article_key")
    cov["index"] = {
        "article_ticker_rows": int(len(idx)), "unique_articles": int(len(a)),
        "tickers_with_any_article": int(idx.ticker.nunique()),
        "by_subset": a.subset.value_counts().to_dict(),
        "effective_precision": a.effective_precision.value_counts().to_dict(),
        "published_min": str(a.published_utc.min()), "published_max": str(a.published_utc.max()),
        "unique_articles_by_year": {int(k): int(v) for k, v in a.published_utc.dt.year.value_counts().sort_index().items()},
        "n_tickers_tagged_quantiles": a.n_tickers_tagged.quantile([.5, .9, .99]).to_dict(),
        "url_present_share": float(a.url.notna().mean()),
    }
    cov["elapsed_s"] = round(time.time() - t0, 1)
    (REPO / "data_manifests" / "news_date_coverage.json").write_text(json.dumps(cov, indent=1, default=str),
                                                                     encoding="utf-8")
    print(json.dumps({k: v for k, v in cov.items() if k != "subsets"}, indent=1, default=str))
    for sub, v in sorted(cov["subsets"].items()):
        print(f"{sub:<30} rows={v['rows']:>11,} {v['date_min'][:10]} .. {v['date_max'][:10]} bad={v['unparseable_dates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
