"""Fetch article text for the selected (sample, article) pairs only, streaming the `text` column shard by shard.

Keeps, per needed article: headline (first paragraph, <= 300 chars), lead (first 1,500 chars), full length and a
normalised-text hash for exact de-duplication. Article text is licensed for non-commercial research only and is never
committed (artifacts/ is git-ignored) or shown in full in any public output.

Input: artifacts/news/s500_selected_pairs.parquet. Output: artifacts/news/text/<subset>__<shard>.parquet
Run: .venv/Scripts/python -I scripts/data/fetch_news_text.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATASET = "Brianferrell787/financial-news-multisource"


def fetch_shard(args, attempts: int = 4):
    """Retry a shard on network errors; a shard that still fails is reported, not fatal (re-run to resume)."""
    last = None
    for i in range(attempts):
        try:
            return _fetch_shard(args)
        except Exception as e:                       # network timeouts are expected on a home connection
            last = e
            time.sleep(5 * (i + 1))
    return args[1], args[2], f"FAILED after {attempts} attempts: {type(last).__name__}", 0


def _fetch_shard(args):
    rev, subset, shard, rows_by_rg = args
    from huggingface_hub import HfFileSystem
    from nexus_alpha.data import news as N
    out_p = REPO / "artifacts" / "news" / "text" / f"{subset}__{shard:03d}.parquet"
    if out_p.exists():
        return subset, shard, "cached", 0
    fs = HfFileSystem()
    path = f"datasets/{DATASET}@{rev}/data/{subset}/{subset}.{shard:03d}.parquet"
    rec = []
    with fs.open(path, "rb", block_size=16 << 20) as fh:
        pf = pq.ParquetFile(fh)
        for rg, wanted in rows_by_rg.items():
            wanted = set(wanted)
            offset = 0
            for batch in pf.iter_batches(batch_size=20000, row_groups=[rg], columns=["text"]):
                col = batch.column(0).to_pylist()
                for i, txt in enumerate(col):
                    r = offset + i
                    if r in wanted:
                        txt = txt or ""
                        rec.append((f"{subset}/{shard}/{rg}/{r}", N.headline(txt)[:300], txt[:1500], len(txt),
                                    N.text_hash(txt)))
                offset += len(col)
    df = pd.DataFrame(rec, columns=["article_key", "headline", "lead", "text_len", "text_hash"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    return subset, shard, "fetched", len(df)


def main() -> int:
    t0 = time.time()
    info = json.loads((REPO / "data_manifests" / "news_dataset_info.json").read_text(encoding="utf-8"))
    sel = pd.read_parquet(REPO / "artifacts" / "news" / "s500_selected_pairs.parquet",
                          columns=["article_key", "subset", "shard", "row_group", "row_in_group"]).drop_duplicates("article_key")
    jobs = []
    for (subset, shard), g in sel.groupby(["subset", "shard"]):
        jobs.append((info["sha"], subset, int(shard), g.groupby("row_group").row_in_group.apply(list).to_dict()))
    jobs.sort(key=lambda j: -sum(len(v) for v in j[3].values()))
    with cf.ProcessPoolExecutor(max_workers=2) as ex:
        for subset, shard, status, n in ex.map(fetch_shard, jobs):
            print(f"{subset}.{shard:03d}: {status} {n} rows, {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
