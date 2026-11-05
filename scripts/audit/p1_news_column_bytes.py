"""Phase 1: compressed bytes per column for every news shard, from parquet footers only (download budgeting).

Output: data_manifests/news_column_bytes.csv
Run: .venv/Scripts/python -I scripts/audit/p1_news_column_bytes.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = Path(__file__).resolve().parents[2]
DATASET = "Brianferrell787/financial-news-multisource"


def footer(fs, rev, path):
    with fs.open(f"datasets/{DATASET}@{rev}/{path}", "rb") as fh:
        md = pq.ParquetFile(fh).metadata
    sizes = {}
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for j in range(rg.num_columns):
            c = rg.column(j)
            sizes[c.path_in_schema] = sizes.get(c.path_in_schema, 0) + c.total_compressed_size
    return {"path": path, "subset": path.split("/")[1], "rows": md.num_rows, "row_groups": md.num_row_groups,
            **{f"bytes_{k}": v for k, v in sizes.items()}}


def main() -> int:
    info = json.loads((REPO / "data_manifests" / "news_dataset_info.json").read_text(encoding="utf-8"))
    rev = info["sha"]
    shards = [f["path"] for f in info["files"] if f["path"].startswith("data/") and f["path"].endswith(".parquet")]
    fs = HfFileSystem()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda p: footer(fs, rev, p), shards))
    df = pd.DataFrame(rows).sort_values("path")
    df.to_csv(REPO / "data_manifests" / "news_column_bytes.csv", index=False)
    g = df.groupby("subset")[["rows", "bytes_date", "bytes_text", "bytes_extra_fields"]].sum()
    g = (g.assign(GB_date=g.bytes_date / 1e9, GB_text=g.bytes_text / 1e9, GB_extra=g.bytes_extra_fields / 1e9)
         .drop(columns=["bytes_date", "bytes_text", "bytes_extra_fields"]).sort_values("GB_text", ascending=False))
    print(g.round(3).to_string())
    print("TOTAL GB date/text/extra:", round(df.bytes_date.sum() / 1e9, 2), round(df.bytes_text.sum() / 1e9, 2),
          round(df.bytes_extra_fields.sum() / 1e9, 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
