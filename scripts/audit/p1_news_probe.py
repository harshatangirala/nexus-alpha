"""Phase 1: probe every subset of the news corpus without downloading article bodies.

For each subset, opens the shards remotely (HTTP range reads through HfFileSystem), records the parquet row counts
from footers, and reads a sample of the `date` and `extra_fields` columns from the first row group of the first shard
to learn which subsets carry ticker tags, their timestamp precision and timezone hints. Article text is never read.

Output: data_manifests/news_subset_probe.json
Run: .venv/Scripts/python -I scripts/audit/p1_news_probe.py
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = Path(__file__).resolve().parents[2]
DATASET = "Brianferrell787/financial-news-multisource"
SAMPLE_ROWS = 400


def main() -> int:
    info = json.loads((REPO / "data_manifests" / "news_dataset_info.json").read_text(encoding="utf-8"))
    shards = collections.defaultdict(list)
    for f in info["files"]:
        p = f["path"]
        if p.startswith("data/") and p.endswith(".parquet"):
            shards[p.split("/")[1]].append(p)
    fs = HfFileSystem()
    out = {"dataset": DATASET, "revision": info["sha"], "probed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "subsets": {}}
    for subset, paths in sorted(shards.items()):
        rec = {"n_shards": len(paths), "rows_per_shard": {}, "errors": []}
        for p in sorted(paths):
            try:
                with fs.open(f"datasets/{DATASET}@{info['sha']}/{p}", "rb") as fh:
                    md = pq.ParquetFile(fh).metadata
                    rec["rows_per_shard"][p] = md.num_rows
            except Exception as e:
                rec["errors"].append(f"{p}: {type(e).__name__}: {str(e)[:300]}")
                break
        rec["n_rows"] = sum(rec["rows_per_shard"].values())
        first = sorted(paths)[0]
        try:
            with fs.open(f"datasets/{DATASET}@{info['sha']}/{first}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                rec["columns"] = pf.schema_arrow.names
                tbl = pf.read_row_group(0, columns=["date", "extra_fields"]).slice(0, SAMPLE_ROWS)
            dates = tbl.column("date").to_pylist()
            extras = tbl.column("extra_fields").to_pylist()
            keys, prec, tz, src, ds = (collections.Counter() for _ in range(5))
            with_stocks, n_stocks = 0, []
            hhmm = collections.Counter()
            for d, x in zip(dates, extras):
                try:
                    e = json.loads(x) if x else {}
                except Exception:
                    e = {"_unparseable": True}
                keys.update(e.keys())
                prec[e.get("time_precision")] += 1
                tz[e.get("tz_hint")] += 1
                src[str(e.get("source"))[:40]] += 1
                ds[e.get("dataset")] += 1
                st = e.get("stocks")
                if st:
                    with_stocks += 1
                    n_stocks.append(len(st) if isinstance(st, list) else 1)
                if isinstance(d, str) and len(d) >= 16:
                    hhmm[d[11:16]] += 1
            rec["sample"] = {
                "n": len(dates), "date_min": min(dates) if dates else None, "date_max": max(dates) if dates else None,
                "extra_keys": dict(keys.most_common(30)), "time_precision": dict(prec), "tz_hint": dict(tz),
                "source_top": dict(src.most_common(8)), "dataset_field": dict(ds.most_common(5)),
                "share_with_stocks": round(with_stocks / max(len(dates), 1), 3),
                "mean_stocks_when_present": round(sum(n_stocks) / len(n_stocks), 2) if n_stocks else None,
                "top_hh_mm_utc": dict(hhmm.most_common(6)),
            }
        except Exception as e:
            rec["errors"].append(f"sample {first}: {type(e).__name__}: {str(e)[:300]}")
        out["subsets"][subset] = rec
        s = rec.get("sample", {})
        print(f"{subset:<30} shards={len(paths):>3} rows={rec['n_rows']:>11,} stocks_share={s.get('share_with_stocks')} "
              f"prec={s.get('time_precision')} err={len(rec['errors'])}", flush=True)
    out["total_rows"] = sum(v["n_rows"] for v in out["subsets"].values())
    (REPO / "data_manifests" / "news_subset_probe.json").write_text(json.dumps(out, indent=1, default=str),
                                                                    encoding="utf-8")
    print("total rows:", f"{out['total_rows']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
