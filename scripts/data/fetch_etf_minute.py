"""Extract index and macro-proxy ETF minute bars from the same 1-minute source TriFrame used.

Source: Hugging Face dataset `mito0o852/OHLCV-1m` (Finnhub-derived, monthly files, all tickers mixed; row groups are not
sorted by ticker, so every month file must be downloaded whole). Each month is downloaded, filtered to the tickers
below, written to `artifacts/raw/etf_1m/{YYYY-MM}.parquet`, and the download is deleted (the C: drive is small).

Seal: by default only months before 2023-01 are fetched. Months on/after the seal are fetched only with
`--include-sealed`, which is reserved for the owner-authorised sealed evaluation and for building live features.
Resumable: months already extracted are skipped. Data stays in `artifacts/` (git-ignored; the source declares no licence).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from nexus_alpha import paths  # noqa: E402
from nexus_alpha.seal import SEAL_DATE  # noqa: E402

DATASET = "mito0o852/OHLCV-1m"
# Broad market, size, style, the eleven SPDR sectors, rates, credit, dollar, commodities, international.
TICKERS = ["SPY", "IVV", "VOO", "QQQ", "DIA", "IWM", "MDY",
           "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
           "TLT", "IEF", "SHY", "TIP", "HYG", "LQD", "UUP", "GLD", "SLV", "USO", "DBC", "EEM", "EFA"]


def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2001-01")
    ap.add_argument("--end", default="2022-12")
    ap.add_argument("--include-sealed", action="store_true")
    a = ap.parse_args()
    last_dev = (SEAL_DATE - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    end = a.end if a.include_sealed else min(a.end, last_dev)
    out_dir = paths.ARTIFACTS / "raw" / "etf_1m"
    tmp = paths.ARTIFACTS / "raw" / "hf_ohlcv1m"
    out_dir.mkdir(parents=True, exist_ok=True)
    want = pa.array(TICKERS)
    todo = [m for m in months(a.start, end) if not (out_dir / f"{m}.parquet").exists()]
    print(f"{len(todo)} months to fetch ({a.start}..{end}); tickers={len(TICKERS)}", flush=True)
    for m in todo:
        t0 = time.time()
        fn = f"data/ohlcv_{m}.parquet"
        for attempt in range(4):
            try:
                p = hf_hub_download(DATASET, fn, repo_type="dataset", local_dir=str(tmp))
                break
            except Exception as e:  # network hiccups: back off and retry
                print(f"{m}: download failed ({e!r}); retry {attempt + 1}", flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            print(f"{m}: giving up", flush=True)
            continue
        t = pq.read_table(p)
        t = t.filter(pc.is_in(t["ticker"], value_set=want))
        pq.write_table(t, out_dir / f"{m}.parquet.tmp", compression="zstd")
        os.replace(out_dir / f"{m}.parquet.tmp", out_dir / f"{m}.parquet")
        os.remove(p)
        n = pc.unique(t["ticker"]).to_pylist()
        print(f"{m}: {t.num_rows:,} rows, {len(n)} tickers, {time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
