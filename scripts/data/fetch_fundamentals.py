"""Download SEC XBRL company facts for every ticker of the source universe (resumable; owner's declared User-Agent).

    python scripts/data/fetch_fundamentals.py
CIKs come from the SEC study's S02 universe (ticker -> CIK, read-only source). Writes artifacts/fundamentals/raw/ and
artifacts/fundamentals/cik_map.parquet.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.fundamentals import sec_facts as SF  # noqa: E402


def cik_map() -> pd.DataFrame:
    u = pd.read_parquet(glob.glob(str(paths.SEC_FULL500_RUN / "S02" / "*" / "universe.parquet"))[0])
    m = u[["ticker", "cik"]].dropna().astype({"cik": int}).drop_duplicates("ticker")
    out = paths.artifact("fundamentals", "cik_map.parquet")
    m.to_parquet(out, index=False)
    return m


def main() -> None:
    t0 = time.time()
    m = cik_map()
    ciks = sorted(m.cik.unique())
    done = sum(SF.raw_path(c).exists() for c in ciks)
    print(f"{len(m)} tickers, {len(ciks)} CIKs, {done} cached", flush=True)
    ok, none, fail = 0, 0, []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(SF.fetch, c): c for c in ciks}
        for i, f in enumerate(cf.as_completed(futs)):
            try:
                r = f.result()
                ok += r is not None
                none += r is None
            except Exception as e:
                fail.append((futs[f], repr(e)[:80]))
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(ciks)} ({time.time() - t0:.0f}s) ok {ok} none {none} failed {len(fail)}", flush=True)
    print(f"done: ok {ok}, no XBRL {none}, failed {len(fail)} {fail[:5]} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
