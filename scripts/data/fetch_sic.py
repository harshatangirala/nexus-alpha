"""SIC codes for tickers the SEC study's S02 universe lacks (E23), from SEC's submissions endpoint.

    python scripts/data/fetch_sic.py

Same declared User-Agent and <= 5 requests per second as the companyfacts download. Writes
artifacts/fundamentals/sic_sec.parquet (ticker, cik, sic). SIC is the filer's current code, as in S02: industry
membership is not point in time (disclosed in docs/V3_RESEARCH.md, E23).
"""
from __future__ import annotations

import glob
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.fundamentals import sec_facts as SF  # noqa: E402

URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"


def main() -> None:
    u = pd.read_parquet(glob.glob(str(paths.SEC_FULL500_RUN / "S02" / "*" / "universe.parquet"))[0])[["ticker", "sic"]]
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet")
    need = cm[cm.ticker.isin(u[pd.to_numeric(u.sic, errors="coerce").isna()].ticker)]
    rows = []
    for t, c in need.itertuples(index=False):
        SF._RL.wait()
        req = urllib.request.Request(URL.format(cik=int(c)), headers={"User-Agent": SF._ua()})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                js = json.loads(r.read())
            rows.append((t, int(c), str(js.get("sic") or "") or None))
        except Exception as e:
            rows.append((t, int(c), None))
            print(t, "failed:", type(e).__name__)
    out = pd.DataFrame(rows, columns=["ticker", "cik", "sic"])
    out.to_parquet(paths.artifact("fundamentals", "sic_sec.parquet"), index=False)
    print(f"{len(out)} tickers without S02 SIC; filled {out.sic.notna().sum()}")


if __name__ == "__main__":
    main()
