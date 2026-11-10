"""Business-address state or country of the forward-test universe (dashboard globe only), from SEC's submissions endpoint.

    python scripts/data/fetch_addresses.py

Same declared User-Agent and <= 5 requests per second as fetch_sic.py. Writes artifacts/fundamentals/addresses.parquet
(ticker, cik, city, state_or_country, state_or_country_description). The address is the filer's current one, so it is
display context only: it is never a model input.
"""
from __future__ import annotations

import json
import sys
import time
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
    t0 = time.time()
    tick = set(pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=["ticker"]).ticker)
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet")
    cm = cm[cm.ticker.isin(tick)].drop_duplicates("ticker")
    rows, failed = [], []
    for t, c in cm[["ticker", "cik"]].itertuples(index=False):
        SF._RL.wait()
        req = urllib.request.Request(URL.format(cik=int(c)), headers={"User-Agent": SF._ua()})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                js = json.loads(r.read())
            b = (js.get("addresses") or {}).get("business") or {}
            rows.append({"ticker": t, "cik": int(c), "city": b.get("city"), "state_or_country": b.get("stateOrCountry"),
                         "state_or_country_description": b.get("stateOrCountryDescription")})
        except Exception as e:      # noqa: BLE001  (reported, not fatal: the globe shows what it has)
            failed.append((t, f"{type(e).__name__}: {e}"))
    out = pd.DataFrame(rows)
    out.to_parquet(paths.artifact("fundamentals", "addresses.parquet"), index=False)
    print(f"{len(out)} addresses, {len(failed)} failed, {time.time() - t0:.0f}s; missing tickers: "
          f"{sorted(tick - set(cm.ticker))[:10]}")
    if failed:
        print("failed:", failed[:10])


if __name__ == "__main__":
    main()
