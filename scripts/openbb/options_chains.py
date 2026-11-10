"""Option chains from CBOE's delayed quotes through OpenBB (runs in artifacts/envs/openbb, never in the project env).

    <openbb env python> scripts/openbb/options_chains.py --tickers AAPL MSFT --out chains.parquet [--concurrency 8]
    <openbb env python> scripts/openbb/options_chains.py --tickers-file tickers.txt --out chains.parquet

Uses OpenBB's CBOE provider (`openbb_cboe`, `CboeOptionsChainsFetcher`: its own extract and transform steps) directly,
with asyncio concurrency. The `obb.cboe.options.chains` command layer gives the same rows, but took about 40 s per
stock under threads; the fetcher takes about 2 s.

Writes one parquet with the contract fields the E26 definitions use (nexus_alpha.options), plus <out>.errors.json.
The project environment only reads that file. Data: CBOE delayed quotes; kept private (CBOE's terms), never committed.
This interpreter's default certificate store is outdated, so SSL_CERT_FILE is pointed at certifi's current bundle
(verification stays on).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
RETRIES = 5
KEEP = ["underlying_symbol", "underlying_price", "expiration", "dte", "strike", "option_type", "bid", "ask",
        "implied_volatility", "open_interest", "volume"]


def fallback_transform(raw: dict):
    """Minimal parse of the raw CBOE payload OpenBB downloaded, used only when OpenBB's own transform raises (seen for
    TSLA on 2026-11-10: a missing last-trade time breaks its date parsing). Same fields, same contract-symbol rule."""
    import re
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pandas as pd
    d = raw.get("data", {})
    rows = []
    for o in d.get("options", []):
        m = re.match(r"^(\D*)(\d{6})([CP])(\d{8})$", str(o.get("option", "")))
        if not m:
            continue
        rows.append({"underlying_symbol": m.group(1), "expiration": datetime.strptime(m.group(2), "%y%m%d").date(),
                     "option_type": "call" if m.group(3) == "C" else "put", "strike": int(m.group(4)) / 1000,
                     "bid": o.get("bid"), "ask": o.get("ask"), "implied_volatility": o.get("iv"),
                     "open_interest": o.get("open_interest"), "volume": o.get("volume")})
    df = pd.DataFrame(rows)
    if not len(df):
        raise ValueError("no parsable contracts in the raw payload")
    today = datetime.now(tz=ZoneInfo("America/New_York")).date()
    df["dte"] = [(e - today).days for e in df.expiration]
    df["expiration"] = df.expiration.astype(str)
    df["underlying_price"] = d.get("current_price")
    return df


async def run(tickers: list[str], concurrency: int, timeout: float):
    import pandas as pd
    from openbb_cboe.models.options_chains import CboeOptionsChainsFetcher as F
    sem = asyncio.Semaphore(concurrency)
    parts, errors, fallbacks = [], {}, []

    async def fetch_raw(q):
        """CBOE's CDN sometimes answers with a problem document (no "data") under load: retry with backoff. A symbol
        missing from CBOE's options directory is permanent and is not retried."""
        import random
        last = None
        for attempt in range(RETRIES):
            try:
                async with sem:
                    raw = await asyncio.wait_for(F.aextract_data(q, None), timeout)
                if isinstance(raw, dict) and "data" in raw:
                    return raw
                last = ValueError(f"problem document from CBOE: {sorted(raw)[:6] if isinstance(raw, dict) else type(raw)}")
            except Exception as e:  # noqa: BLE001
                if "not found in the Cboe options directory" in str(e):
                    raise
                last = e
            await asyncio.sleep(2 ** attempt * 2 + random.random())
        raise last

    async def one(sym: str):
        try:
            q = F.transform_query({"symbol": sym})
            raw = await fetch_raw(q)
            import copy
            try:
                df = pd.DataFrame(F.transform_data(q, copy.deepcopy(raw)).result.model_dump())
            except Exception:  # noqa: BLE001  (OpenBB transform bug on some payloads; documented fallback)
                df = fallback_transform(raw)
                fallbacks.append(sym)
            df = df[[c for c in KEEP if c in df.columns]]
            df.insert(0, "ticker", sym)
            parts.append(df)
        except Exception as e:  # noqa: BLE001  (a stock without listed options is common; recorded)
            msg = str(e).strip().splitlines()[-1][:160] if str(e).strip() else ""
            errors[sym] = f"{type(e).__name__}: {msg}"

    await asyncio.gather(*(one(t) for t in tickers))
    return parts, errors, fallbacks


def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=[])
    ap.add_argument("--tickers-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=1)   # CBOE's CDN throttles bursts (2026-11-10)
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    tickers = list(a.tickers)
    if a.tickers_file:
        tickers += [t.strip() for t in open(a.tickers_file, encoding="utf-8") if t.strip()]
    t0 = time.time()
    parts, errors, fallbacks = asyncio.run(run(tickers, a.concurrency, a.timeout))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ticker"] + KEEP)
    out["expiration"] = out["expiration"].astype(str)
    out.to_parquet(a.out, index=False)
    with open(a.out + ".errors.json", "w", encoding="utf-8") as fh:
        json.dump({"requested": len(tickers), "ok": len(parts), "errors": errors,
                   "fallback_parser": sorted(fallbacks), "seconds": round(time.time() - t0, 1)}, fh, indent=1)
    print(f"chains: {len(parts)} of {len(tickers)} tickers, {len(out)} contracts, {time.time() - t0:.0f}s", flush=True)
    return 0 if parts or not tickers else 1


if __name__ == "__main__":
    sys.exit(main())
