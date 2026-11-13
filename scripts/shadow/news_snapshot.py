"""Point-in-time archive of live company headlines (audit 11, opportunity O4; not part of any model).

The forward test has no live news: the frozen models run with the news branch missing, so the text branch is never
tested forward (finding A-99). This script starts the evidence that a future, pre-registered forward test of the text
branch would need: for each universe stock, Yahoo Finance's RSS headlines with their publication time and the time
this machine saw them (`fetched_utc`). A headline can only ever be used for decisions after `fetched_utc`, whatever its
stated publication time, so the archive is point in time by construction.

* Output: `artifacts/live_news/headlines_<UTC date>.parquet` (git-ignored: article text is never published), appended
  and de-duplicated by (ticker, link, headline) within a day; a JSON summary next to it.
* Polite: one request at a time with a pause; a failed ticker is recorded and skipped; a run never raises.
* Standalone: the nightly task does not call it. The owner may add it as a last step of `scripts/shadow/run_daily.cmd`
  (after the push), where a failure cannot affect any forecast.

    python scripts/shadow/news_snapshot.py                 # whole universe (about 500 requests, ~5 minutes)
    python scripts/shadow/news_snapshot.py --tickers AAPL MSFT --pause 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO / "src"))
OUT_DIR = REPO / "artifacts" / "live_news"
COLS = ["fetched_utc", "ticker", "published_utc", "headline", "text", "link", "source"]


def universe() -> list[str]:
    """The forward test's universe, read exactly as shadow_forecast.py reads it."""
    from nexus_alpha.cv import store as ST
    return sorted(ST.sample_index().ticker.unique())


def snapshot(tickers: list[str], fetch, pause: float = 0.6, now=None) -> tuple[pd.DataFrame, dict]:
    """fetch(ticker) -> list of dicts (headline, text, published_utc, link, source). Returns new rows and a summary."""
    rows, failed = [], {}
    for i, t in enumerate(tickers):
        stamp = (now or pd.Timestamp.now(tz="UTC")).isoformat(timespec="seconds")
        try:
            for a in fetch(t) or []:
                rows.append({"fetched_utc": stamp, "ticker": t, "published_utc": a.get("published_utc", ""),
                             "headline": a.get("headline", ""), "text": a.get("text", ""), "link": a.get("link", ""),
                             "source": a.get("source", "")})
        except Exception as e:  # network, rate limit, parse error: record and move on
            failed[t] = repr(e)[:160]
        if pause and i < len(tickers) - 1:
            time.sleep(pause)
    df = pd.DataFrame(rows, columns=COLS)
    return df, {"tickers": len(tickers), "with_headlines": int(df.ticker.nunique()) if len(df) else 0,
                "headlines": int(len(df)), "failed": len(failed), "failed_examples": dict(list(failed.items())[:5])}


def append(df: pd.DataFrame, out_dir: Path = OUT_DIR, day: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    day = day or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    f = out_dir / f"headlines_{day}.parquet"
    if f.exists():
        df = pd.concat([pd.read_parquet(f), df], ignore_index=True)
    # keep the FIRST sighting of each headline (the earliest time it could have been used)
    df = df.sort_values("fetched_utc").drop_duplicates(["ticker", "link", "headline"], keep="first")
    df.to_parquet(f, index=False)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--pause", type=float, default=0.6)
    a = ap.parse_args()
    from nexus_alpha.trade_tools import yahoo_headlines
    tickers = a.tickers or universe()
    if not tickers:
        print("news snapshot: no tickers (universe unavailable); nothing fetched", flush=True)
        return
    df, summ = snapshot(tickers, lambda t: yahoo_headlines(t, limit=20), pause=a.pause)
    f = append(df)
    f.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(f"news snapshot: {summ['headlines']} headlines for {summ['with_headlines']}/{summ['tickers']} stocks, "
          f"{summ['failed']} failed -> {f.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
