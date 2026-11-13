"""E29b news-on pass (D-058): the frozen v1.1 forecast WITH the archived live company headlines, for the stocks that
have at least one usable headline. Logged under shadow/v1.1_news with its own hash chain and pushed before the
decision time, exactly like the other forecasters (docs/E29_PREREGISTRATION.md).

    python scripts/shadow/news_on_forecast.py [--dry-run] [--limit N] [--tickers AAPL MSFT]

Point in time:
* headlines come from artifacts/live_news (news_snapshot.py); a headline is eligible only if this machine had fetched it
  (fetched_utc) at or before the forecast's decision time, and the run time is before the decision time;
* each headline's own publication time then goes through the model's usual availability rule (`_articles_frame`:
  inside the 5-day window before the decision time, at most 7 articles, newest first);
* bars come from tonight's shared cache (the same bars the no-news v1.1 run used), so the only difference between the
  two logs is the news.

Nothing is fitted; the frozen bundle is hash-verified first. A stock without an eligible headline is not logged here
(E29b's endpoint is defined on stocks with news). Rows after the decision time are flagged late and never scored.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "shadow"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha import shadow as SH  # noqa: E402

VERSION, BASE = "v1.1_news", "v1.1"
ARCHIVE = paths.ARTIFACTS / "live_news"
MAX_ARTICLES, LOOKBACK = 7, pd.Timedelta(days=5)
TICKER_TIMEOUT_S = 180


def archive(decision_utc: pd.Timestamp, root: Path = ARCHIVE) -> pd.DataFrame:
    """Every archived headline this machine had fetched at or before `decision_utc`, de-duplicated (first sighting kept)."""
    files = sorted(root.glob("headlines_*.parquet")) if root.exists() else []
    if not files:
        return pd.DataFrame(columns=["ticker", "fetched_utc", "published_utc", "headline", "text", "link", "source"])
    a = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    a["fetched_utc"] = pd.to_datetime(a.fetched_utc, utc=True, errors="coerce")
    a["published_utc"] = pd.to_datetime(a.published_utc, utc=True, errors="coerce")
    a = a[a.fetched_utc.notna() & (a.fetched_utc <= decision_utc)]
    a["published_utc"] = a.published_utc.fillna(a.fetched_utc)      # no stated time: known from when it was seen
    a = a.sort_values("fetched_utc").drop_duplicates(["ticker", "link", "headline"], keep="first")
    return a.reset_index(drop=True)


def articles_for(a: pd.DataFrame, ticker: str, decision_utc: pd.Timestamp) -> list[dict]:
    g = a[(a.ticker == ticker) & (a.published_utc <= decision_utc) & (a.published_utc > decision_utc - LOOKBACK)]
    g = g.sort_values("published_utc", ascending=False).head(MAX_ARTICLES)
    return [{"headline": r.headline, "text": r.text or "", "published_utc": r.published_utc.isoformat(),
             "time_known": True, "source": r.source or "yahoo rss", "ticker_match": "archive (Yahoo RSS by ticker)"}
            for r in g.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from shadow_forecast import with_timeout
    from nexus_alpha.cv import store as ST
    from nexus_alpha.inference.pipeline import NexusPredictor

    bdir, rec = SH.verify_frozen(BASE)
    asof = SH.expected_asof()
    decision = SH.decision_utc_for(asof)
    tonight = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    if not a.dry_run:
        os.environ["NEXUS_LIVE_CACHE"] = str(paths.ARTIFACTS / "live_cache" / tonight)
    arc = archive(min(decision, SH.now_utc()))
    tickers = a.tickers or sorted(ST.sample_index().ticker.unique())
    if a.limit:
        tickers = tickers[: a.limit]
    with_news = [t for t in tickers if articles_for(arc, t, decision)]
    print(f"E29b news-on: session {asof.date()}, {len(arc)} archived headlines, {len(with_news)} of {len(tickers)} "
          f"stocks have an eligible headline", flush=True)
    if not with_news:
        print("nothing to log")
        return
    P = NexusPredictor(bdir)
    rows, failed, t0 = [], [], time.time()
    for i, t in enumerate(with_news):
        if not a.dry_run and SH.now_utc() > decision - pd.Timedelta(minutes=10):
            failed.extend({"ticker": x, "error": "not attempted: decision-time deadline"} for x in with_news[i:])
            break
        arts = articles_for(arc, t, decision)
        try:
            r = with_timeout(lambda: P.predict(t, mode="live", articles=arts, explain=False), TICKER_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            failed.append({"ticker": t, "error": repr(e)[:200]})
            continue
        if str(r.instrument.asof_session) != str(asof.date()):
            failed.append({"ticker": t, "error": f"last bar {r.instrument.asof_session}, expected {asof.date()}"})
            continue
        used = sum(1 for n in (r.evidence.get("news") or []) if n.get("used"))
        run = SH.now_utc()
        for h, f in r.forecasts.items():
            rows.append({"asof_session": str(asof.date()), "ticker": t, "horizon": h, "p_down": f.down_probability,
                         "p_side": f.sideways_probability, "p_up": f.up_probability, "ret_q10": f.return_q10,
                         "ret_q50": f.expected_return, "ret_q90": f.return_q90,
                         "direction_skill_validated": f.direction_skill_validated,
                         "reference_close": r.instrument.reference_price, "data_quality": r.instrument.data_quality,
                         "bundle_id": rec["bundle_id"], "run_utc": run.isoformat(timespec="seconds"),
                         "decision_utc": decision.isoformat(), "late": bool(run > decision),
                         "n_articles_supplied": len(arts), "n_articles_used": used})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(with_news)} done, {time.time() - t0:.0f}s", flush=True)
    if not rows:
        print("no news-on forecasts produced", failed[:3])
        sys.exit(1)
    df = pd.DataFrame(rows)
    if a.dry_run:
        print(f"[dry-run] {asof.date()}: {df.ticker.nunique()} tickers, mean articles used "
              f"{df.groupby('ticker').n_articles_used.first().mean():.1f}, late {int(df.late.sum())}, failed {len(failed)}")
        return
    out = SH.SHADOW_ROOT / VERSION / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{asof.date()}.csv"
    if f.exists():
        print(f"{f.name}: already logged; nothing rewritten")
        return
    df.to_csv(f, index=False, float_format="%.6g")
    SH.append_chain(VERSION, {"run_utc": SH.now_utc().isoformat(timespec="seconds"), "asof_session": str(asof.date()),
                              "file": f"forecasts/{f.name}", "file_sha256": SH.sha256_file(f), "n_rows": len(df),
                              "n_failed": len(failed), "late_rows": int(df.late.sum())})
    probs = SH.verify_chain(VERSION)
    print(f"logged {f.name}: {df.ticker.nunique()} tickers x {df.horizon.nunique()} horizons, late rows "
          f"{int(df.late.sum())}, failed {len(failed)}; chain {'intact' if not probs else probs}", flush=True)
    sys.stdout.flush()
    os._exit(0 if not probs else 1)


if __name__ == "__main__":
    main()
