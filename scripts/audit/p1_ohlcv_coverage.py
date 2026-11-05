"""Phase 1: OHLCV coverage audit by ticker and timeframe, on the development (pre-seal) period only.

Reads every bar file through nexus_alpha.data.bars (seal pushed into the parquet scan). Footer row counts give the total
rows per file without reading values, so the number of post-seal rows is known only as a count.

Outputs: data_manifests/ohlcv_coverage_by_ticker.csv, data_manifests/ohlcv_coverage_summary.json
Run: .venv/Scripts/python -I scripts/audit/p1_ohlcv_coverage.py
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nexus_alpha import paths
from nexus_alpha.data import bars as B
from nexus_alpha.data.calendar import calendar
from nexus_alpha.seal import SEAL_DATE

REPO = Path(__file__).resolve().parents[2]


def audit_one(ticker: str) -> dict:
    cal = calendar()
    sess = pd.DatetimeIndex(cal.date)
    early = set(cal.date[cal.early_close])
    r = {"ticker": ticker}
    d = B.read_daily(ticker)
    h = B.read_hourly(ticker)
    r["daily_rows_total"] = B.total_rows("1day", ticker)
    r["hourly_rows_total"] = B.total_rows("1hour", ticker)
    r["weekly_rows_total"] = B.total_rows("1week", ticker)
    r["daily_rows_dev"] = len(d)
    r["daily_rows_after_seal"] = r["daily_rows_total"] - len(d)
    if len(d):
        r["daily_first"], r["daily_last"] = d.date.iloc[0].date(), d.date.iloc[-1].date()
        span = sess[(sess >= d.date.iloc[0]) & (sess <= d.date.iloc[-1])]
        r["daily_missing_sessions"] = int(len(span.difference(pd.DatetimeIndex(d.date))))
        r["daily_extra_non_sessions"] = int(len(pd.DatetimeIndex(d.date).difference(sess)))
        r["daily_duplicates"] = int(d.date.duplicated().sum())
        r["daily_ohlc_violations"] = int(B.ohlc_violations(d).sum())
        src = d.source.value_counts()
        for s in ("1min", "yahoo", "yahoo_ext"):
            r[f"daily_src_{s}"] = int(src.get(s, 0))
        r["daily_zero_volume"] = int((d.volume == 0).sum())
        r["daily_years"] = int(d.date.dt.year.nunique())
    r["hourly_rows_dev"] = len(h)
    r["hourly_rows_after_seal"] = r["hourly_rows_total"] - len(h)
    if len(h):
        et = pd.DatetimeIndex(h.timestamp).tz_convert("America/New_York")
        day = et.tz_localize(None).normalize()
        r["hourly_first"], r["hourly_last"] = day[0].date(), day[-1].date()
        r["hourly_offgrid"] = int(((et.minute != 30) | (et.hour < 9) | (et.hour > 15)).sum())
        r["hourly_duplicates"] = int(pd.Series(h.timestamp).duplicated().sum())
        r["hourly_ohlc_violations"] = int(B.ohlc_violations(h).sum())
        counts = pd.Series(1, index=day).groupby(level=0).sum()
        exp = pd.Series(np.where(counts.index.isin(early), 4, 7), index=counts.index)
        r["hourly_sessions"] = int(len(counts))
        r["hourly_sessions_incomplete"] = int((counts < exp).sum())
        r["hourly_sessions_over"] = int((counts > exp).sum())
        dspan = d[(d.date >= day[0]) & (d.date <= day[-1])]
        r["hourly_sessions_missing_vs_daily"] = int(len(pd.DatetimeIndex(dspan.date).difference(counts.index)))
        # daily vs hourly agreement on 1min-sourced days
        hh = h.assign(day=day.values)
        agg = hh.groupby("day").agg(ho=("open", "first"), hc=("close", "last"), hh=("high", "max"), hl=("low", "min"))
        m = d[d.source == "1min"].set_index("date").join(agg, how="inner")
        if len(m):
            rel = lambda a, b: (a - b).abs() / b.abs()
            r["dh_close_relerr_med"] = float(rel(m.hc, m.close).median())
            r["dh_close_relerr_gt_0p5pct"] = float((rel(m.hc, m.close) > 0.005).mean())
            r["dh_open_relerr_gt_0p5pct"] = float((rel(m.ho, m.open) > 0.005).mean())
            r["dh_compared_days"] = int(len(m))
    # weekly: frozen file vs aggregation of daily bars (contract C-05), dev period
    wf = B.read_weekly_frozen(ticker)
    wd = B.weekly_from_daily(d)
    r["weekly_rows_dev"] = len(wf)
    j = wf.merge(wd, on="week_start", suffixes=("_f", "_d"), how="outer", indicator=True)
    r["weekly_only_frozen"] = int((j["_merge"] == "left_only").sum())
    r["weekly_only_derived"] = int((j["_merge"] == "right_only").sum())
    b = j[j["_merge"] == "both"]
    mism = np.zeros(len(b), bool)
    for c in ("open", "high", "low", "close"):
        mism |= ~np.isclose(b[f"{c}_f"].values, b[f"{c}_d"].values, rtol=1e-9, atol=1e-9)
    mism |= ~np.isclose(b["volume_f"].values, b["volume_d"].values, rtol=1e-6, atol=1e-3)
    mism |= b["trading_days_f"].values != b["trading_days_d"].values
    r["weekly_mismatch_vs_daily"] = int(mism.sum())
    return r


def main() -> int:
    t0 = time.time()
    tickers = B.available_tickers()
    rows = []
    with cf.ProcessPoolExecutor(max_workers=5) as ex:
        for i, r in enumerate(ex.map(audit_one, tickers, chunksize=8)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tickers)} {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    uni = pd.read_csv(paths.UNIVERSE_CSV, usecols=["ticker", "name", "exchange", "is_adr"])
    df = df.merge(uni, on="ticker", how="left")
    df["in_original_941"] = df.name.notna()
    out = REPO / "data_manifests"
    df.to_csv(out / "ohlcv_coverage_by_ticker.csv", index=False)

    # per-year availability (dev period): tickers with any daily bar, and with any hourly bar
    years = range(2001, 2023)
    first_d = pd.to_datetime(df.daily_first)
    last_d = pd.to_datetime(df.daily_last)
    first_h = pd.to_datetime(df.hourly_first)
    per_year = {y: {"tickers_daily": int(((first_d.dt.year <= y) & (last_d.dt.year >= y)).sum()),
                    "tickers_hourly": int(((first_h.dt.year <= y) & (last_d.dt.year >= y)).sum())} for y in years}
    num = lambda c: df[c].fillna(0)
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seal_date": str(SEAL_DATE.date()),
        "n_tickers": int(len(df)),
        "n_original_941": int(df.in_original_941.sum()),
        "exchange_counts": df.exchange.fillna("unknown (38 additions)").value_counts().to_dict(),
        "adr_count": int(df.is_adr.fillna(False).astype(bool).sum()),
        "daily": {"rows_dev": int(num("daily_rows_dev").sum()), "rows_after_seal_count_only": int(num("daily_rows_after_seal").sum()),
                  "first_min": str(first_d.min().date()), "first_max": str(first_d.max().date()),
                  "last_min": str(last_d.min().date()), "last_max": str(last_d.max().date()),
                  "tickers_full_2001_2022": int(((first_d <= "2001-01-31") & (last_d >= "2022-12-29")).sum()),
                  "missing_sessions_total": int(num("daily_missing_sessions").sum()),
                  "tickers_with_missing_sessions": int((num("daily_missing_sessions") > 0).sum()),
                  "duplicates_total": int(num("daily_duplicates").sum()),
                  "ohlc_violations_total": int(num("daily_ohlc_violations").sum()),
                  "source_1min": int(num("daily_src_1min").sum()), "source_yahoo": int(num("daily_src_yahoo").sum()),
                  "source_yahoo_ext": int(num("daily_src_yahoo_ext").sum()),
                  "zero_volume_days": int(num("daily_zero_volume").sum()),
                  "years_per_ticker_quantiles": df.daily_years.quantile([0, .1, .25, .5, .75, .9, 1]).round(1).to_dict()},
        "hourly": {"rows_dev": int(num("hourly_rows_dev").sum()), "rows_after_seal_count_only": int(num("hourly_rows_after_seal").sum()),
                   "first_min": str(first_h.min().date()), "first_max": str(first_h.max().date()),
                   "sessions_total": int(num("hourly_sessions").sum()),
                   "sessions_incomplete": int(num("hourly_sessions_incomplete").sum()),
                   "sessions_over_expected": int(num("hourly_sessions_over").sum()),
                   "sessions_missing_vs_daily": int(num("hourly_sessions_missing_vs_daily").sum()),
                   "offgrid_bars": int(num("hourly_offgrid").sum()),
                   "duplicates_total": int(num("hourly_duplicates").sum()),
                   "ohlc_violations_total": int(num("hourly_ohlc_violations").sum()),
                   "daily_vs_hourly_close_relerr_median_of_medians": float(df.dh_close_relerr_med.median()),
                   "share_days_close_relerr_gt_0p5pct_mean": float(df.dh_close_relerr_gt_0p5pct.mean()),
                   "share_days_open_relerr_gt_0p5pct_mean": float(df.dh_open_relerr_gt_0p5pct.mean())},
        "weekly": {"rows_dev": int(num("weekly_rows_dev").sum()),
                   "mismatch_vs_daily_total": int(num("weekly_mismatch_vs_daily").sum()),
                   "only_in_frozen": int(num("weekly_only_frozen").sum()),
                   "only_in_derived": int(num("weekly_only_derived").sum())},
        "per_year_dev": per_year,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "ohlcv_coverage_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_year_dev"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
