"""Build 1-hour, 1-day and 1-week bars for the index / sector / macro-proxy ETFs (SPY = the S&P 500 tracker).

Same conventions as TriFrame's frozen stock bars (data contract 2.1), so SPY charts and features are built exactly like
the stocks':
* regular session only (09:30-16:00 New York; early-close sessions end at 13:00 per the market calendar);
* hourly bars stamped at their start, 09:30 ... 15:30 New York (stored in UTC), the last one 30 minutes long;
* daily open = first regular-session minute, close = last; split-adjusted (Yahoo split history), not dividend-adjusted;
* weekly bars built from daily with `bars.weekly_from_daily`;
* sessions missing from the minute feed are filled from Yahoo daily bars (`source = yahoo`) and have no hourly bars.
Every daily close is compared with Yahoo's; the agreement is written to reports/data/etf_bar_verification.csv.
Bars are stored in artifacts/etf/ (git-ignored) and read through sealed readers (nexus_alpha.data.etf_bars).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.data.calendar import calendar  # noqa: E402
from nexus_alpha.seal import SEAL_DATE, SEAL_UTC  # noqa: E402


def yahoo(sym: str, include_sealed: bool = False) -> tuple[pd.DataFrame, pd.Series, float, pd.Timestamp]:
    """Yahoo daily bars before the seal, pre-seal splits, the product of post-seal splits (Yahoo's history is adjusted
    for them; our bars are not, so the comparison undoes it) and the fund's first Yahoo date (inception)."""
    import yfinance as yf
    t = yf.Ticker(sym)
    d = t.history(period="max", auto_adjust=False, actions=True)
    di = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    daily = pd.DataFrame({"date": di, "open": d["Open"].values, "high": d["High"].values, "low": d["Low"].values,
                          "close": d["Close"].values, "volume": d["Volume"].values})
    splits = pd.Series(d["Stock Splits"].values, index=di)
    splits = splits[splits > 0]
    inception = daily.date.min()
    if include_sealed:                      # one share basis across the whole series: every split, no post-seal factor
        return daily, splits, 1.0, inception
    post = float(np.prod(splits[splits.index >= SEAL_DATE].values)) if len(splits) else 1.0
    return daily[daily.date < SEAL_DATE], splits[splits.index < SEAL_DATE], post, inception


def main() -> None:
    build(include_sealed=False)


def build(include_sealed: bool = False) -> None:
    """Development bars (before the seal) into artifacts/etf; with include_sealed, the whole series into
    artifacts/etf_sealed, which only the opened sealed evaluation reads (development artifacts are never modified)."""
    if include_sealed:
        from nexus_alpha import seal
        seal.require_open("ETF bars on/after the seal")
    src = sorted((paths.ARTIFACTS / "raw" / "etf_1m").glob("*.parquet"))
    raw = pd.concat([pd.read_parquet(p) for p in src], ignore_index=True)
    cal = calendar()
    if not include_sealed:
        raw = raw[raw.timestamp < SEAL_UTC]
        cal = cal[cal.date < SEAL_DATE]
    early = set(cal.date[cal.early_close.astype(bool)])
    sessions = pd.DatetimeIndex(cal.date)
    et = raw.timestamp.dt.tz_convert("America/New_York")
    raw = raw.assign(et=et, day=et.dt.tz_localize(None).dt.normalize())
    mins = et.dt.hour * 60 + et.dt.minute
    close_min = np.where(raw.day.isin(early), 13 * 60, 16 * 60)
    raw = raw[(mins >= 9 * 60 + 30) & (mins < close_min) & raw.day.isin(sessions)]
    out_root = paths.ARTIFACTS / ("etf_sealed" if include_sealed else "etf")
    ver = []
    for tk, g in raw.groupby("ticker"):
        g = g.sort_values("timestamp")
        yd, splits, post_seal_split, inception = yahoo(tk, include_sealed)
        # ticker reuse: the minute feed has rows under some ETF symbols before the fund existed (e.g. VOO, EEM)
        g = g[g.day >= inception]
        if g.empty:
            continue
        for c in ("open", "high", "low", "close"):
            yd[c] = yd[c] * post_seal_split          # back to the pre-seal share basis of our bars
        yd["volume"] = yd["volume"] / post_seal_split
        # split adjustment: prices before a split date are divided by the cumulative later split factor
        fac = pd.Series(1.0, index=g.index)
        vfac = pd.Series(1.0, index=g.index)
        for sd, ratio in splits.items():
            before = g.day < sd
            fac[before] /= ratio
            vfac[before] *= ratio
        for c in ("open", "high", "low", "close"):
            g[c] = g[c] * fac
        g["volume"] = g["volume"] * vfac
        gd = g.groupby("day")
        daily = pd.DataFrame({"open": gd.open.first(), "high": gd.high.max(), "low": gd.low.min(),
                              "close": gd.close.last(), "volume": gd.volume.sum(), "minute_bars": gd.size()})
        daily.index.name = "date"
        daily = daily.reset_index()
        daily["source"] = "minute"
        # fill sessions missing from the feed (inside the ETF's life) from Yahoo
        first = max(daily.date.min(), yd.date.min()) if len(yd) else daily.date.min()
        life = sessions[(sessions >= first)]
        miss = life.difference(pd.DatetimeIndex(daily.date))
        fill = yd[yd.date.isin(miss)].assign(minute_bars=0, source="yahoo")
        daily = pd.concat([daily, fill[daily.columns]], ignore_index=True).sort_values("date").reset_index(drop=True)
        # hourly
        slot = ((g.et.dt.hour * 60 + g.et.dt.minute) - (9 * 60 + 30)) // 60
        g = g.assign(slot=slot)
        gh = g.groupby(["day", "slot"])
        hourly = pd.DataFrame({"open": gh.open.first(), "high": gh.high.max(), "low": gh.low.min(),
                               "close": gh.close.last(), "volume": gh.volume.sum(), "minute_bars": gh.size()}).reset_index()
        hourly["timestamp"] = (pd.to_datetime(hourly.day) + pd.Timedelta(hours=9, minutes=30)
                               + pd.to_timedelta(hourly.slot, unit="h")).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        hourly = hourly[["timestamp", "open", "high", "low", "close", "volume", "minute_bars"]]
        weekly = B.weekly_from_daily(daily[["date", "open", "high", "low", "close", "volume"]])
        for tf, df in (("1day", daily), ("1hour", hourly), ("1week", weekly)):
            p = out_root / tf / f"{tk}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
        m = daily.merge(yd[["date", "close"]], on="date", suffixes=("", "_yahoo"))
        m = m[m.source == "minute"]
        e = (m.close / m.close_yahoo - 1) * 100
        viol = int(B.ohlc_violations(daily).sum())
        ver.append({"ticker": tk, "first_day": str(daily.date.min().date()), "last_day": str(daily.date.max().date()),
                    "sessions": len(daily), "yahoo_filled": int((daily.source == "yahoo").sum()),
                    "hourly_bars": len(hourly), "splits_applied": len(splits), "post_seal_split_factor": post_seal_split,
                    "inception_yahoo": str(inception.date()),
                    "close_vs_yahoo_std_pct": round(float(e.std()), 4),
                    "close_within_0p5pct": round(float((e.abs() <= 0.5).mean() * 100), 2),
                    "close_within_2pct": round(float((e.abs() <= 2).mean() * 100), 2), "ohlc_violations": viol})
        print(ver[-1], flush=True)
    rep = REPO / "reports" / ("sealed" if include_sealed else "data")
    rep.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ver).to_csv(rep / "etf_bar_verification.csv", index=False)


if __name__ == "__main__":
    main()
