"""As-known-at-t financial statements from company facts (only facts filed on or before t are used).

For every distinct filing date F of a company, `snapshots` returns one row with:
* TTM flows (sum of the last four consecutive quarters as known at F) and the TTM four quarters earlier;
* the latest quarter's diluted EPS and diluted shares, and those of four quarters earlier;
* balance-sheet instants at the latest period end on or before F and about one year earlier;
* shares outstanding from the cover page (dei) and its date.
Quarterly values come from 3-month facts; a missing fourth quarter is the fiscal year minus the 9-month year-to-date
with the same start. For each period the value used is the latest one filed on or before F, and among concepts with
several tags the highest-priority tag with a value wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QDUR = (80, 100)
YDUR = (350, 380)
NINE = (260, 290)
TTM_CONCEPTS = ["revenue", "cogs", "gross_profit", "operating_income", "pretax_income", "net_income", "eps_diluted", "da",
                "interest_expense", "income_tax", "rnd", "sga", "cfo", "capex", "dividends", "buybacks"]
LAST_Q = ["eps_diluted", "diluted_shares"]
INSTANTS = ["assets", "assets_current", "liabilities", "liabilities_current", "equity", "cash", "lt_debt_total",
            "lt_debt_noncurrent", "lt_debt_current", "st_borrowings", "inventory", "receivables", "retained_earnings", "ppe"]


def _best(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per period: highest-priority tag, then the latest filing."""
    return df.sort_values(["rank", "filed"], ascending=[True, False]).drop_duplicates(keys, keep="first")


def _quarters(flow: pd.DataFrame) -> pd.Series:
    """Quarterly values indexed by period end (as known in `flow`).

    Direct 3-month facts first. Missing quarters come from year-to-date chains with the same fiscal-year start
    (10-Q cash-flow statements report 6- and 9-month YTD): Q = YTD(end) - YTD(previous end), e.g. Q2 = 6M - 3M,
    Q3 = 9M - 6M, Q4 = 12M - 9M."""
    if flow.empty:
        return pd.Series(dtype=float)
    f = flow.assign(dur=(flow.end - flow.start).dt.days)
    q = _best(f[(f.dur >= QDUR[0]) & (f.dur <= QDUR[1])], ["end"]).set_index("end").value
    q = q[~q.index.duplicated()]
    ytd = _best(f[f.dur >= QDUR[0]], ["start", "end"])
    add = {}
    for st, g in ytd.groupby("start"):
        g = g.sort_values("end")
        ends, vals, durs = g.end.tolist(), g.value.tolist(), g.dur.tolist()
        for i in range(1, len(g)):
            step = (ends[i] - ends[i - 1]).days
            if not (75 <= step <= 105) or durs[i] > YDUR[1]:
                continue
            e = ends[i]
            if not any(abs((e - k).days) <= 20 for k in list(q.index) + list(add)):
                add[e] = vals[i] - vals[i - 1]
    if add:
        q = pd.concat([q, pd.Series(add)])
    q = q[~q.index.duplicated()].sort_index()
    # collapse ends a few days apart (52/53-week fiscal calendars report slightly different end dates)
    keep = [i for i in range(len(q)) if i == len(q) - 1 or (q.index[i + 1] - q.index[i]).days > 20]
    return q.iloc[keep]


WINDOW = pd.Timedelta(days=1100)   # TTM and the TTM four quarters earlier need about 2 years; YTD chains a little more
STALE_DAYS = 45      # an item whose latest period ends > 45 days before the company's latest period is stale (missing)


def _ttm(q: pd.Series, offset: int) -> tuple[float, pd.Timestamp | None]:
    """Sum of 4 consecutive quarters ending `offset` quarters before the latest; NaN if not consecutive."""
    if len(q) < 4 + offset:
        return np.nan, None
    w = q.iloc[len(q) - 4 - offset: len(q) - offset]
    gaps = np.diff(w.index.values).astype("timedelta64[D]").astype(int)
    if not ((gaps >= 75) & (gaps <= 105)).all():
        return np.nan, None
    return float(w.sum()), w.index[-1]


def _lastq(q: pd.Series, offset: int) -> float:
    return float(q.iloc[-1 - offset]) if len(q) > offset else np.nan


def snapshots(facts: pd.DataFrame, at: list | None = None) -> pd.DataFrame:
    """at: compute only at these dates (default: every filing date)."""
    if facts.empty:
        return pd.DataFrame()
    facts = facts.sort_values("filed")
    filed_dates = sorted(facts.filed.unique()) if at is None else sorted(pd.to_datetime(at))
    flows = {c: g for c, g in facts[facts.kind == "flow"].groupby("concept")}
    inst = {c: g for c, g in facts[facts.kind == "instant"].groupby("concept")}
    rows = []
    for F in filed_dates:
        r = {"snap_date": F}
        latest_end = None
        ends = {}
        for c in TTM_CONCEPTS + ["diluted_shares"]:
            g = flows.get(c)
            q = _quarters(g[(g.filed <= F) & (g.end >= F - WINDOW)]) if g is not None else pd.Series(dtype=float)
            if c in TTM_CONCEPTS:
                r[f"ttm_{c}"], end = _ttm(q, 0)
                r[f"ttm_{c}_prev"], _ = _ttm(q, 4)
                ends[c] = end
            if c in LAST_Q:
                r[f"q_{c}"] = _lastq(q, 0)
                r[f"q_{c}_prev"] = _lastq(q, 4)
                ends[f"q_{c}"] = q.index[-1] if len(q) else None
        g_a = inst.get("assets")
        a_end = g_a[g_a.filed <= F].end.max() if g_a is not None and len(g_a[g_a.filed <= F]) else None
        known = [e for e in list(ends.values()) + [a_end] if e is not None and not pd.isna(e)]
        latest_end = max(known) if known else None
        r["period_end"] = latest_end
        if latest_end is not None:          # drop flows whose latest quarter is stale relative to the company's latest period
            for c, e in ends.items():
                if e is None or (latest_end - e).days > STALE_DAYS:
                    key = c if c.startswith("q_") else f"ttm_{c}"
                    r[key] = np.nan
                    r[f"{key}_prev"] = np.nan
        for c in INSTANTS:
            g = inst.get(c)
            if g is None:
                r[c] = r[f"{c}_prev"] = np.nan
                continue
            k = _best(g[(g.filed <= F) & (g.end >= F - WINDOW)], ["end"]).sort_values("end")
            if k.empty:
                r[c] = r[f"{c}_prev"] = np.nan
                continue
            last = k.iloc[-1]
            if latest_end is not None and (latest_end - last.end).days > STALE_DAYS:
                r[c] = r[f"{c}_prev"] = np.nan          # stale instant (the tag stopped being reported)
                continue
            r[c] = float(last.value)
            r[f"{c}_end"] = last.end
            prev = k[(last.end - k.end).dt.days.between(320, 410)]
            r[f"{c}_prev"] = float(prev.value.iloc[-1]) if len(prev) else np.nan
        g = inst.get("shares_outstanding")
        if g is not None and len(g[g.filed <= F]):
            k = g[g.filed <= F].sort_values(["end", "filed"]).iloc[-1]
            r["shares_outstanding"], r["shares_date"] = float(k.value), k.end
        else:
            r["shares_outstanding"], r["shares_date"] = np.nan, pd.NaT
        f_today = facts[facts.filed == F]
        r["filed_form"] = f_today.form.iloc[0] if len(f_today) else None
        rows.append(r)
    s = pd.DataFrame(rows)
    return s
