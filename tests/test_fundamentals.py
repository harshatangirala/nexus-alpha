"""Point-in-time fundamentals (E22): no look-ahead, restatements only when filed, YTD chains, staleness, splits."""
import numpy as np
import pandas as pd

from nexus_alpha.fundamentals import ratios as RT, statements as ST

T = pd.Timestamp


def fact(concept, kind, value, start, end, filed, rank=0, form="10-Q"):
    return {"cik": 1, "concept": concept, "tag": concept, "rank": rank, "kind": kind, "unit": "USD", "value": value,
            "start": T(start) if start else pd.NaT, "end": T(end), "filed": T(filed), "form": form, "accn": f"a{filed}",
            "fy": None, "fp": None}


def quarters_facts():
    rows = []
    # revenue: 3-month facts for 8 quarters (2019Q1..2020Q4), 100 each, filed 40 days after quarter end
    ends = pd.date_range("2019-03-31", periods=8, freq="QE")
    for e in ends:
        rows.append(fact("revenue", "flow", 100.0, e - pd.Timedelta(days=89), e, e + pd.Timedelta(days=40)))
    # cash from operations reported only as YTD (Q1 3M, then 6M, 9M, FY) per fiscal year
    for y in (2019, 2020):
        s = T(f"{y}-01-01")
        for k, e in enumerate(pd.date_range(f"{y}-03-31", periods=4, freq="QE")):
            rows.append(fact("cfo", "flow", 30.0 * (k + 1), s, e, e + pd.Timedelta(days=40),
                             form="10-K" if k == 3 else "10-Q"))
    rows.append(fact("assets", "instant", 1000.0, None, "2020-12-31", "2021-02-09"))
    rows.append(fact("assets", "instant", 900.0, None, "2019-12-31", "2020-02-09"))
    rows.append(fact("equity", "instant", 500.0, None, "2017-12-31", "2018-02-09"))          # stale: tag stopped
    return pd.DataFrame(rows)


def snap_at(s, d):
    return s[s.snap_date <= T(d)].iloc[-1]


def test_ttm_and_ytd_chains():
    s = ST.snapshots(quarters_facts())
    r = snap_at(s, "2021-03-01")
    assert r.ttm_revenue == 400.0 and r.ttm_revenue_prev == 400.0
    assert r.ttm_cfo == 120.0                       # quarters from 3M + (6M-3M) + (9M-6M) + (FY-9M), 30 each
    assert r.assets == 1000.0 and r.assets_prev == 900.0
    assert np.isnan(r.equity)                       # stale instant (2017) is missing, not carried forward


def test_no_lookahead_and_restatement_only_when_filed():
    f = quarters_facts()
    base = ST.snapshots(f)
    late = pd.DataFrame([fact("revenue", "flow", 999.0, "2020-10-01", "2020-12-31", "2021-06-01", form="10-K/A")])
    s = ST.snapshots(pd.concat([f, late], ignore_index=True))
    assert snap_at(s, "2021-03-01").ttm_revenue == snap_at(base, "2021-03-01").ttm_revenue == 400.0
    assert snap_at(s, "2021-06-02").ttm_revenue == 300.0 + 999.0      # the restatement counts once it is filed


def test_split_factor_and_asis_price():
    splits = pd.Series([4.0, 10.0], index=pd.DatetimeIndex(["2021-07-20", "2024-06-10"]))
    f = RT.split_factor_after(pd.DatetimeIndex(["2021-07-15", "2021-07-20", "2023-01-01", "2025-01-01"]), splits)
    assert list(f) == [40.0, 10.0, 10.0, 1.0]


def test_dcf_monotone_in_fcf():
    v = RT.dcf_equity(np.array([100.0, 200.0]), np.array([0.05, 0.05]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert v[1] > v[0] > 0
