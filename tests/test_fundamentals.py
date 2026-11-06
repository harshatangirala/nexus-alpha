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


def test_development_loader_never_returns_post_seal_facts():
    from nexus_alpha import seal
    from nexus_alpha.fundamentals import sec_facts as SF
    f = SF.load(320193)                                    # Apple: its raw file also holds 2023-2026 filings
    assert len(f) and f.filed.max() < seal.SEAL_DATE
    assert not seal.is_open()


def _snap_row(**kw):
    base = {"snap_date": T("2021-11-01"), "shares_outstanding": 100.0, "shares_date": T("2021-10-20"),
            "q_diluted_shares": 102.0, "q_diluted_shares_prev": 101.0, "q_eps_diluted": 1.0, "q_eps_diluted_prev": 1.0,
            "ttm_net_income": 400.0, "ttm_eps_diluted": 4.0, "filed_form": "10-Q"}
    for c in ST.INSTANTS:
        base.setdefault(c, np.nan)
        base.setdefault(f"{c}_prev", np.nan)
    for c in ST.TTM_CONCEPTS:
        base.setdefault(f"ttm_{c}", np.nan)
        base.setdefault(f"ttm_{c}_prev", np.nan)
    base.update(kw)
    return pd.DataFrame([base])


def _bars(close=10.0, d="2021-12-31"):
    return pd.DataFrame({"date": [T(d)], "close": [close]})


def test_splits_after_vendor_basis_are_not_applied_to_price():
    # audit 3 (KLAC): a 10:1 split after the vendor's adjustment date is not in the vendor's prices
    late = pd.Series([10.0], index=pd.DatetimeIndex([RT.VENDOR_SPLIT_BASIS + pd.Timedelta(days=100)]))
    d = RT.daily(_snap_row(), _bars(), late)
    assert d.price_as_traded.iloc[0] == 10.0 and np.isclose(d.market_cap.iloc[0], 1000.0)
    early = pd.Series([4.0], index=pd.DatetimeIndex([T("2024-06-10")]))           # applied by the vendor: undo it
    assert RT.daily(_snap_row(), _bars(2.5), early).price_as_traded.iloc[0] == 10.0


def test_share_basis_guards():
    assert np.isnan(RT.daily(_snap_row(filed_form="20-F"), _bars(), None).market_cap.iloc[0])          # ADS vs ordinary
    assert RT.daily(_snap_row(filed_form="20-F"), _bars(), None).foreign_filer.iloc[0] == 1.0
    assert np.isnan(RT.daily(_snap_row(), _bars(), None, share_class_ambiguous=True).market_cap.iloc[0])  # BRK.A / BRK.B
    assert np.isnan(RT.daily(_snap_row(shares_outstanding=1000.0), _bars(), None).market_cap.iloc[0])   # 10x disagreement
    fb = RT.daily(_snap_row(shares_outstanding=np.nan), _bars(), None)                                   # cover missing
    assert np.isclose(fb.market_cap.iloc[0], 1020.0)
    ni_eps = RT.daily(_snap_row(shares_outstanding=np.nan, q_diluted_shares=np.nan), _bars(), None)      # NI / EPS
    assert np.isclose(ni_eps.market_cap.iloc[0], 1000.0)


def test_weighted_shares_are_not_differenced():
    # weighted diluted shares are an average: Q4 must not be FY - 9M (audit 3 found 0 and negative values)
    rows = []
    s = T("2020-01-01")
    for k, e in enumerate(pd.date_range("2020-03-31", periods=4, freq="QE")):
        rows.append(fact("diluted_shares", "flow", 100.0, s, e, e + pd.Timedelta(days=40), form="10-K" if k == 3 else "10-Q"))
    for e in pd.date_range("2020-03-31", periods=3, freq="QE"):
        rows.append(fact("diluted_shares", "flow", 100.0, e - pd.Timedelta(days=89), e, e + pd.Timedelta(days=40)))
    f = pd.DataFrame(rows)
    f.loc[f.concept == "diluted_shares", "unit"] = "shares"
    r = ST.snapshots(f).iloc[-1]
    assert r.q_diluted_shares == 100.0


def test_stale_cover_share_count_is_not_reused():
    f = quarters_facts()
    old = fact("shares_outstanding", "instant", 50.0, None, "2015-04-30", "2015-05-01")
    s = ST.snapshots(pd.concat([f, pd.DataFrame([old])], ignore_index=True))
    assert np.isnan(snap_at(s, "2021-03-01").shares_outstanding)


def test_live_snapshot_keeps_the_latest_form():
    # audit 3: a live snapshot (non-filing day) must still know the filer type, or the 20-F guard never fires (TSM)
    f = quarters_facts()
    f.loc[f.filed == f.filed.max(), "form"] = "20-F"
    s = ST.snapshots(f, at=[f.filed.max() + pd.Timedelta(days=30)])
    assert s.filed_form.iloc[-1] == "20-F"


def test_vendor_split_basis_contract():
    """Data contract (audit 3): the vendor's prices carry splits up to VENDOR_SPLIT_BASIS and none after. As-traded
    closes on 2021-12-31: KLAC 430.11 (10:1 in 2026-06 not applied by the vendor), GOOG 2,893.59 (20:1 in 2022),
    NVDA 294.11 (10:1 in 2024). A vendor re-adjustment would move these and must be caught here."""
    import pytest
    from tests.conftest import have_source_data
    if not have_source_data():
        pytest.skip("source data not available")
    from nexus_alpha.data import bars as B
    from nexus_alpha.fundamentals import splits as SP
    for t, want in (("KLAC", 430.11), ("GOOG", 2893.59), ("NVDA", 294.11)):
        b = B.read_daily(t)
        close = float(b[b.date == "2021-12-31"].close.iloc[0])
        f = RT.split_factor_after(pd.DatetimeIndex(["2021-12-31"]), SP.for_ticker(t)[lambda x: x.index <= RT.VENDOR_SPLIT_BASIS])
        assert abs(close * f[0] / want - 1) < 0.005, (t, close * f[0], want)
