"""Point-in-time tests for the v2 macro/market layer (synthetic data; no network, no licensed files)."""
import numpy as np
import pandas as pd

from nexus_alpha.macro import features as MF
from nexus_alpha.macro import fred as FR
from nexus_alpha.macro import gmd as G


def _walk(idx, seed, start=100.0, vol=0.01):
    r = np.random.default_rng(seed).normal(0, vol, len(idx))
    return pd.Series(start * np.exp(np.cumsum(r)), index=idx)


def _inputs(n=700):
    idx = pd.bdate_range("2010-01-04", periods=n)
    etf = pd.DataFrame({t: _walk(idx, i) for i, t in enumerate(["SPY", "HYG", "LQD", "IEF", "TLT", "IWM", "QQQ", "EEM"]
                                                                  + MF.SECTORS)})
    spy = pd.DataFrame({"date": idx, "close": etf.SPY.values})
    spy["open"] = spy.close.shift(1).fillna(spy.close) * 1.001
    spy["high"] = spy[["open", "close"]].max(axis=1) * 1.004
    spy["low"] = spy[["open", "close"]].min(axis=1) * 0.996
    mkt = pd.DataFrame({k: _walk(idx, 100 + i, 20 if k.startswith(("vix", "y")) else 100, 0.02)
                        for i, k in enumerate(["vix", "vix3m", "y3m", "y5y", "y10y", "y30y", "dxy", "wti", "gold",
                                               "copper", "spx"])}, index=idx)
    stock = pd.DataFrame({"date": idx, "close": (_walk(idx, 999) * 0.5 + etf.SPY * 0.5).values})
    return idx, etf, spy, mkt, stock


def test_market_features_are_truncation_invariant():
    idx, etf, spy, mkt, stock = _inputs()
    full_d = MF.date_features(etf, spy, None, mkt)
    full_s = MF.stock_relative(stock, etf)
    for cut in (idx[300], idx[450], idx[-2]):
        d = MF.date_features(etf.loc[:cut], spy[spy.date <= cut], None, mkt.loc[:cut])
        s = MF.stock_relative(stock[stock.date <= cut], etf.loc[:cut])
        pd.testing.assert_series_equal(full_d.loc[cut], d.loc[cut], check_names=False)
        pd.testing.assert_series_equal(full_s.loc[cut], s.loc[cut], check_names=False)


def test_gmd_publication_lag():
    panel = pd.DataFrame({"x": np.arange(2000, 2030, dtype=float)}, index=pd.RangeIndex(2000, 2030, name="year"))
    f = G.asof_features(pd.DatetimeIndex(["2009-04-30", "2009-05-01", "2009-12-31", "2010-01-02"]), panel)
    assert list(f.gmd_data_year) == [2007, 2008, 2008, 2008]
    assert list(f.gmd_x) == [2007.0, 2008.0, 2008.0, 2008.0]


def test_alfred_vintages_only_after_release():
    obs = pd.date_range("2017-01-01", "2018-12-01", freq="MS")
    rows = []
    # vintage A released 2019-01-11 (covers through 2018-12); vintage B released 2019-02-13 revises 2018-12 upward
    for o in obs:
        rows.append(("CPIAUCSL", o, pd.Timestamp("2019-01-11"), pd.Timestamp("2019-02-13"), 100 + (o.year - 2017) * 12 + o.month))
        v = 100 + (o.year - 2017) * 12 + o.month + (5 if o == obs[-1] else 0)
        rows.append(("CPIAUCSL", o, pd.Timestamp("2019-02-13"), pd.Timestamp("2262-04-11"), v))
    st = pd.DataFrame(rows, columns=["series_id", "date", "realtime_start", "realtime_end", "value"])
    f = FR.asof_features(pd.DatetimeIndex(["2019-01-10", "2019-01-11", "2019-02-12", "2019-02-13"]), st)
    yoy = f["fred_cpi_yoy"]
    assert np.isnan(yoy.iloc[0])                       # before the first vintage: missing, never a later vintage
    assert yoy.iloc[1] == yoy.iloc[2]                  # vintage A in force
    assert yoy.iloc[3] > yoy.iloc[2]                   # the revision appears only from its release date
