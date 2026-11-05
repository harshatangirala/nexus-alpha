import numpy as np
import pandas as pd
import pytest

from nexus_alpha.data import bars as B
from nexus_alpha.data.calendar import offset_session
from nexus_alpha.indicators.core import STRUCTURED_ALL, WARM, indicators, structured_features
from nexus_alpha.seal import SEAL_DATE, SEAL_UTC
from nexus_alpha.targets import COST_FLOOR, HORIZONS, make_labels

from .conftest import source_data, synthetic_daily, synthetic_hourly


# ------------------------------------------------------------------ causality: changing the future never changes the past
def test_structured_features_ignore_future_bars():
    d = synthetic_daily(500, seed=3)
    h = synthetic_hourly(d)
    asof = d.date.iloc[[260, 300, 350]]
    f0 = structured_features(d, h, asof)
    d2 = d.copy()
    k = 351                                                     # corrupt every bar after the last as-of date
    d2.loc[k:, ["open", "high", "low", "close"]] *= np.random.default_rng(9).uniform(0.5, 2.0, (len(d2) - k, 1))
    d2.loc[k:, "high"] = d2.loc[k:, ["open", "high", "low", "close"]].max(axis=1)
    d2.loc[k:, "low"] = d2.loc[k:, ["open", "high", "low", "close"]].min(axis=1)
    d2.loc[k:, "volume"] *= 7
    h2 = synthetic_hourly(d2)              # same draws per day: hourly bars up to session 350 are identical, later differ
    sess = pd.DatetimeIndex(h.timestamp).tz_convert("America/New_York").tz_localize(None).normalize()
    early = np.asarray(sess <= d.date.iloc[350])
    pd.testing.assert_frame_equal(h[early].reset_index(drop=True), h2[early].reset_index(drop=True))
    assert not np.allclose(h.close.values[~early], h2.close.values[~early])
    f1 = structured_features(d2, h2, asof)
    pd.testing.assert_frame_equal(f0, f1)


def test_weekly_features_use_completed_weeks_only():
    d = synthetic_daily(300, seed=5)
    h = synthetic_hourly(d)
    # pick a Wednesday: weekly indicators must equal those at the preceding Friday's completed week
    wed = d.date[d.date.dt.day_of_week == 2].iloc[-5]
    fri_prev = d.date[(d.date < wed) & (d.date.dt.day_of_week == 4)].iloc[-1]
    f = structured_features(d, h, pd.DatetimeIndex([wed, fri_prev]))
    wcols = [c for c in f.columns if c.startswith("w_") and c != "w_wtd_ret"]
    np.testing.assert_allclose(f.loc[wed, wcols].values, f.loc[fri_prev, wcols].values, equal_nan=True)
    assert abs(f.loc[fri_prev, "w_wtd_ret"]) < 1e-12               # Friday close is the completed week's close


def test_indicator_warmup_and_ranges():
    d = synthetic_daily(260, seed=2)
    x = indicators(d)
    assert x.sma20.iloc[:19].isna().all() and x.sma20.iloc[19:].notna().all()
    assert x.sma50.iloc[:49].isna().all()
    assert ((x.rsi14.dropna() >= 0) & (x.rsi14.dropna() <= 100)).all()
    assert (x.atr14 > 0).all()
    assert not x.warm.iloc[WARM - 2] and x.warm.iloc[WARM - 1]
    np.testing.assert_allclose(x.bb_up - x.bb_mid, x.bb_mid - x.bb_lo, rtol=1e-12, equal_nan=True)


def test_feature_list_is_complete_and_finite_after_warmup():
    d = synthetic_daily(500, seed=4)
    h = synthetic_hourly(d)
    f = structured_features(d, h, d.date.iloc[300:310])
    assert set(STRUCTURED_ALL) <= set(f.columns)
    assert np.isfinite(f[STRUCTURED_ALL].values).all()


def test_hourly_features_are_nan_when_the_asof_session_has_no_hourly_bars():
    d = synthetic_daily(300, seed=6)
    h = synthetic_hourly(d)
    day = d.date.iloc[280]
    sess = pd.DatetimeIndex(h.timestamp).tz_convert("America/New_York").tz_localize(None).normalize()
    f = structured_features(d, h[sess != day], pd.DatetimeIndex([day]))
    assert f.filter(like="h_").isna().all(axis=None)


# ------------------------------------------------------------------ labels
def test_labels_match_hand_computation():
    d = synthetic_daily(300, seed=7)
    lab = make_labels(d)
    t = 250
    for name, h in HORIZONS.items():
        r = d.close.iloc[t + h] / d.open.iloc[t + 1] - 1
        tau = max(0.5 * lab.sigma_ewma.iloc[t] * np.sqrt(h), COST_FLOOR)
        y = 2 if r > tau else (0 if r < -tau else 1)
        assert lab[f"ret_{name}"].iloc[t] == pytest.approx(r, rel=1e-12)
        assert lab[f"y_{name}"].iloc[t] == y
        assert lab[f"end_{name}"].iloc[t] == d.date.iloc[t + h]


def test_labels_are_nan_where_the_window_is_incomplete():
    d = synthetic_daily(120, seed=8)
    lab = make_labels(d)
    assert lab["y_1M"].iloc[-21:].isna().all() and lab["y_1M"].iloc[-22:-21].notna().all()
    assert lab["y_1D"].iloc[-1:].isna().all() and lab["y_1D"].iloc[-2:-1].notna().all()
    # label ends come from the calendar even past the data
    assert lab["end_1M"].notna().all()


def test_sigma_is_causal():
    d = synthetic_daily(200, seed=10)
    a = make_labels(d).sigma_ewma.iloc[150]
    d2 = d.copy()
    d2.loc[151:, "close"] *= 3
    assert make_labels(d2).sigma_ewma.iloc[150] == a


# ------------------------------------------------------------------ real data (sealed)
@source_data
def test_sealed_readers_never_return_sealed_rows():
    for t in ("AAPL", "MSFT", "JPM"):
        d = B.read_daily(t)
        h = B.read_hourly(t)
        assert d.date.max() < SEAL_DATE and len(d) > 4000
        assert h.timestamp.max() < SEAL_UTC
        assert B.total_rows("1day", t) > len(d)              # the file does hold later rows: the seal is doing work


@source_data
def test_indicators_match_frozen_triframe_file_on_dev_period():
    import pyarrow.parquet as pq
    from nexus_alpha import paths
    worst = 0.0
    for t in ("AAPL", "XOM", "JNJ", "AAON"):
        d = B.read_daily(t)
        x = indicators(d)
        fz = pq.read_table(paths.OHLCV_FINAL / "meta" / "indicators" / f"{t}.parquet",
                           filters=[("date", "<", SEAL_DATE.to_pydatetime())]).to_pandas()
        assert len(fz) == len(d)
        for col in ("sma20", "sma50", "macd", "macd_hist", "rsi14", "bb_up", "bb_lo", "atr14", "vol_ratio"):
            a, b = x[col].values, fz[col].values
            m = np.isfinite(a) & np.isfinite(b)
            assert (np.isfinite(a) == np.isfinite(b)).all(), (t, col)
            scale = np.maximum(np.abs(b[m]), 1e-9) if col not in ("macd", "macd_hist") else d.close.values[m]
            worst = max(worst, float(np.max(np.abs(a[m] - b[m]) / scale)))
    assert worst < 1e-9, worst


@source_data
def test_weekly_window_partial_flag_uses_calendar():
    sb = B.StockBars("AAPL")
    k_last = len(sb.dates) - 1                                 # 2022-12-30, a Friday: week complete
    assert not sb.weekly_window(k_last, 10).partial.iloc[-1]
    k_wed = sb.k_of("2022-12-28")
    w = sb.weekly_window(k_wed, 10)
    assert w.partial.iloc[-1] and w.close.iloc[-1] == sb.d.close.iloc[k_wed]
    assert offset_session(pd.DatetimeIndex([sb.dates[k_wed]]), 1)[0] == pd.Timestamp("2022-12-29")
