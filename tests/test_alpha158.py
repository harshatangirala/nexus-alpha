"""Alpha158 (Qlib definitions, nexus_alpha.indicators.alpha158): names, causality, closed forms, parity report."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nexus_alpha.indicators import alpha158 as A

REPO = Path(__file__).resolve().parents[1]


def _bars(n: int = 150, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    c = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * np.exp(rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    v = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v})


def test_names_follow_qlib_without_vwap():
    assert len(A.NAMES) == 157 and len(set(A.NAMES)) == 157
    assert "VWAP0" not in A.NAMES
    assert {"KMID", "KSFT2", "OPEN0", "LOW0", "ROC60", "IMXD30", "CORD5", "VSUMD60"} <= set(A.NAMES)
    assert list(A.alpha158(_bars()).columns) == A.COLUMNS


def test_future_bars_never_change_past_values():
    b = _bars(200)
    full = A.alpha158(b)
    for t in (0, 4, 59, 60, 120):
        pd.testing.assert_frame_equal(A.alpha158(b.iloc[: t + 1]), full.iloc[: t + 1], rtol=1e-12, atol=1e-12)
    b2 = b.copy()
    b2.loc[150:, ["open", "high", "low", "close"]] *= 3
    b2.loc[150:, "volume"] *= 7
    pd.testing.assert_frame_equal(A.alpha158(b2).iloc[:150], full.iloc[:150])


def test_closed_forms_on_a_straight_line():
    n = 80
    c = 10 + 0.5 * np.arange(n)
    b = pd.DataFrame({"open": c - 0.1, "high": c + 0.2, "low": c - 0.3, "close": c, "volume": np.full(n, 1000.0)})
    f = A.alpha158(b)
    t = n - 1
    assert f["a158_KMID"].iloc[t] == pytest.approx(0.1 / (c[t] - 0.1))
    assert f["a158_KLEN"].iloc[t] == pytest.approx(0.5 / (c[t] - 0.1))
    for d in A.WINDOWS:
        w = c[t - d + 1: t + 1]
        assert f[f"a158_BETA{d}"].iloc[t] == pytest.approx(0.5 / c[t])
        assert f[f"a158_RSQR{d}"].iloc[t] == pytest.approx(1.0)
        assert f[f"a158_RESI{d}"].iloc[t] == pytest.approx(0.0, abs=1e-10)
        assert f[f"a158_ROC{d}"].iloc[t] == pytest.approx(c[t - d] / c[t])
        assert f[f"a158_MA{d}"].iloc[t] == pytest.approx(w.mean() / c[t])
        assert f[f"a158_STD{d}"].iloc[t] == pytest.approx(w.std(ddof=1) / c[t])
        assert f[f"a158_IMAX{d}"].iloc[t] == pytest.approx(1.0)        # the newest high is the maximum
        assert f[f"a158_IMIN{d}"].iloc[t] == pytest.approx(1 / d)      # the oldest low is the minimum
        assert f[f"a158_RANK{d}"].iloc[t] == pytest.approx(1.0)
        assert f[f"a158_CNTP{d}"].iloc[t] == pytest.approx(1.0)
        assert f[f"a158_SUMP{d}"].iloc[t] == pytest.approx(1.0)
        assert f[f"a158_VSTD{d}"].iloc[t] == pytest.approx(0.0)
        assert np.isnan(f[f"a158_CORR{d}"].iloc[t])                    # constant volume: Qlib masks the correlation
        lo, hi = w[0] - 0.3, c[t] + 0.2
        assert f[f"a158_RSV{d}"].iloc[t] == pytest.approx((c[t] - lo) / (hi - lo))


def test_partial_windows_use_the_bars_available_so_far():
    f = A.alpha158(_bars(10))
    assert np.isnan(f["a158_STD5"].iloc[0]) and not np.isnan(f["a158_STD5"].iloc[1])     # min_periods=1, ddof=1
    assert f["a158_IMAX60"].iloc[0] == pytest.approx(1 / 60)


def test_missing_bars_raise_and_inf_becomes_missing():
    b = _bars()
    b.loc[10, "close"] = np.nan
    with pytest.raises(ValueError):
        A.alpha158(b)
    b = _bars()
    b.loc[20:25, "volume"] = 0.0                 # log(v / 0 + 1) = inf in Qlib's CORD
    x = A.alpha158(b).to_numpy()
    assert not np.isinf(x).any()


def test_features_at_refuses_stale_or_short_bars():
    b = _bars(120)
    b.insert(0, "date", pd.bdate_range("2024-01-01", periods=len(b)))
    last = b.date.iloc[-1]
    pd.testing.assert_series_equal(A.features_at(b, last), A.alpha158(b).iloc[-1], check_names=False)
    pd.testing.assert_series_equal(A.features_at(b, b.date.iloc[80]), A.alpha158(b.iloc[:81]).iloc[-1],
                                   check_names=False)          # later bars are ignored
    with pytest.raises(ValueError, match="last bar"):
        A.features_at(b, last + pd.Timedelta(days=3))          # the as-of session has no bar: stale cache
    with pytest.raises(ValueError, match="needs at least"):
        A.features_at(b.iloc[:40], b.date.iloc[39])            # too short for 60-session windows


def test_v3_alpha_is_frozen_only_at_horizons_that_passed_live_parity():
    """D-049: the deployed horizons are exactly those whose rank_score agreement (Yahoo vs vendor) was >= 0.95."""
    import json as _json
    rec = _json.loads((REPO / "registry" / "frozen_models" / "v3_alpha.json").read_text(encoding="utf-8"))
    rep = _json.loads((REPO / "reports" / "verification" / "alpha158_live_parity.json").read_text(encoding="utf-8"))
    passed = {h for h, v in rep["end_to_end_by_horizon"].items() if v["spearman_rank_score"] >= 0.95}
    assert set(rec["horizons"]) == passed and rep["criterion_1_price_features"] is True
    assert rec["kind"] == "alpha" and rec["bundle_id"] == rep["bundle"]


def test_qlib_parity_report_shows_identical_definitions():
    rep = json.loads((REPO / "reports" / "verification" / "alpha158_qlib_parity.json").read_text(encoding="utf-8"))
    assert rep["definition_parity"] is True and rep["features"] == 157 and rep["rows"] > 100_000
    assert all(v["nan_mismatch"] == 0 for v in rep["by_feature"].values())
    assert all(v["agree_share_with_float32_inputs"] == 1.0 for v in rep["correlation_diagnostic"].values())
