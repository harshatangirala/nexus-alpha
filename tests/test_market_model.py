"""E19 live features must use the same arithmetic as development (D-031)."""
import numpy as np
import pandas as pd

from nexus_alpha.data import bars as B
from nexus_alpha.indicators.market import market_context
from nexus_alpha.macro.market_model import UNIVERSE_COLS, universe_context


def test_universe_context_matches_development_arithmetic():
    tickers = list(B.available_tickers())[:12]
    dev = market_context(tickers, workers=4)
    panel = pd.concat({t: B.read_daily(t).set_index("date").close for t in tickers}, axis=1).sort_index()
    live = universe_context(panel)
    common = dev.index.intersection(live.index)
    assert len(common) > 1000
    for c in UNIVERSE_COLS:
        a, b = dev.loc[common, c].to_numpy(float), live.loc[common, c].to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        assert (np.isnan(a) == np.isnan(b)).mean() > 0.999, c
        assert np.allclose(a[both], b[both], rtol=1e-9, atol=1e-12), c


def test_stock_predictor_never_picks_a_market_or_v2_bundle():
    """Regression (2026-11-06): the newest bundle folder was the E19 market model, which broke the app's predictor."""
    from nexus_alpha.inference.pipeline import latest_bundle
    b = latest_bundle()
    assert b is not None and "_e19_" not in b.name and "_e14_" not in b.name
