"""Parity of vendored source-project code with the stored artifacts it must reproduce (needs the local source data)."""
import numpy as np
import pandas as pd
import pytest

from nexus_alpha import paths
from nexus_alpha.data import bars as B
from nexus_alpha.indicators import triframe_visible as FV

from .conftest import source_data

FRAME = paths.ARTIFACTS / "samples" / "s500_frame.parquet"


@source_data
@pytest.mark.skipif(not FRAME.exists(), reason="sample frame not built")
def test_vendored_visible_features_match_frozen_triframe_values():
    fr = pd.read_parquet(FRAME, columns=["sample_id", "ticker", "asof_date"] + ["x_" + f for f in FV.DEPLOY])
    pick = fr.sample(40, random_state=3)
    worst = 0.0
    for _, r in pick.iterrows():
        d = B.read_daily(r.ticker)
        k = int(d.date.searchsorted(r.asof_date))
        got = FV.from_series(d.open, d.high, d.low, d.close, d.volume, [k]).iloc[0]
        for f in FV.DEPLOY:
            a, b = got[f"x_{f}"], r[f"x_{f}"]
            if np.isnan(a) and np.isnan(b):
                continue
            worst = max(worst, abs(a - b) / max(1.0, abs(b)))
    assert worst < 1e-5, worst          # stored as float32


@source_data
def test_vendored_calendar_and_renderer_hashes_recorded():
    from nexus_alpha.cv import render
    assert render.TRIPLET_BYTES == 17760 and render.VERSION == "r1"
