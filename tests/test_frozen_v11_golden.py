"""The frozen v1.1 model must keep producing exactly its recorded outputs while v2 is built on shared code.

Fixture recorded 2026-11-06 right after the freeze (historical mode, sealed bars, so inputs are deterministic). If this
test fails, a shared module changed v1.1 behaviour: revert the change or move it into a v2-only path.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from nexus_alpha import paths

FIX = Path(__file__).parent / "fixtures" / "v11_golden.json"


def _ready() -> bool:
    if not FIX.exists():
        return False
    bid = json.loads(FIX.read_text())["bundle_id"]
    return (paths.ARTIFACTS / "bundles" / bid).exists() and (paths.OHLCV_FINAL / "1day").exists()


@pytest.mark.skipif(not _ready(), reason="frozen bundle or sealed bars not available on this machine")
def test_v11_outputs_unchanged():
    warnings.filterwarnings("ignore")
    from nexus_alpha import shadow as SH
    from nexus_alpha.inference.pipeline import NexusPredictor
    g = json.loads(FIX.read_text())
    bdir, rec = SH.verify_frozen("v1.1")          # also proves the weights are the frozen ones
    assert rec["bundle_id"] == g["bundle_id"]
    P = NexusPredictor(bdir)
    for c in g["cases"]:
        r = P.predict(c["ticker"], asof=c["asof"], mode="historical", explain=False)
        for h, want in c["forecasts"].items():
            f = r.forecasts[h]
            got = [f.down_probability, f.sideways_probability, f.up_probability, f.return_q10, f.expected_return,
                   f.return_q90]
            np.testing.assert_allclose(np.array(got, float), np.array(want, float), rtol=0, atol=1e-7,
                                       err_msg=f"{c['ticker']} {c['asof']} {h}")
