"""Kronos integration (E24, D-043): vendored code integrity, point-in-time context, label-rule probabilities, and exact
parity of the NEXUS sampler with the vendored predictor (deterministic decoding), its cache and its seeding."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nexus_alpha.kronos import engine as KE

VENDOR = Path(__file__).resolve().parents[1] / "src" / "nexus_alpha" / "kronos" / "vendor"


def test_vendored_code_matches_recorded_originals():
    notes = (VENDOR / "VENDORED.md").read_text(encoding="utf-8")
    want = dict(re.findall(r"`model/(\w+\.py)`: `([0-9a-f]{64})`", notes))
    assert hashlib.sha256((VENDOR / "module.py").read_bytes()).hexdigest() == want["module.py"]
    k = (VENDOR / "kronos.py").read_bytes()
    nl = b"\r\n" if b"\r\n" in k else b"\n"
    restored = k.replace(
        b"# NEXUS vendoring: package-relative import instead of sys.path.append(\"../\") + `from model.module import *`" + nl +
        b"from .module import *", b"sys.path.append(\"../\")" + nl + b"from model.module import *")
    assert hashlib.sha256(restored).hexdigest() == want["kronos.py"], "vendored kronos.py differs beyond the documented change"
    assert "MIT License" in (VENDOR / "LICENSE").read_text(encoding="utf-8")


def _bars(n=120, seed=0):
    from tests.conftest import synthetic_daily
    d = synthetic_daily(n=n, seed=seed)
    if "volume" not in d:
        d["volume"] = 1e6
    return d


def test_context_uses_bars_up_to_asof_only():
    d = _bars()
    k = 100
    a = KE.context(d, k, 64, 5)
    d2 = d.copy()
    d2.loc[k + 1:, ["open", "high", "low", "close", "volume"]] *= 3.0          # change the future
    b = KE.context(d2, k, 64, 5)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert KE.context(d, 10, 64, 5) is None                                   # not enough history


def test_label_probabilities_follow_the_v2_rule():
    # 1 series, 4 paths, horizon 2; open[t+1] = 100 on every path; sigma = 0.01
    paths = np.zeros((1, 4, 2, 6))
    paths[0, :, 0, 0] = 100.0
    paths[0, :, 1, 3] = [102.0, 100.2, 99.0, 103.0]                         # R_2 = +2%, +0.2%, -1%, +3%
    out = KE.label_probabilities(paths, np.array([0.01]), {"2D": 2}, alpha=0.5)["2D"]
    # tau = max(0.5 * 0.01 * sqrt(2), 0.001) = 0.00707: down 1, side 1, up 2
    assert np.allclose(out["p"][0], (np.array([1, 1, 2]) + 0.5) / (4 + 1.5))
    assert np.isclose(out["ret_mean"][0], np.mean([0.02, 0.002, -0.01, 0.03]))
    assert np.isclose(out["p"][0].sum(), 1.0)


def _weights_present():
    from nexus_alpha.kronos import weights as W
    try:
        W.verify("Kronos-mini")
        W.verify("Kronos-Tokenizer-2k")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _weights_present(), reason="Kronos weights not downloaded (scripts/kronos/fetch_weights.py)")
def test_sampler_parity_cache_seeding_with_vendored_predictor():
    import torch
    from nexus_alpha.data.calendar import offset_session
    from nexus_alpha.kronos import weights as W
    from nexus_alpha.kronos.vendor import KronosPredictor
    torch.set_num_threads(2)
    model, tok, mc = W.load("Kronos-mini")
    d = _bars(n=200, seed=3)
    k, L, H = 150, 64, 5
    c = KE.context(d, k, L, H)
    w = d.iloc[k + 1 - L: k + 1]
    x_ts = pd.Series(pd.to_datetime(w.date.to_numpy()))
    y_ts = pd.Series(pd.DatetimeIndex([offset_session(pd.DatetimeIndex([x_ts.iloc[-1]]), i)[0] for i in range(1, H + 1)]))
    orig = KronosPredictor(model, tok, device="cpu", max_context=mc).predict(
        w[["open", "high", "low", "close", "volume"]].astype(float).reset_index(drop=True), x_ts, y_ts, pred_len=H,
        T=1.0, top_p=1e-9, sample_count=1, verbose=False)[["open", "high", "low", "close"]].to_numpy()
    ref = KE.sample_paths(model, tok, [c], [1], 1, H, mc, top_p=1e-9, cache=False)[0, 0, :, :4]
    kv = KE.sample_paths(model, tok, [c], [1], 1, H, mc, top_p=1e-9, cache=True)[0, 0, :, :4]
    assert np.allclose(orig, ref, rtol=1e-6) and np.allclose(ref, kv, rtol=1e-6)
    a = KE.sample_paths(model, tok, [c], [7], 4, H, mc, cache=False)
    b = KE.sample_paths(model, tok, [c], [7], 4, H, mc, cache=True)
    assert np.allclose(a, b, rtol=1e-5)                                       # cache = reference under sampling
    assert np.array_equal(b, KE.sample_paths(model, tok, [c], [7], 4, H, mc))  # deterministic
    pair = KE.sample_paths(model, tok, [c, c], [99, 7], 4, H, mc)
    assert np.allclose(pair[1:2], b, rtol=1e-5)                               # independent of the batch
    assert not np.allclose(pair[0], pair[1])                                  # different seeds, different paths
