"""E26 statistics on synthetic data: the incremental test finds information that is there, and none that is not."""
import numpy as np
import pandas as pd

from nexus_alpha.evaluation import iv_benchmark as IB


def _rows(n_dates=160, per_date=60, nexus_informative=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        for _ in range(per_date):
            iv = rng.uniform(0.15, 0.6)
            extra = rng.normal(0, 1)                      # information only NEXUS has (if informative)
            s20 = iv * np.sqrt(20 / 252) * np.exp(0.35 * extra if nexus_informative else 0)
            r = rng.normal(0, s20)
            tau = 0.5 * iv / np.sqrt(252) * np.sqrt(20)
            p_side = 1 / (1 + np.exp(1.2 * (extra if nexus_informative else rng.normal(0, 1)) + 0.4))
            rows.append({"asof_session": f"d{d:04d}", "p_side": p_side, "iv30": iv, "real_tau": tau,
                         "real_y": 2 if r > tau else (0 if r < -tau else 1)})
    return pd.DataFrame(rows)


def test_incremental_information_is_detected():
    out = IB.evaluate(_rows(nexus_informative=True), 20)
    assert out["sessions"] == 160 and out["endpoint2_ll_ivonly_minus_both"]["ci95"][0] > 0
    assert out["decision"] == "NEXUS adds to implied volatility"


def test_noise_is_not_mistaken_for_information():
    out = IB.evaluate(_rows(nexus_informative=False, seed=1), 20)
    assert out["endpoint2_ll_ivonly_minus_both"]["ci95"][0] <= 0
    assert out["decision"] == "NEXUS's volatility skill is already priced"


def test_holm_step_down():
    assert IB.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2}) == {"a": True, "b": False, "c": False, "d": False}
    assert IB.holm({"a": 0.001, "b": 0.012, "c": 0.02, "d": 0.04}) == {"a": True, "b": True, "c": True, "d": True}
    assert IB.holm({"a": 0.02, "b": 0.03}) == {"a": True, "b": True}
    assert IB.holm({"a": 0.03, "b": 0.03}) == {"a": False, "b": False}


def test_too_few_sessions_never_decides():
    out = IB.evaluate(_rows(n_dates=30, nexus_informative=True), 20)
    assert out["decision"].startswith("insufficient")
    assert IB.evaluate(_rows(n_dates=1), 20)["decision"].startswith("insufficient")
