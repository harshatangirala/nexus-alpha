"""The E11/E16 engine keeps an informative feature block and rejects a pure-noise block (synthetic data)."""
import numpy as np
import pandas as pd

from nexus_alpha.evaluation import ladder as L


def _frame(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"asof_date": pd.to_datetime("2005-01-03") + pd.to_timedelta(rng.integers(0, 6000, n), unit="D"),
                      "x0": rng.normal(size=n), "x1": rng.normal(size=n), "noise": rng.normal(size=n)})
    yr = d.asof_date.dt.year
    d["block"] = np.select([yr <= 2015, yr == 2016, yr <= 2018], ["train", "inner", "valid"], "test")
    logit = np.stack([-d.x1, np.zeros(n), d.x1], 1)
    p = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
    d["y_1D"] = (p.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    return d


def test_keeps_signal_rejects_noise(tmp_path):
    d = _frame()
    sets = {"A": ["x0"], "B": ["x0", "x1"], "C": ["x0", "x1", "noise"]}
    res, rows = L.fit_eval(d, sets, ["1D"], 2, tmp_path, log=lambda s: None)
    g = L.group_gains(rows, [("B", "A"), ("C", "B")], ["1D"], n_boot=200)
    for m in ("lgbm", "logistic"):
        assert L.keep_decision(g, m, "B", "A")
        assert not L.keep_decision(g, m, "C", "B")
