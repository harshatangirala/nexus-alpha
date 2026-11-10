"""Forward promotion criteria (D-042, D-049, D-051) on synthetic forward logs."""
import numpy as np
import pandas as pd

from nexus_alpha.evaluation import forward_criteria as FC


def _logs(n_sessions=150, n_stocks=60, cand_skill=(1.0, 0.0), ranker_skill=0.03, seed=0):
    rng = np.random.default_rng(seed)
    cands = {f"c{i}": [] for i in range(len(cand_skill))}
    ranker = []
    for d in range(n_sessions):
        sig = rng.normal(0, 1, n_stocks)
        ret = 0.02 * rng.normal(0, 1, n_stocks) + 0.02 * 0.3 * sig
        for i, s in enumerate(cand_skill):
            score = s * sig + rng.normal(0, 1, n_stocks)
            cands[f"c{i}"] += [{"asof_session": f"d{d:04d}", "ticker": f"t{j}", "rank_score": score[j], "real_ret": ret[j]}
                               for j in range(n_stocks)]
        rs = ranker_skill * sig + rng.normal(0, 1, n_stocks)
        ranker += [{"asof_session": f"d{d:04d}", "ticker": f"t{j}", "rank_score": rs[j], "real_ret": ret[j]}
                   for j in range(n_stocks)]
    return {k: pd.DataFrame(v) for k, v in cands.items()}, pd.DataFrame(ranker)


def test_a_skilled_candidate_is_promoted_and_a_null_one_is_not():
    cands, ranker = _logs()
    out = FC.evaluate(cands, ranker)
    assert out["c0"]["decision"].startswith("promote") and out["c0"]["p_ic_above_0"] < 0.01
    assert out["c1"]["decision"].startswith("not promoted")


def test_a_weak_signal_is_not_promoted_on_150_sessions():
    """A daily rank IC near 0.03 (the size seen on validation) is too small to clear the bar in 150 sessions."""
    cands, ranker = _logs(cand_skill=(0.15, 0.0))
    out = FC.evaluate(cands, ranker)
    assert not out["c0"]["decision"].startswith("promote")


def test_nothing_is_decided_before_120_sessions():
    cands, ranker = _logs(n_sessions=60)
    out = FC.evaluate(cands, ranker)
    assert all(v["decision"].startswith("insufficient") for v in out.values())


def test_block_p_value_is_one_sided():
    rng = np.random.default_rng(3)
    assert FC.block_p_value(pd.Series(rng.normal(0.05, 0.1, 200))) < 0.01
    assert FC.block_p_value(pd.Series(rng.normal(-0.05, 0.1, 200))) > 0.99
