"""E28 language branch (D-053): filing text cleaning, the SEC price-impact expert's point-in-time evidence, the
dataset blocks and the placebo. Synthetic data only; nothing here reads the SEC study or the sealed period."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nexus_alpha.multimodal.dataset import NLP_BLOCKS, build
from nexus_alpha.multimodal.trainer import ModalityData
from nexus_alpha.nlp import sec_impact as SI
from nexus_alpha.nlp import transformer_text as TT
from nexus_alpha.seal import SEAL_UTC

REPO = Path(__file__).resolve().parents[1]


def _e28():
    spec = importlib.util.spec_from_file_location("e28", REPO / "scripts" / "experiments" / "e28_vision_nlp_fusion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e28"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cleaning_drops_safe_harbour_paragraphs_and_is_deterministic():
    text = ("Item 2.02 Results of Operations.\n\nRevenue rose 12% to $4.1 billion.\n\n"
            "This report contains forward-looking statements within the meaning of the Private Securities Litigation "
            "Reform Act of 1995.\n\nThe company undertakes no obligation to update them.")
    out = TT.clean_filing_text(text)
    assert out == "Item 2.02 Results of Operations. Revenue rose 12% to $4.1 billion."
    assert TT.clean_filing_text(text) == out and TT.clean_filing_text(None) == "" and TT.clean_filing_text("") == ""
    assert len(TT.clean_filing_text("word " * 5000)) == 4000


def test_encoder_is_the_chronologically_consistent_one_pinned_by_revision():
    assert TT.MODEL == "manelalab/chrono-bert-v1-20101231"       # pretrained only on text dated before 2011
    assert len(TT.REVISION) == 40 and TT.MAX_TOKENS == 256


def test_expert_features_leave_out_year_and_accounting_families():
    names = ["px_ret_1", "meta_year", "meta_month", "acc_roa", "sue_last", "ear_last", "_avail_accounting", "_avail_ear",
             "lm_negative", "event_id", "_avail_price", "_avail_lexical"]
    # _avail_* are availability timestamps: as numbers they are calendar time, a meta_year proxy (D-054)
    assert SI.feature_columns(names) == ["px_ret_1", "meta_month", "lm_negative"]


def test_expert_reads_no_label_near_the_seal():
    assert SI.LABEL_READ_BEFORE.tz_localize("UTC") < SEAL_UTC
    assert SI.FIT_BEFORE < SI.ES_BEFORE <= pd.Timestamp("2017-01-01", tz="UTC")      # never fits on validation years


def _events():
    tab = pd.DataFrame({"ticker": ["AAA"] * 3 + ["BBB"],
                        "event_type": ["8K", "periodic_10Q", "8K", "8K"],
                        "available_utc": pd.to_datetime(["2017-01-02 22:00", "2017-01-20 21:00", "2017-03-01 21:00",
                                                         "2017-01-25 21:00"], utc=True)},
                       index=pd.Index(["e1", "e2", "e3", "e4"], name="event_id"))
    pred = pd.DataFrame({f"p{c}_h{h}": v for h in (1, 5) for c, v in
                         (("down", [.2, .1, .3, .3]), ("side", [.6, .2, .4, .4]), ("up", [.2, .7, .3, .3]))},
                        index=tab.index)
    return tab, pred


def test_sample_evidence_is_point_in_time():
    tab, pred = _events()
    s = pd.DataFrame({"sample_id": ["s_before", "s_at", "s_mid", "s_stale", "s_other"],
                      "ticker": ["AAA", "AAA", "AAA", "AAA", "CCC"],
                      "decision_utc": pd.to_datetime(["2017-01-02 13:00", "2017-01-20 21:00", "2017-01-25 14:00",
                                                      "2017-02-20 14:00", "2017-01-25 14:00"], utc=True)})
    f = SI.sample_features(s, tab, pred)
    assert f.loc["s_before", "si_present"] == 0                 # the filing became public after the decision time
    assert f.loc["s_at", "si_present"] == 1 and f.loc["s_at", "si_is_periodic"] == 1   # available exactly at decision
    assert np.isclose(f.loc["s_at", "si_pmove_h1"], 0.8) and np.isclose(f.loc["s_at", "si_dir_h1"], 0.6)
    assert f.loc["s_mid", "si_n_events_30d"] == 2 and np.isclose(f.loc["s_mid", "si_max_pmove_h5_30d"], 0.8)
    assert f.loc["s_stale", "si_present"] == 0                  # latest filing older than 14 days
    assert f.loc["s_other", "si_present"] == 0                  # no events for the ticker
    assert np.isnan(f.loc["s_before", "si_pmove_h1"])           # absent is missing, never zero


def test_language_blocks_are_zeroed_and_flagged_when_absent_and_separate_from_classical_sec():
    n = 8
    fr = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)],
                       "sec_covered": [1, 1, 1, 1, 0, 0, 1, 1], "sec_n_8k_7d": np.arange(n, dtype=float),
                       "sec_tx_8k_pc0": [.1, .2, np.nan, .4, np.nan, np.nan, .7, .8],
                       "sec_tx_8k_present": [1, 1, 0, 1, 0, 0, 1, 1], "sec_tx_present": [1, 1, 0, 1, 0, 0, 1, 1],
                       "fb_tone_mean": [.5, np.nan, .1, .2, np.nan, .3, .4, .1], "fb_n": [2, 0, 1, 1, 0, 3, 1, 1],
                       "si_pmove_h1": [.6] * n, "si_present": [1, 0, 1, 0, 1, 0, 1, 0]})
    for h in ("1D", "2D", "5D", "10D", "20D", "1M"):
        fr[f"y_{h}"] = 1
        fr[f"z_{h}"] = 0.0
    d, pre = build(fr, np.ones(n, bool), ("sec",) + tuple(NLP_BLOCKS))
    assert pre.sec_cols == ["sec_n_8k_7d"]                       # ChronoBERT columns never leak into the classical block
    assert pre.nlp_cols["sec_tx"] == ["sec_tx_8k_pc0", "sec_tx_8k_present"]
    assert pre.nlp_cols["news_tx"] == ["fb_tone_mean"] and pre.nlp_cols["sec_impact"] == ["si_pmove_h1"]
    j = {m: d.names.index(m) for m in d.names}
    assert d.mask[:, j["sec_tx"]].tolist() == [1, 1, 0, 1, 0, 0, 1, 1]
    assert d.mask[:, j["news_tx"]].tolist() == [1, 0, 1, 1, 0, 1, 1, 1]
    for m in ("sec_tx", "news_tx", "sec_impact"):
        assert np.all(d.x[m][d.mask[:, j[m]] == 0] == 0)


def test_a_timestamp_can_never_enter_the_fusion_as_a_number():
    import pytest
    from nexus_alpha.multimodal.dataset import check_feature_types
    fr = pd.DataFrame({"sec_n_8k_7d": [1.0, 2.0], "sec_ok_flag": [True, False],
                       "sec_avail": pd.to_datetime(["2015-01-02", "2016-01-04"], utc=True),
                       "sec_age": pd.to_timedelta([1, 2], unit="D"), "sec_kind": ["8K", "10Q"]})
    check_feature_types(fr, ["sec_n_8k_7d", "sec_ok_flag"])
    for c in ("sec_avail", "sec_age", "sec_kind"):
        with pytest.raises(TypeError):
            check_feature_types(fr, ["sec_n_8k_7d", c])
    fr2 = fr.drop(columns=["sec_kind"])
    fr2["y_1D"] = 1
    with pytest.raises(TypeError):                               # build() runs the guard when it fits
        build(fr2.assign(**{f"{p}_{h}": v for h in ("1D", "2D", "5D", "10D", "20D", "1M") for p, v in (("y", 1), ("z", 0.0))},
                         sample_id=["a", "b"], sec_covered=1), np.ones(2, bool), ("sec",))


def test_placebo_moves_values_and_flags_together_within_each_date():
    e28 = _e28()
    n = 12
    dates = np.repeat(["2017-01-03", "2017-01-04", "2017-01-05"], 4)
    x = {"num": np.arange(n, dtype=np.float32)[:, None], "sec_tx": (np.arange(n, dtype=np.float32)[:, None] + 1) * 10}
    mask = np.ones((n, 2), np.float32)
    mask[[1, 5, 9], 1] = 0
    x["sec_tx"][[1, 5, 9]] = 0
    d = ModalityData(x, mask, np.zeros((n, 6), np.int64), np.zeros((n, 6), np.float32), ["num", "sec_tx"])
    p = e28.permute_within_date(d, ("sec_tx",), dates, seed=11)
    assert np.array_equal(p.x["num"], d.x["num"]) and np.array_equal(p.mask[:, 0], d.mask[:, 0])
    assert not np.array_equal(p.x["sec_tx"], d.x["sec_tx"])
    for day in np.unique(dates):
        r = dates == day
        assert sorted(p.x["sec_tx"][r, 0]) == sorted(d.x["sec_tx"][r, 0])
    assert np.all((p.x["sec_tx"][:, 0] == 0) == (p.mask[:, 1] == 0))     # a moved absent row stays absent


def test_e28_configurations_match_the_preregistration():
    e28 = _e28()
    assert e28.CONFIGS["BASE"] == ("img_W", "img_D", "img_H", "num", "news", "sec")
    assert e28.CONFIGS["MM"] == e28.CONFIGS["BASE"] + ("news_tx", "sec_tx", "sec_impact")
    assert e28.SEEDS == (7, 11, 23) and e28.PLACEBO_SEED == 11
    assert e28.TRAIN == dict(epochs=30, batch=1024, lr=3e-4, weight_decay=1e-4, patience=4, schedule="cosine")
    assert e28.FCFG["variant"] == "gated" and e28.FCFG["dropout"] == 0.3 and e28.FCFG["modality_dropout"] == 0.15
    assert len(e28.prereg_sha256()) == 64


def test_dashboard_language_view_is_point_in_time(monkeypatch):
    from nexus_alpha import language_view as LV
    meta = pd.DataFrame({"event_id": ["a", "b", "c", "d"], "ticker": ["AAA", "BBB", "AAA", "CCC"],
                         "event_type": ["8K", "8K", "periodic_10Q", "8K"],
                         "available_utc": pd.to_datetime(["2015-01-05", "2016-03-01", "2017-02-01", "2018-06-01"], utc=True)})
    E = np.array([[1, 0], [0.9, 0.1], [0, 1], [1, 0.05]], np.float32)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    monkeypatch.setattr(LV, "_vectors", lambda: (meta, E, np.ones(4, bool)))
    preds = meta.set_index("event_id").assign(p_move_h1=[.5, .6, .7, .8], p_move_h5=.5, lean_h1=0.0, lean_h5=0.0)
    monkeypatch.setattr(LV, "event_predictions", lambda: preds.reset_index().set_index("event_id"))
    sim = LV.similar_filings("d", k=3, one_per_firm=False)
    assert list(sim.event_id) == ["a", "b", "c"] and (sim.available_utc < meta.available_utc[3]).all()
    assert list(LV.similar_filings("d", k=3).event_id) == ["a", "b"]    # default: best filing per firm (c is AAA too)
    assert LV.similar_filings("a", k=3).empty                           # nothing was public before the first filing
    assert "a" not in set(LV.similar_filings("c", k=3).event_id)        # other firms only
    monkeypatch.setattr(LV, "event_predictions", lambda: preds.reset_index())
    tl = LV.filing_timeline("aaa", "2016-12-31")
    assert list(tl.kind) == ["8-K"]                                     # the 2017 10-Q is after the as-of date


def test_analogue_search_removes_the_common_direction(monkeypatch):
    from nexus_alpha import language_view as LV
    mu = np.array([10.0, 10.0, 0.0], np.float32)                 # the shared direction of mean-pooled vectors
    meta = pd.DataFrame({"event_id": ["x", "y"], "ticker": ["AAA", "BBB"], "event_type": ["8K", "8K"],
                         "available_utc": pd.to_datetime(["2015-01-05", "2016-03-01"], utc=True)})
    E = np.array([[1, 0, 0], [0, 0, 1]], np.float32)             # already centred and normalised, as _vectors returns
    monkeypatch.setattr(LV, "_centre", lambda: mu)
    monkeypatch.setattr(LV, "_vectors", lambda: (meta, E, np.ones(2, bool)))
    monkeypatch.setattr(LV, "event_predictions", lambda: None)
    monkeypatch.setattr(LV, "_labels", lambda: None)
    out, s = LV.analogues(mu + np.array([0, 0, 0.1], np.float32), k=1)   # raw cosine would favour "x"
    assert list(out.event_id) == ["y"] and np.isclose(s["mean_cosine"], 1.0)


def test_the_documented_e28_run_is_pinned_identically_and_its_preregistration_is_unchanged():
    import json
    import re
    pins = {f: re.findall(r'(?:E28_RUN = |"e28": )"([^"]+)"', (REPO / f).read_text(encoding="utf-8"))
            for f in ("src/nexus_alpha/language_view.py", "scripts/colab/nb_sections_nlp.py",
                      "scripts/colab/stage_notebook_inputs.py")}
    runs = {r for v in pins.values() for r in v}
    assert len(runs) == 1 and all(len(v) == 1 for v in pins.values()), pins
    run = runs.pop()
    d = REPO / "experiments" / run
    assert run != "PENDING" and (d / "metrics.json").exists()
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    assert m["decision"] in ("KEPT", "NOT KEPT")
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "COMPLETED" and not man["config"]["quick"]
    # the pre-registered E28 section must still be the text the run was made under (results go in their own section)
    assert man["config"]["prereg_sha256"] == _e28().prereg_sha256()


def test_dashboard_tables_from_a_metrics_record():
    from nexus_alpha import language_view as LV
    m = {"results": {"BASE": {"modalities": ["num"], "valid_mean_log_loss": 1.06, "test_mean_log_loss": 1.07},
                     "MM": {"modalities": ["num", "sec_tx"], "valid_mean_log_loss": 1.059, "test_mean_log_loss": 1.071}},
         "gates": {"MM": {"names": ["num", "sec_tx"], "valid": [0.7, 0.3], "test": [0.6, 0.4]}}}
    ct = LV.config_table(m).set_index("config")
    assert np.isclose(ct.loc["MM", "valid gain vs BASE"], 0.001) and np.isclose(ct.loc["MM", "test gain vs BASE"], -0.001)
    gt = LV.gate_table(m)
    assert list(gt.group) == ["indicators", "language"] and np.isclose(gt.gate.sum(), 1.0)
