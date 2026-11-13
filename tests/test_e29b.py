"""E29b: the news-on archive filter (point in time) and the scorer's start condition and endpoint."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    sys.path.insert(0, str(REPO / "scripts" / "shadow"))
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "shadow" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_archive_keeps_only_headlines_fetched_before_the_decision(tmp_path):
    N = _load("news_on_forecast")
    dec = pd.Timestamp("2026-11-20 14:00", tz="UTC")
    pd.DataFrame([
        {"fetched_utc": "2026-11-19T01:00:00+00:00", "ticker": "AAA", "published_utc": "2026-11-18T20:00:00+00:00",
         "headline": "old seen early", "text": "", "link": "l1", "source": "s"},
        {"fetched_utc": "2026-11-20T15:00:00+00:00", "ticker": "AAA", "published_utc": "2026-11-19T20:00:00+00:00",
         "headline": "seen after the decision", "text": "", "link": "l2", "source": "s"},
        {"fetched_utc": "2026-11-20T01:00:00+00:00", "ticker": "AAA", "published_utc": "2026-11-18T20:00:00+00:00",
         "headline": "old seen early", "text": "", "link": "l1", "source": "s"},          # duplicate, later sighting
        {"fetched_utc": "2026-11-20T01:00:00+00:00", "ticker": "AAA", "published_utc": "2026-11-10T20:00:00+00:00",
         "headline": "outside the 5-day window", "text": "", "link": "l3", "source": "s"},
    ]).to_parquet(tmp_path / "headlines_2026-11-20.parquet")
    a = N.archive(dec, tmp_path)
    assert set(a.headline) == {"old seen early", "outside the 5-day window"}
    assert a[a.headline == "old seen early"].fetched_utc.iloc[0] == pd.Timestamp("2026-11-19 01:00", tz="UTC")
    arts = N.articles_for(a, "AAA", dec)
    assert [x["headline"] for x in arts] == ["old seen early"]
    assert N.articles_for(a, "BBB", dec) == []


def test_start_condition_needs_ten_clean_nights_and_an_on_time_log(tmp_path):
    E = _load("e29_score")
    for d in range(1, 11):
        (tmp_path / f"headlines_2026-11-{d:02d}.summary.json").write_text(
            json.dumps({"tickers": 500, "headlines": 9000 if d != 3 else 0, "failed": 0}), encoding="utf-8")
    chain = [{"run_utc": "2026-11-05T02:00:00+00:00", "late_rows": "0"}]
    assert E.e29b_start(tmp_path, chain) is None                    # night 3 was not clean: only 9 clean nights
    (tmp_path / "headlines_2026-11-11.summary.json").write_text(
        json.dumps({"tickers": 500, "headlines": 8000, "failed": 3}), encoding="utf-8")
    t = E.e29b_start(tmp_path, chain)
    assert t == pd.Timestamp("2026-11-12", tz="UTC")                # the 10th clean night is 11-11 (+1 day)
    assert E.e29b_start(tmp_path, [{"run_utc": "2026-11-05T02:00:00+00:00", "late_rows": "4"}]) is None


def test_news_gain_is_log_loss_without_minus_with():
    E = _load("e29_score")
    scored = pd.DataFrame({"asof_session": ["2026-12-01"] * 2, "ticker": ["A", "B"], "horizon": ["5D", "5D"],
                           "p_down": [0.3, 0.3], "p_side": [0.4, 0.4], "p_up": [0.3, 0.3], "real_y": [2, 0],
                           "late": [False, False]})
    news = pd.DataFrame({"asof_session": ["2026-12-01"], "ticker": ["A"], "horizon": ["5D"], "p_down": [0.2],
                         "p_side": [0.3], "p_up": [0.5], "late": [False], "run_utc": ["2026-12-02T02:00:00+00:00"]})
    g = E.news_gains(scored, news, pd.Timestamp("2026-11-30", tz="UTC"))
    assert len(g) == 1 and g.rows.iloc[0] == 1
    assert np.isclose(g.gain.iloc[0], -np.log(0.3) + np.log(0.5))
    assert E.news_gains(scored, news, pd.Timestamp("2026-12-03", tz="UTC")).empty     # logged before the start
    assert E.news_gains(scored, news, None).empty
