"""Texts typed on the desks carry a date, a time and a time zone; these decide whether a model may read them.

Covers the conversion to UTC (India has no daylight saving; New York does), the as-of close, the availability rule
(minute vs date-only), the chronology statuses, and A-82: pasted company news obeys the same 5-day window as training.
"""
import datetime as dt

import pandas as pd
import pytest

from nexus_alpha import text_time as TX


def test_ist_and_new_york_times_convert_to_utc():
    assert TX.to_utc("2026-11-05", "21:30", "IST (India, UTC+5:30)") == pd.Timestamp("2026-11-05 16:00", tz="UTC")
    # New York: UTC-4 in summer (EDT), UTC-5 in winter (EST)
    assert TX.to_utc("2021-06-15", dt.time(8, 30), "ET (New York)") == pd.Timestamp("2021-06-15 12:30", tz="UTC")
    assert TX.to_utc("2021-12-15", dt.time(8, 30), "ET (New York)") == pd.Timestamp("2021-12-15 13:30", tz="UTC")
    assert TX.to_utc("2021-12-15", "08:30", "UTC") == pd.Timestamp("2021-12-15 08:30", tz="UTC")
    assert TX.to_utc(None) is None
    # the repeated hour when New York clocks go back (2021-11-07 01:30 happens twice): no error, a valid time
    assert TX.to_utc("2021-11-07", "01:30", "ET (New York)") is not None


def test_date_only_text_counts_from_the_end_of_the_day():
    pub = TX.to_utc("2021-06-14", None, "IST (India, UTC+5:30)")
    assert pub == pd.Timestamp("2021-06-14", tz="UTC")
    assert TX.available_utc(pub, time_known=False) == pd.Timestamp("2021-06-15 08:00", tz="UTC")
    assert TX.available_utc(pd.Timestamp("2021-06-14 14:00", tz="UTC"), True) == pd.Timestamp("2021-06-14 14:15", tz="UTC")


def test_chronology_statuses():
    decision = pd.Timestamp("2021-06-16 13:00", tz="UTC")                 # 09:00 New York after the 15 June close
    items = [
        {"kind": "news", "headline": "beat", "published_utc": pd.Timestamp("2021-06-15 21:00", tz="UTC")},
        {"kind": "news", "headline": "late", "published_utc": pd.Timestamp("2021-06-16 13:30", tz="UTC")},
        {"kind": "news", "headline": "old", "published_utc": pd.Timestamp("2021-06-01 12:00", tz="UTC")},
        {"kind": "news", "headline": "undated", "published_utc": None},
        {"kind": "sec", "headline": "8-K", "published_utc": pd.Timestamp("2021-06-15 20:10", tz="UTC")},
        {"kind": "macro", "headline": "CPI", "published_utc": pd.Timestamp("2021-06-10 12:30", tz="UTC")},
    ]
    ch = TX.chronology(items, decision, "2021-06-15").set_index("headline")
    assert ch.loc["beat", "status"] == "model input (news branch)"
    assert ch.loc["beat", "phase"] == "after the close, before the decision"
    assert ch.loc["late", "status"].startswith("excluded: public only after")
    assert ch.loc["old", "status"].startswith("excluded: older than the 5-day")
    assert ch.loc["undated", "status"] == "excluded: no publication date"
    assert ch.loc["8-K", "status"].startswith("context") and ch.loc["8-K", "phase"] == "after the close, before the decision"
    assert ch.loc["CPI", "status"].startswith("context") and ch.loc["CPI", "phase"] == "before the as-of close"
    assert ch.loc["beat", "hours before decision"] == pytest.approx(15.75)


def test_asof_close_is_four_pm_new_york():
    assert TX.asof_close_utc("2021-06-15") == pd.Timestamp("2021-06-15 20:00", tz="UTC")
    assert TX.asof_close_utc("2021-12-15") == pd.Timestamp("2021-12-15 21:00", tz="UTC")


def test_pasted_news_obeys_the_training_window():
    # A-82: the news branch was trained on articles of the 5 days before the decision; older pasted ones are excluded
    from nexus_alpha.inference.pipeline import _articles_frame
    decision = pd.Timestamp("2021-06-16 13:00", tz="UTC")
    arts = [{"headline": "fresh", "published_utc": "2021-06-15 21:00"},
            {"headline": "stale", "published_utc": "2021-06-02 21:00"},
            {"headline": "future", "published_utc": "2021-06-16 14:00"},
            {"headline": "no time", "published_utc": "not a date"}]
    rows, listing = _articles_frame(arts, "MSFT", decision)
    used = {x["headline"]: x["used"] for x in listing}
    assert used == {"fresh": True, "stale": False, "future": False, "no time": False}
    assert list(rows.headline) == ["fresh"]
    why = {x["headline"]: x["reason"] for x in listing}
    assert "5-day news window" in why["stale"] and "after the decision time" in why["future"]
    assert listing[0]["hours_before_decision"] == pytest.approx(15.75)
