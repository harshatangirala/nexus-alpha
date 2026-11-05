import hashlib

import pandas as pd
import pytest

from nexus_alpha import seal
from nexus_alpha.data.calendar import calendar, decision_time_utc, offset_session, session_index, sessions

from .conftest import source_data


def test_dev_mask_dates_and_label_ends():
    d = pd.to_datetime(["2022-12-01", "2022-12-30", "2023-01-03", "2024-05-01"])
    assert seal.dev_mask(d).tolist() == [True, True, False, False]
    ends = pd.to_datetime(["2022-12-30", "2023-01-31", "2023-02-01", pd.NaT])
    assert seal.dev_mask(d, ends).tolist() == [True, False, False, False]


def test_assert_dev_raises():
    with pytest.raises(seal.HoldoutSealed):
        seal.assert_dev(pd.to_datetime(["2021-01-04", "2023-01-03"]))


def test_tz_aware_dates_are_compared_in_new_york():
    # 2023-01-03 04:00 UTC is still 2023-01-02 23:00 in New York (a holiday, before the first sealed session)
    d = pd.Series(pd.to_datetime(["2023-01-03 04:00", "2023-01-03 15:00"], utc=True))
    assert seal.dev_mask(d).tolist() == [True, False]


def test_unlock_requires_owner_config_and_is_single_use(tmp_governance, tmp_path, monkeypatch):
    with pytest.raises(seal.HoldoutSealed):
        seal.unlock("test")
    cfg = tmp_path / "frozen.json"
    cfg.write_text('{"frozen": true}')
    monkeypatch.setenv("NEXUS_HOLDOUT_CONFIG", str(cfg))
    monkeypatch.setenv("NEXUS_HOLDOUT_UNLOCK", "0" * 64)
    with pytest.raises(seal.HoldoutSealed):
        seal.unlock("wrong key")
    monkeypatch.setenv("NEXUS_HOLDOUT_UNLOCK", hashlib.sha256(cfg.read_bytes()).hexdigest())
    assert seal.unlock("owner", 10) == hashlib.sha256(cfg.read_bytes()).hexdigest()
    with pytest.raises(seal.HoldoutSealed):
        seal.unlock("second use")
    assert seal.verify_log()
    monkeypatch.setenv("NEXUS_HOLDOUT_REUSE_REASON", "documented re-run")
    seal.unlock("logged reuse")
    assert seal.verify_log()


def test_access_log_tamper_is_detected(tmp_governance, tmp_path, monkeypatch):
    cfg = tmp_path / "frozen.json"
    cfg.write_text("{}")
    monkeypatch.setenv("NEXUS_HOLDOUT_CONFIG", str(cfg))
    monkeypatch.setenv("NEXUS_HOLDOUT_UNLOCK", hashlib.sha256(cfg.read_bytes()).hexdigest())
    seal.unlock("owner", 1)
    p = seal._log_path()
    p.write_text(p.read_text().replace(",1,", ",999,"))
    assert not seal.verify_log()


def test_calendar_basics():
    c = calendar()
    assert c.date.is_monotonic_increasing and not c.date.duplicated().any()
    s = sessions()
    assert pd.Timestamp("2012-10-29") not in s and pd.Timestamp("2012-10-30") not in s   # Hurricane Sandy
    assert pd.Timestamp("2022-12-30") in s and pd.Timestamp("2023-01-02") not in s
    assert s.max() >= pd.Timestamp("2027-12-30")
    assert bool(c.set_index("date").early_close.loc["2019-11-29"])                    # day after Thanksgiving


def test_offset_and_decision_time():
    e = offset_session(pd.to_datetime(["2022-12-29"]), 1)[0]
    assert e == pd.Timestamp("2022-12-30")
    e2 = offset_session(pd.to_datetime(["2022-12-30"]), 1)[0]
    assert e2 == pd.Timestamp("2023-01-03")                                             # skips the holiday
    dt = decision_time_utc(pd.to_datetime(["2022-03-11"]))[0]                           # Friday -> Monday 09:00 NY
    assert dt == pd.Timestamp("2022-03-14 09:00", tz="America/New_York").tz_convert("UTC")
    with pytest.raises(ValueError):
        offset_session(pd.to_datetime(["2022-12-25"]), 1)
    assert session_index(pd.to_datetime(["2022-12-25"]))[0] == -1


@source_data
def test_vendored_calendar_matches_source():
    from nexus_alpha import paths
    from nexus_alpha.data import calendar as C
    src = paths.MARKET_CALENDAR_CSV.read_bytes()
    vend = (C.CALENDAR_CSV.parent / "xnys_sessions.csv").read_bytes()
    assert hashlib.sha256(src).hexdigest() == hashlib.sha256(vend).hexdigest()
