"""The nightly runner (Task Scheduler -> run_daily.cmd) must be byte-clean and reference only existing scripts.

Audit 4 (2026-11-07): a patch wrote "\\v" (a vertical tab) into a script path in run_daily.cmd; cmd.exe would have failed
that step silently every night."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CMD = REPO / "scripts" / "shadow" / "run_daily.cmd"


def test_runner_bytes_are_clean_crlf():
    b = CMD.read_bytes()
    assert not [c for c in set(b) if c < 32 and c not in (10, 13)], "control characters in run_daily.cmd"
    assert b.count(b"\n") == b.count(b"\r\n"), "run_daily.cmd must use CRLF throughout"


def test_runner_scripts_exist():
    text = CMD.read_text(encoding="ascii")
    refs = re.findall(r"scripts\\[\w\\]+\.py", text)
    assert refs, "no script references found"
    for r in refs:
        assert (REPO / Path(*r.split("\\"))).exists(), f"run_daily.cmd references a missing script: {r}"


def test_every_logged_version_is_frozen_and_monitored():
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "shadow"))
    import spc_monitor
    text = CMD.read_text(encoding="ascii")
    scored = set(re.findall(r"shadow_score\.py --version ([\w.]+)", text)) | {"v1.1", "v2"}
    for v in scored:
        assert (REPO / "registry" / "frozen_models" / f"{v}.json").exists(), v
        assert v in spc_monitor.UNIVERSE, f"{v} is scored nightly but not in the SPC monitor"


def test_expected_asof_is_the_last_closed_session():
    import pandas as pd
    from nexus_alpha import shadow as SH
    et = lambda x: pd.Timestamp(x, tz="America/New_York")
    assert SH.expected_asof(et("2026-11-06 20:30")) == pd.Timestamp("2026-11-06")      # Friday evening, after the close
    assert SH.expected_asof(et("2026-11-06 15:00")) == pd.Timestamp("2026-11-05")      # Friday, before the close
    assert SH.expected_asof(et("2026-11-07 03:00")) == pd.Timestamp("2026-11-06")      # Saturday
    assert SH.expected_asof(et("2026-11-09 08:00")) == pd.Timestamp("2026-11-06")      # Monday before the open
    assert SH.expected_asof(et("2026-11-09 17:00")) == pd.Timestamp("2026-11-09")


def test_stale_bar_cache_is_refetched(tmp_path, monkeypatch):
    """Audit 4 (2026-11-07): a cache filled before the last close must not be served to a later run."""
    import pandas as pd
    from nexus_alpha import shadow as SH
    from nexus_alpha.inference import pipeline as PL
    want = SH.expected_asof()
    mk = lambda last: pd.DataFrame({"date": pd.to_datetime([last]), "open": [1.0], "high": [1.0], "low": [1.0],
                                    "close": [1.0], "volume": [1.0]})
    calls = []

    def fake(ticker):
        calls.append(ticker)
        return mk(want), pd.DataFrame()

    monkeypatch.setattr(PL, "_fetch_live_uncached", fake)
    monkeypatch.setenv("NEXUS_LIVE_CACHE", str(tmp_path))
    mk(want - pd.Timedelta(days=4)).to_pickle(tmp_path / "ZZZ_daily.pkl")           # stale
    pd.DataFrame().to_pickle(tmp_path / "ZZZ_hourly.pkl")
    d, _ = PL.fetch_live("ZZZ")
    assert calls == ["ZZZ"] and pd.Timestamp(d.date.max()) == want                   # refetched
    d, _ = PL.fetch_live("ZZZ")
    assert calls == ["ZZZ"]                                                          # the fresh cache is then served
