"""Live-news archive (audit 11, O4): point in time by its fetch stamp, first sighting kept, failures recorded, no network."""
import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("news_snapshot", REPO / "scripts" / "shadow" / "news_snapshot.py")
NS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NS)


def fake_fetch(t):
    if t == "BAD":
        raise TimeoutError("no answer")
    return [{"headline": f"{t} beats estimates", "text": "", "published_utc": "2026-11-13 21:05",
             "link": f"https://example.invalid/{t}", "source": "Yahoo Finance RSS"}]


def test_snapshot_records_failures_and_stamps_rows(tmp_path):
    t0 = pd.Timestamp("2026-11-13 22:00", tz="UTC")
    df, summ = NS.snapshot(["AAPL", "BAD", "MSFT"], fake_fetch, pause=0, now=t0)
    assert list(df.columns) == NS.COLS and len(df) == 2
    assert summ["failed"] == 1 and "BAD" in summ["failed_examples"] and summ["with_headlines"] == 2
    assert (df.fetched_utc == t0.isoformat(timespec="seconds")).all()


def test_append_keeps_the_first_sighting(tmp_path):
    a, _ = NS.snapshot(["AAPL"], fake_fetch, pause=0, now=pd.Timestamp("2026-11-13 22:00", tz="UTC"))
    b, _ = NS.snapshot(["AAPL"], fake_fetch, pause=0, now=pd.Timestamp("2026-11-13 23:00", tz="UTC"))
    NS.append(a, tmp_path, day="2026-11-13")
    f = NS.append(b, tmp_path, day="2026-11-13")
    out = pd.read_parquet(f)
    assert len(out) == 1 and out.fetched_utc.iat[0].startswith("2026-11-13T22:00")


def test_output_is_git_ignored():
    assert NS.OUT_DIR.relative_to(REPO).parts[0] == "artifacts"
    assert "artifacts/" in (REPO / ".gitignore").read_text(encoding="utf-8")
