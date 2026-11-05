"""Locations of the read-only source assets and of NEXUS's own artifacts.

Every source location can be overridden by an environment variable so the same code runs on Colab, where Google Drive
is mounted at /content/drive/MyDrive. Source trees are read-only for NEXUS: nothing in this package writes into them.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _p(env: str, default: str | Path) -> Path:
    return Path(os.environ.get(env, str(default)))


DOWNLOADS = _p("NEXUS_DOWNLOADS", Path.home() / "Downloads")

# TriFrame raw data store (frozen bars, chart caches, visible-window features). Read-only.
OHLCV_ROOT = _p("NEXUS_OHLCV_ROOT", DOWNLOADS / "us_stocks_ohlcv")
OHLCV_FINAL = OHLCV_ROOT / "final"
MARKET_CALENDAR_CSV = OHLCV_ROOT / "market_calendar.csv"
UNIVERSE_CSV = OHLCV_ROOT / "universe.csv"

# TriFrame code and results (git copy). Read-only.
TRIFRAME_REPO = _p("NEXUS_TRIFRAME_REPO", Path.home() / "triframe-cv")

# SEC study: repository, finished 150-stock run and the 500-firm CPU part. Read-only.
SEC_REPO = _p("NEXUS_SEC_REPO", DOWNLOADS / "sec-filings-price-impact")
SEC_MINI_RUN = _p("NEXUS_SEC_MINI_RUN", DOWNLOADS / "sec_report_mini" / "full")
SEC_FULL500_RUN = _p("NEXUS_SEC_FULL500_RUN", DOWNLOADS / "sec_inputs_staging" / "full")

# Knowledge-base sources (books and rulebooks). Read-only.
BOOKS_DIR = _p("NEXUS_BOOKS_DIR", DOWNLOADS)
RULEBOOK_DIR = _p("NEXUS_RULEBOOK_DIR", DOWNLOADS)

# NEXUS artifacts: large, never committed (data licences forbid publishing data, features, embeddings or weights).
ARTIFACTS = _p("NEXUS_ARTIFACTS", REPO_ROOT / "artifacts")
GOVERNANCE = _p("NEXUS_GOVERNANCE", REPO_ROOT / "governance")

HF_NEWS_DATASET = "Brianferrell787/financial-news-multisource"
HF_TRIFRAME_RUNS = "KrishnaHarshaTangirala/triframe-cv-runs"
HF_TRIFRAME_S500_BUNDLE = "KrishnaHarshaTangirala/triframe-cv-model-s500"


def artifact(*parts: str) -> Path:
    p = ARTIFACTS.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
