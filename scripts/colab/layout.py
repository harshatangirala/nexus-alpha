"""What the v3 stages need on disk, shared by the Drive staging script and the Colab layout check.

Each entry: (group, description, root, glob pattern relative to that root, minimum number of matches).
Roots are named, so the same table works on this PC (Downloads layout) and on Colab (Drive layout via NEXUS_* variables).
"""
from __future__ import annotations

from pathlib import Path

SEC_STAGES = ("S02", "S03", "S06", "S07")      # universe+SIC, corporate actions, SUE, filing times
BUNDLE_V11_RECORD = "registry/frozen_models/v1.1.json"

# group -> what it is for
GROUPS = {
    "fundamentals_build": "build_fundamentals.py and audit_fundamentals.py",
    "experiments": "e22_fundamentals.py, e23_industry_fundamentals.py, v3_fund_refit.py",
}


def requirements(repo: Path, paths) -> list[tuple[str, str, Path, str, int]]:
    """paths: the nexus_alpha.paths module (honours the NEXUS_* environment variables)."""
    art, ohlcv, sec = paths.ARTIFACTS, paths.OHLCV_ROOT, paths.SEC_FULL500_RUN
    r = [
        ("fundamentals_build", "daily bars (one parquet per stock)", ohlcv / "final" / "1day", "*.parquet", 900),
        ("fundamentals_build", "vendor split history (Yahoo)", ohlcv / "work", "yf/*.splits.parquet", 900),
        ("fundamentals_build", "vendor split history (fresh check)", ohlcv / "work", "verify_fresh/*.splits.parquet", 900),
        ("fundamentals_build", "CIK map", art / "fundamentals", "cik_map.parquet", 1),
        ("fundamentals_build", "SEC raw company facts (or SEC_USER_AGENT to download)", art / "fundamentals" / "raw", "CIK*.json.gz", 900),
        ("fundamentals_build", "SIC codes for S02 gaps", art / "fundamentals", "sic_sec.parquet", 1),
    ]
    for s in SEC_STAGES:
        r.append(("fundamentals_build", f"SEC study stage {s}", sec / s, "*/*.parquet", 1))
    r += [
        ("experiments", "v2 sample frame", art / "samples", "v2_frame.parquet", 1),
        ("experiments", "fundamentals panel", art / "fundamentals", "panel.parquet", 1),
        ("experiments", "frozen v1.1 record", repo, BUNDLE_V11_RECORD, 1),
        ("experiments", "v1.1 bundle (feature list; files are hash-checked)", art / "bundles", "*_e08_final_model_5c4327/*", 5),
        ("experiments", "frozen v3_fund record", repo, "registry/frozen_models/v3_fund.json", 1),
        ("experiments", "v3_fund bundle (files are hash-checked)", art / "bundles", "*_v3_fund_refit_*/*", 10),
    ]
    return r


def check(repo: Path, paths, groups: tuple[str, ...] | None = None) -> tuple[list[str], list[str]]:
    ok, missing = [], []
    for g, what, root, pat, n in requirements(repo, paths):
        if groups and g not in groups:
            continue
        got = len(list(root.glob(pat))) if root.exists() else 0
        (ok if got >= n else missing).append(f"[{g}] {what}: {got} of {n} at {root / pat}")
    return ok, missing
