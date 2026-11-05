"""Forward shadow test: frozen models forecast every evening; outcomes are scored only after they happen.

Protocol (docs/FORWARD_SHADOW_TEST.md):
* A frozen model is pinned by `registry/frozen_models/<version>.json` (SHA-256 of every bundle file). The run refuses to
  start if any file differs, so the logged forecasts always come from exactly the frozen weights.
* Each run forecasts the 500-stock universe from live Yahoo bars as of the last completed session t and writes
  `shadow/<version>/forecasts/<t>.csv`. A file is written once and never rewritten (first write wins).
* Every file is appended to a hash chain (`shadow/<version>/chain.csv`: run time, file SHA-256, previous row hash), so a
  later edit to any forecast breaks the chain. Committing the folder to git adds a public timestamp.
* A forecast counts only if it was written before its decision time (09:00 New York on session t+1). Late files are kept
  and flagged, never scored.
* Scoring uses the training label rule (targets.make_labels: entry at the t+1 open, exit at the t+h close, threshold
  max(0.5 sigma sqrt h, 10 bps)) on Yahoo bars fetched after the horizon has ended.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

SHADOW_ROOT = paths.REPO_ROOT / "shadow"
FROZEN_DIR = paths.REPO_ROOT / "registry" / "frozen_models"
CHAIN_FIELDS = ["run_utc", "asof_session", "file", "file_sha256", "n_rows", "n_failed", "late_rows", "prev_hash", "row_hash"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_record(version: str, bundle_dir: Path, extra: dict) -> dict:
    files = sorted(p for p in bundle_dir.iterdir() if p.is_file())
    return {"version": version, "bundle_id": bundle_dir.name,
            "files": {p.name: sha256_file(p) for p in files}, **extra}


def verify_frozen(version: str) -> tuple[Path, dict]:
    rec = json.loads((FROZEN_DIR / f"{version}.json").read_text(encoding="utf-8"))
    bdir = paths.ARTIFACTS / "bundles" / rec["bundle_id"]
    bad = [n for n, h in rec["files"].items() if not (bdir / n).exists() or sha256_file(bdir / n) != h]
    extra = sorted({p.name for p in bdir.iterdir() if p.is_file()} - set(rec["files"]))
    if bad or extra:
        raise RuntimeError(f"frozen model {version} does not match its record: changed={bad} unexpected={extra}")
    return bdir, rec


# ------------------------------------------------------------------------------------------------ hash chain
def _row_hash(prev: str, r: dict) -> str:
    return hashlib.sha256((prev + "|" + "|".join(str(r[k]) for k in CHAIN_FIELDS[:7])).encode()).hexdigest()


def chain_rows(version: str) -> list[dict]:
    p = SHADOW_ROOT / version / "chain.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_chain(version: str) -> list[str]:
    """Problems found (empty list = chain intact and every listed file unchanged)."""
    probs, prev = [], "GENESIS"
    for r in chain_rows(version):
        if r["prev_hash"] != prev or r["row_hash"] != _row_hash(prev, r):
            probs.append(f"chain broken at {r['asof_session']}")
        f = SHADOW_ROOT / version / r["file"]
        if not f.exists() or sha256_file(f) != r["file_sha256"]:
            probs.append(f"{r['file']} missing or edited after logging")
        prev = r["row_hash"]
    return probs


def append_chain(version: str, rec: dict) -> None:
    p = SHADOW_ROOT / version / "chain.csv"
    rows = chain_rows(version)
    rec["prev_hash"] = rows[-1]["row_hash"] if rows else "GENESIS"
    rec["row_hash"] = _row_hash(rec["prev_hash"], rec)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHAIN_FIELDS)
        if new:
            w.writeheader()
        w.writerow(rec)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.now(dt.timezone.utc))


def decision_utc_for(asof: pd.Timestamp) -> pd.Timestamp:
    """09:00 New York on the next weekday after the as-of session (holidays make this conservative: earlier, not later)."""
    nxt = pd.Timestamp(asof).normalize() + pd.offsets.BDay(1)
    return pd.Timestamp(nxt.strftime("%Y-%m-%d") + " 09:00", tz="America/New_York").tz_convert("UTC")


# ------------------------------------------------------------------------------------------------ scoring
def realized_labels(daily: pd.DataFrame, asof: pd.Timestamp, horizons: dict[str, int]) -> dict[str, dict]:
    """Training label rule on a live daily series: entry open[t+1], exit close[t+h], vol-scaled 3-class threshold."""
    from .indicators.core import ewma_sigma
    from .targets import COST_FLOOR, K_SIGMA
    d = daily.reset_index(drop=True)
    k = int(d.date.searchsorted(pd.Timestamp(asof)))
    if k >= len(d) or d.date.iloc[k] != pd.Timestamp(asof):
        return {}
    sig = float(ewma_sigma(d.close.iloc[: k + 1]).iloc[-1])
    out = {}
    for h, n in horizons.items():
        if k + n >= len(d):
            continue
        r = float(d.close.iloc[k + n] / d.open.iloc[k + 1] - 1)
        tau = max(K_SIGMA * sig * np.sqrt(n), COST_FLOOR)
        out[h] = {"ret": r, "tau": tau, "y": 2 if r > tau else (0 if r < -tau else 1),
                  "end_session": str(d.date.iloc[k + n].date())}
    return out
