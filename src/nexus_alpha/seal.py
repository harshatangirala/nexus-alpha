"""The sealed final-evaluation period and the only way to open it.

NEXUS inherits TriFrame's boundary: the first sealed session is 2023-01-03. Development code may only see bars dated
before it, and only samples whose whole label window ends before it. The guard is enforced at read time: the bar
readers in `nexus_alpha.data.bars` push a date filter into the parquet scan, so sealed rows are never returned.

Honest status of the sealed period (see docs/POINT_IN_TIME_POLICY.md): it is sealed for NEXUS development, but it is
not a virgin test. TriFrame's pilot and 120-stock runs opened 2023-2026 once each, and it is the spent locked test of
the SEC study. NEXUS reports any result on it as "semi-confirmatory".

Unlocking mirrors TriFrame's guard v2: the owner sets NEXUS_HOLDOUT_CONFIG to the frozen configuration file and
NEXUS_HOLDOUT_UNLOCK to its SHA-256. One unlock per configuration hash; every unlock is appended to a hash-chained log
in this repository's governance folder (never in a source project). The key is never created or guessed by code.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import paths

SEAL_DATE = pd.Timestamp("2023-01-03")                      # first sealed session (inherits TriFrame HOLDOUT_START)
SEAL_UTC = pd.Timestamp("2023-01-03 00:00", tz="America/New_York").tz_convert("UTC")
LAST_DEV_SESSION = pd.Timestamp("2022-12-30")
NEWS_CORPUS_END_EXPECTED = pd.Timestamp("2025-12-31")       # verified separately in the news audit

_LOG_FIELDS = ["utc_time", "who", "config_hash", "rows_returned", "reuse_reason", "prev_hash", "row_hash"]


class HoldoutSealed(PermissionError):
    """Raised whenever development code would touch the sealed period."""


def _naive(ts: pd.Series | pd.DatetimeIndex) -> pd.Series:
    s = pd.to_datetime(pd.Series(ts))
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return s


def dev_mask(dates: Iterable, label_ends: Iterable | None = None) -> pd.Series:
    """True where the as-of date and (if given) the label end are both strictly before the seal.
    A missing label end (label not computable from sealed bars) does not unseal a row; such rows carry NaN labels."""
    d = _naive(pd.Series(list(dates)))
    ok = d < SEAL_DATE
    if label_ends is not None:
        e = _naive(pd.Series(list(label_ends)))
        ok &= ~(e >= SEAL_DATE)
    return ok.reset_index(drop=True)


def assert_dev(dates: Iterable, label_ends: Iterable | None = None, what: str = "frame") -> None:
    m = dev_mask(dates, label_ends)
    if not bool(m.all()):
        raise HoldoutSealed(f"{what}: {int((~m).sum())} rows reach the sealed period (on/after {SEAL_DATE.date()}).")


def purge_to_dev(df: pd.DataFrame, date_col: str, end_cols: Iterable[str] = ()) -> pd.DataFrame:
    ends = [c for c in end_cols if c in df.columns]
    keep = dev_mask(df[date_col])
    for c in ends:
        keep &= dev_mask(df[date_col], df[c])
    return df.loc[keep.values].copy()


# ------------------------------------------------------------------------------------------------ owner unlock
def _log_path() -> Path:
    paths.GOVERNANCE.mkdir(parents=True, exist_ok=True)
    return paths.GOVERNANCE / "holdout_access_log.csv"


def _rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_hash(prev: str, rec: dict) -> str:
    return hashlib.sha256((prev + "|" + "|".join(rec[k] for k in _LOG_FIELDS[:5])).encode()).hexdigest()


def verify_log() -> bool:
    prev = "GENESIS"
    for r in _rows(_log_path()):
        if r["prev_hash"] != prev or r["row_hash"] != _row_hash(prev, r):
            return False
        prev = r["row_hash"]
    return True


def unlock(who: str, rows_returned: int = 0) -> str:
    """Owner-authorised, single-use unlock. Returns the config hash. Raises HoldoutSealed otherwise."""
    cfg = os.environ.get("NEXUS_HOLDOUT_CONFIG", "")
    key = os.environ.get("NEXUS_HOLDOUT_UNLOCK", "")
    if not cfg or not os.path.isfile(cfg):
        raise HoldoutSealed("Sealed. The owner must set NEXUS_HOLDOUT_CONFIG (frozen config file) and "
                            "NEXUS_HOLDOUT_UNLOCK (its sha256).")
    sha = hashlib.sha256(Path(cfg).read_bytes()).hexdigest()
    if key != sha:
        raise HoldoutSealed("NEXUS_HOLDOUT_UNLOCK does not match the sha256 of the frozen config.")
    if not verify_log():
        raise HoldoutSealed("holdout access log failed its hash-chain check.")
    reason = os.environ.get("NEXUS_HOLDOUT_REUSE_REASON", "")
    if any(r["config_hash"] == sha and not r["reuse_reason"] for r in _rows(_log_path())) and not reason:
        raise HoldoutSealed("This frozen config has already opened the sealed period once (single-use rule).")
    p = _log_path()
    rows = _rows(p)
    rec = {"utc_time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "who": who,
           "config_hash": sha, "rows_returned": str(int(rows_returned)), "reuse_reason": reason}
    rec["prev_hash"] = rows[-1]["row_hash"] if rows else "GENESIS"
    rec["row_hash"] = _row_hash(rec["prev_hash"], rec)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(rec)
    return sha
