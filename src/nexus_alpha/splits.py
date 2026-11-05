"""Chronological splits with label-window purging and an embargo (docs/PROJECT_SPECIFICATION.md section 6, D-006).

Primary development split (as-of session dates, inclusive):
    train 2003-2015 | inner 2016 (early stopping) | valid 2017-2018 (selection) | calib 2019 (calibration and
    decision thresholds) | test 2020-2022 (development test; TriFrame already read these years, so it is a historical
    diagnostic, not a fresh test) | sealed from 2023-01-03 (see seal.py).

Purge rule between consecutive blocks A -> B: a row of A is kept only if the session EMBARGO sessions after its
longest label end (21 sessions) is still before B's first session. So no training outcome window overlaps a later
block's outcome windows, with a further 5-session buffer. Rows are never split randomly; stocks are not held out
here (a separate leave-stocks-out check is in evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.calendar import offset_session, sessions
from .seal import SEAL_DATE
from .targets import MAX_H

EMBARGO_SESSIONS = 5


@dataclass(frozen=True)
class Block:
    name: str
    start: str
    end: str


PRIMARY = (
    Block("train", "2003-01-01", "2015-12-31"),
    Block("inner", "2016-01-01", "2016-12-31"),
    Block("valid", "2017-01-01", "2018-12-31"),
    Block("calib", "2019-01-01", "2019-12-31"),
    Block("test", "2020-01-01", "2022-12-31"),
)


def _first_session_on_or_after(d: str) -> pd.Timestamp:
    s = sessions()
    return s[s.searchsorted(pd.Timestamp(d))]


def assign_blocks(asof: pd.Series, blocks=PRIMARY, max_h: int = MAX_H, embargo: int = EMBARGO_SESSIONS) -> pd.Series:
    """Block name per row, or None if the row is purged/embargoed or outside every block. Also seals: no row whose
    longest label window reaches the seal can be in a block (its long-horizon labels are NaN anyway)."""
    a = pd.DatetimeIndex(pd.to_datetime(asof)).normalize()
    end = offset_session(a, max_h)
    end_emb = offset_session(end, embargo) if embargo else end
    out = np.full(len(a), None, dtype=object)
    for i, b in enumerate(blocks):
        inb = (a >= pd.Timestamp(b.start)) & (a <= pd.Timestamp(b.end))
        nxt = _first_session_on_or_after(blocks[i + 1].start) if i + 1 < len(blocks) else SEAL_DATE
        if i + 1 < len(blocks):
            ok = end_emb < nxt
        else:
            ok = np.ones(len(a), bool)        # last block: horizons that reach the seal are NaN-masked per horizon
        out[np.asarray(inb & ok)] = b.name
    return pd.Series(out, index=getattr(asof, "index", None))


def walk_forward_blocks(test_year: int, first_train_year: int = 2003) -> tuple[Block, ...]:
    """Expanding-window fold: train [first_train_year, Y-2], valid Y-1 (early stopping, calibration), test Y."""
    return (Block("train", f"{first_train_year}-01-01", f"{test_year - 2}-12-31"),
            Block("valid", f"{test_year - 1}-01-01", f"{test_year - 1}-12-31"),
            Block("test", f"{test_year}-01-01", f"{test_year}-12-31"))


WALK_FORWARD_TEST_YEARS = tuple(range(2008, 2023))


def verify_integrity(df: pd.DataFrame, block_col: str = "block", asof_col: str = "asof_date", blocks=PRIMARY,
                     max_h: int = MAX_H, embargo: int = EMBARGO_SESSIONS) -> dict:
    """Independent check of the purge: for every pair of consecutive blocks, the latest (label end + embargo) of the
    earlier block must precede the first as-of of the later block. Raises AssertionError on violation."""
    rep = {}
    names = [b.name for b in blocks]
    for a_name, b_name in zip(names[:-1], names[1:]):
        A = df[df[block_col] == a_name]
        B = df[df[block_col] == b_name]
        if A.empty or B.empty:
            rep[f"{a_name}->{b_name}"] = "empty"
            continue
        ends = offset_session(pd.DatetimeIndex(A[asof_col]), max_h)
        worst = offset_session(pd.DatetimeIndex([ends.max()]), embargo)[0]
        first_b = pd.Timestamp(B[asof_col].min())
        assert worst < first_b, f"purge violated between {a_name} and {b_name}: {worst} >= {first_b}"
        rep[f"{a_name}->{b_name}"] = {"last_end_plus_embargo": str(worst.date()), "next_first_asof": str(first_b.date()),
                                      "gap_sessions": int(sessions().searchsorted(first_b) - sessions().searchsorted(worst))}
    last = df[df[block_col] == names[-1]]
    if not last.empty:
        assert pd.Timestamp(last[asof_col].max()) < SEAL_DATE, "a development row is dated in the sealed period"
    return rep
