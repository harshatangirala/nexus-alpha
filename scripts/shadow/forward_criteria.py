"""Evaluate the forward promotion criteria of v3_fund and v3_alpha against ranker_v3 (D-042, D-049, D-051).

    python scripts/shadow/forward_criteria.py

Reads the scored forward rows (shadow/<version>/scores.csv; on-time rows at 20D) and writes
shadow/forward_criteria.json. Before 120 scored sessions at 20D for every candidate it reports progress only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.evaluation import forward_criteria as FC  # noqa: E402

CANDIDATES, RANKER, H = ["v3_fund", "v3_alpha"], "v2", "20D"


def rows(version: str) -> pd.DataFrame:
    f = SH.SHADOW_ROOT / version / "scores.csv"
    if not f.exists():
        return pd.DataFrame(columns=["asof_session", "ticker", "rank_score", "real_ret"])
    s = pd.read_csv(f)
    s = s[(s.horizon == H) & ~s.late.astype(bool)]
    return s[["asof_session", "ticker", "rank_score", "real_ret"]]


def main() -> None:
    res = FC.evaluate({c: rows(c) for c in CANDIDATES}, rows(RANKER))
    dst = SH.SHADOW_ROOT / "forward_criteria.json"
    dst.write_text(json.dumps({"horizon": H, "ranker": f"{RANKER} rank_score (ranker_v3)", "results": res}, indent=1) + "\n",
                   encoding="utf-8")
    for c, r in res.items():
        print(f"forward criterion {c}: {r['sessions']} sessions at {H}; {r['decision']}")


if __name__ == "__main__":
    main()
