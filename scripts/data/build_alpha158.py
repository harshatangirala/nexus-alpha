"""Alpha158 features (Qlib definitions; nexus_alpha.indicators.alpha158) at every as-of session of the v2 frame.

For each frame ticker: the sealed daily bars (2001 up to the seal), the 157 features on the full history, then the
frame's as-of sessions only. Output: artifacts/features/alpha158.parquet (ticker, asof_date, a158_*; float32) and
alpha158_meta.json. Pre-registered with E25 (docs/V3_RESEARCH.md). Reads no label.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.indicators import alpha158 as A  # noqa: E402


def main() -> None:
    t0 = time.time()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "v2_frame.parquet", columns=["ticker", "asof_date"])
    fr["asof_date"] = pd.to_datetime(fr.asof_date).astype("datetime64[ns]")
    assert not fr.duplicated().any(), "the frame has duplicate (ticker, as-of) rows"
    parts = []
    for i, (t, g) in enumerate(fr.groupby("ticker", sort=True)):
        d = B.read_daily(t)
        f = A.alpha158(d)
        f.insert(0, "asof_date", pd.to_datetime(d.date).astype("datetime64[ns]").to_numpy())
        want = set(g.asof_date)
        f = f[f.asof_date.isin(want)]
        assert len(f) == len(want), f"{t}: {len(want) - len(f)} as-of sessions have no bar"
        f.insert(0, "ticker", t)
        parts.append(f.astype({c: "float32" for c in A.COLUMNS}))
        if (i + 1) % 100 == 0:
            print(f"{i + 1} tickers ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)
    assert len(out) == len(fr)
    dst = paths.artifact("features", "alpha158.parquet")
    out.to_parquet(dst, index=False)
    miss = out[A.COLUMNS].isna().mean()
    meta = {"rows": len(out), "tickers": int(out.ticker.nunique()), "features": len(A.COLUMNS),
            "asof_range": [str(out.asof_date.min().date()), str(out.asof_date.max().date())],
            "missing_share_max": float(miss.max()), "missing_share_mean": float(miss.mean()),
            "most_missing": miss.sort_values(ascending=False).head(5).round(5).to_dict(),
            "code_sha256": hashlib.sha256((REPO / "src" / "nexus_alpha" / "indicators" / "alpha158.py").read_bytes()).hexdigest(),
            "elapsed_s": round(time.time() - t0, 1)}
    paths.artifact("features", "alpha158_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
