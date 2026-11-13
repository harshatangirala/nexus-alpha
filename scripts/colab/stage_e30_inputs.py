"""Stage the input of notebook section 13B (extended fundamentals and the E30 replication) for Colab.

    python scripts/colab/stage_e30_inputs.py            # artifacts/notebook/fundamentals/, then the Drive copy if mounted

Writes `fundamentals/e30_stock_months.parquet`: one row per stock and month-end session (2005-2022, the same panel
`scripts/experiments/e30_factor_replication.py` builds): the as-known SEC XBRL snapshot columns (TTM flows, balance
sheet instants, now and a year earlier), the stored daily panel's price-based ratios on that session, the two-digit SIC
code and the realised 20-session return R = close[t+20] / open[t+1] - 1. Everything is read from the sealed development
stores (before 2023-01-03); nothing from the sealed period is staged. The notebook recomputes the published composites
from these columns with `fundamentals/extended.py` and repeats the E30 statistics.

It is a separate script because the main staging script rebuilds the whole 750 MB folder (about an hour).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.dont_write_bytecode = True

from nexus_alpha import paths  # noqa: E402

DRIVE = Path(r"G:\My Drive\antigravity_workspace\nexus_artifacts\notebook")


def _e30():
    spec = importlib.util.spec_from_file_location("e30", REPO / "scripts" / "experiments" / "e30_factor_replication.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    t0 = time.time()
    E = _e30()
    dates = E.signal_dates()
    df = E.build_panel(dates)
    drop = [c for c in df.columns if df[c].dtype == object and c not in ("ticker", "filed_form")]
    df = df.drop(columns=drop)
    assert pd.to_datetime(df.date).max() < pd.Timestamp("2023-01-03"), "sealed rows must never be staged"
    for c in df.columns:
        if str(df[c].dtype) == "float64":
            df[c] = df[c].astype(np.float32)
    # the extended metrics are recomputed in the notebook: keep only raw inputs
    from nexus_alpha.fundamentals import extended as X
    derived = set(X.metrics(df.head(2).copy()).columns) - set(df.columns) | set(X.metrics(df.head(2).copy()).columns)
    df = df.drop(columns=[c for c in derived if c in df.columns and c not in ("pe", "eps_growth", "piotroski_f", "altman_z")])
    out = paths.ARTIFACTS / "notebook" / "fundamentals"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "e30_stock_months.parquet"
    df.to_parquet(f, index=False, compression="zstd")
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    meta = {"rows": int(len(df)), "tickers": int(df.ticker.nunique()), "first": str(pd.to_datetime(df.date).min().date()),
            "last": str(pd.to_datetime(df.date).max().date()), "bytes": f.stat().st_size, "sha256": h,
            "columns": list(df.columns)}
    (out / "e30_stock_months.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"staged {meta['rows']:,} stock-months, {meta['tickers']} tickers, {meta['bytes'] / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
    if DRIVE.parent.exists():
        (DRIVE / "fundamentals").mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, DRIVE / "fundamentals" / f.name)
        shutil.copy2(out / "e30_stock_months.json", DRIVE / "fundamentals" / "e30_stock_months.json")
        print(f"copied to {DRIVE / 'fundamentals'}")


if __name__ == "__main__":
    main()
