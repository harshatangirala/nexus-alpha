"""Chief-audit checks of the fundamentals panel (E22 data).

    python scripts/verify/audit_fundamentals.py
1. point in time: 40 random (ticker, date) cells recomputed from facts filed on or before that date only;
2. split continuity: market cap across known splits (no jump by the split factor);
3. ranking sanity: largest market caps at 2021-12-31;
4. currency: reporting units of revenue per company (non-USD would corrupt ratios);
5. coverage by year and the share of extreme values per ratio.
Writes reports/verification/fundamentals_audit.json.
"""
from __future__ import annotations

import gzip
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "data"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import bars as B  # noqa: E402
from nexus_alpha.fundamentals import ratios as RT, sec_facts as SF, statements as ST  # noqa: E402


def main() -> None:
    import build_fundamentals as BF
    rep = {}
    P = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "panel.parquet")
    cm = pd.read_parquet(paths.ARTIFACTS / "fundamentals" / "cik_map.parquet").set_index("ticker").cik
    spl = BF.splits_table()
    # 1. point in time
    rng = np.random.default_rng(3)
    cells = P.dropna(subset=["earnings_yield"]).sample(40, random_state=3)
    bad = []
    for r in cells.itertuples():
        f = SF.load(cm[r.ticker])
        f = f[f.filed <= r.date]
        s = ST.snapshots(f, at=[f.filed.max()])
        sp = spl[spl.ticker == r.ticker].set_index("date").value
        d = RT.daily(s, B.read_daily(r.ticker), sp, start=str(r.date.date()), end=str(r.date.date()))
        v = float(d.earnings_yield.iloc[0]) if len(d) else np.nan
        if not np.isclose(v, r.earnings_yield, rtol=1e-4, equal_nan=True):
            bad.append((r.ticker, str(r.date.date()), float(r.earnings_yield), v))
    rep["pit_recompute"] = {"cells": 40, "mismatches": bad}
    # 2. split continuity
    checks = [("AAPL", "2020-08-31"), ("TSLA", "2020-08-31"), ("NVDA", "2021-07-20"), ("AMZN", "2022-06-06"), ("GOOGL", "2022-07-18")]
    cont = {}
    for t, d in checks:
        x = P[(P.ticker == t) & (P.date >= pd.Timestamp(d) - pd.Timedelta(days=6)) & (P.date <= pd.Timestamp(d) + pd.Timedelta(days=6))]
        if len(x) >= 2:
            mc = x.sort_values("date").market_cap.to_numpy()
            cont[f"{t} {d}"] = {"max_day_change": float(np.nanmax(np.abs(np.diff(np.log(mc))))), "mcap_bn": float(mc[-1] / 1e9)}
    rep["split_continuity"] = cont
    # 3. ranking sanity
    top = P[P.date == P.date[P.date <= "2021-12-31"].max()].nlargest(10, "market_cap")[["ticker", "market_cap"]]
    rep["top10_market_cap_2021_12_31"] = [(t, round(m / 1e9)) for t, m in top.itertuples(index=False)]
    # 4. currency units of revenue
    units = Counter()
    non_usd = []
    for t, c in cm.items():
        p = SF.raw_path(c)
        if not p.exists():
            continue
        js = json.loads(gzip.decompress(p.read_bytes()))
        g = js.get("facts", {}).get("us-gaap", {})
        u = set()
        for tag in SF.FLOW["revenue"] + ["Assets"]:
            u |= set((g.get(tag) or {}).get("units", {}).keys())
        units.update(u)
        if u and "USD" not in u:
            non_usd.append((t, sorted(u)))
        elif len(u - {"USD"}):
            non_usd.append((t, sorted(u)))
    rep["revenue_asset_units"] = dict(units)
    rep["companies_with_non_usd_units"] = non_usd
    # 5. coverage and extremes
    from nexus_alpha.fundamentals.ratios import RATIOS
    rep["coverage_by_year"] = P.assign(y=P.date.dt.year).groupby("y")[["earnings_yield", "fcf_yield", "roe", "piotroski_f", "sue"]].apply(
        lambda x: x.notna().mean()).round(3).to_dict()
    rep["extreme_share"] = {c: float(((P[c].abs() > 50) & P[c].notna()).mean()) for c in RATIOS}
    out = REPO / "reports" / "verification" / "fundamentals_audit.json"
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("coverage_by_year", "extreme_share")}, indent=1, default=str)[:3500])
    print("extreme (|x|>50) share > 1%:", {k: round(v, 3) for k, v in rep["extreme_share"].items() if v > 0.01})


if __name__ == "__main__":
    main()
