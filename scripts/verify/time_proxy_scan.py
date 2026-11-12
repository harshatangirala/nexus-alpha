"""Audit 7 (D-054): no model may read calendar time. Two checks on the inputs of every model that exists:

1. type: every input is a plain number (a timestamp or duration read as a number IS calendar time);
2. behaviour: no input tracks the calendar, |Spearman(feature, as-of date)| <= 0.9 on 40,000 random development rows
   (SEC expert: 20,000 events against their availability time).

Models scanned: the E28 fusion blocks (all nine), the SEC price-impact expert as stored (its meta.json), and the five
frozen forward-test bundles (LightGBM feature names, `cols` in bundle_meta, pickled fusion preprocessors). Inputs that
live outside the development frame (fundamentals, Alpha158, the market model's frame) are listed as not scanned here;
their point-in-time audits are `audit_fundamentals.py`, `alpha158_*_parity.py` and E19's D-030.

The name-based exclusions missed three `_avail_*` timestamps in the first E28 run; these checks would have caught them.

    .venv/Scripts/python -I scripts/verify/time_proxy_scan.py           # exit 1 on any finding
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha import shadow as SH  # noqa: E402
from nexus_alpha.multimodal.dataset import NLP_BLOCKS, load_tables  # noqa: E402
from nexus_alpha.indicators.core import STRUCTURED_ALL  # noqa: E402

RHO_MAX = 0.9
FROZEN = ("v1.1", "v2", "market_e19", "v3_fund", "v3_alpha")


def spearman_to_time(v: np.ndarray, t: np.ndarray) -> float | None:
    v = v.astype(float)
    ok = np.isfinite(v)
    if ok.sum() < 1000 or np.std(v[ok]) == 0:
        return None
    return float(pd.Series(v[ok]).rank().corr(pd.Series(t[ok]).rank()))


def is_plain_number(s: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)
            and not pd.api.types.is_timedelta64_dtype(s))


def bundle_inputs(version: str) -> set[str]:
    bdir, _ = SH.verify_frozen(version)                        # also re-checks the frozen hashes
    cols: set[str] = set()
    meta = json.loads((bdir / "bundle_meta.json").read_text(encoding="utf-8"))
    cols |= set(meta.get("cols", []))
    for f in bdir.glob("lgbm_*.txt"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("feature_names="):
                    cols |= set(line.strip().split("=", 1)[1].split(" "))
                    break
    for name in ("preprocessor.pkl", "fusion_preprocessor.pkl"):
        p = bdir / name
        if p.exists():
            with open(p, "rb") as fh:
                pre = pickle.load(fh)
            for a in ("num_cols", "news_cols", "sec_cols", "rule_cols"):
                cols |= set(getattr(pre, a, []) or [])
            for v in (getattr(pre, "nlp_cols", None) or {}).values():
                cols |= set(v)
    return cols


def main() -> int:
    findings: list[str] = []
    fr = load_tables(nlp=True)
    t = pd.to_datetime(fr.asof_date).astype("int64").to_numpy()
    ix = np.random.default_rng(0).choice(len(fr), 40_000, replace=False)
    x33 = [c for c in fr.columns if c.startswith("x_")]
    groups = {"E28 fusion": set(STRUCTURED_ALL) | set(x33)
              | {c for c in fr.columns if c.startswith(("news_", "sec_", "rule_"))}
              | {c for c in fr.columns for p, _ in NLP_BLOCKS.values() if c.startswith(p)}}
    for v in FROZEN:
        groups[f"frozen {v}"] = bundle_inputs(v)
    cache: dict[str, float | None] = {}
    for g, cols in groups.items():
        inside = sorted(c for c in cols if c in fr.columns)
        outside = sorted(c for c in cols if c not in fr.columns)
        bad_type = [c for c in inside if not is_plain_number(fr[c])]
        drift = []
        for c in inside:
            if c in bad_type:
                continue
            if c not in cache:
                cache[c] = spearman_to_time(fr[c].to_numpy()[ix], t[ix])
            if cache[c] is not None and abs(cache[c]) > RHO_MAX:
                drift.append(f"{c} ({cache[c]:+.3f})")
        print(f"{g}: {len(inside)} inputs scanned, {len(outside)} outside the frame (other audits); "
              f"non-numeric: {bad_type or 'none'}; calendar-like: {drift or 'none'}")
        findings += [f"{g}: type {c}" for c in bad_type] + [f"{g}: calendar {c}" for c in drift]
    # the stored SEC expert, on its own events
    import pyarrow.parquet as pq
    from nexus_alpha.nlp import sec_impact as SI
    used = json.loads((paths.ARTIFACTS / "features" / "sec_impact" / "meta.json").read_text(encoding="utf-8"))["features"]
    sch = pq.read_schema(SI._stage("S09", "features.parquet"))
    bad_type = [c for c in used if not re.match(r"(double|float|int|uint|bool)", str(sch.field(c).type))]
    tab, _ = SI.load_table()
    jx = np.random.default_rng(0).choice(len(tab), min(20_000, len(tab)), replace=False)
    te = tab.available_utc.astype("int64").to_numpy()[jx]
    drift = []
    for c in used:
        if c in bad_type:
            continue
        r = spearman_to_time(tab[c].to_numpy()[jx], te)
        if r is not None and abs(r) > RHO_MAX:
            drift.append(f"{c} ({r:+.3f})")
    print(f"SEC expert (stored): {len(used)} inputs; non-numeric: {bad_type or 'none'}; calendar-like: {drift or 'none'}")
    findings += [f"SEC expert: type {c}" for c in bad_type] + [f"SEC expert: calendar {c}" for c in drift]
    print("FINDINGS:" if findings else "no model reads calendar time", *findings, sep="\n  ")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
