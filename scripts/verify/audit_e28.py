"""Hard checks of the E28 language inputs and decision (D-053). Every check recomputes from raw inputs, independently
of the builders where possible; any failure exits 1.

    .venv/Scripts/python -I scripts/verify/audit_e28.py [--run <experiments folder name>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.data import sec as S  # noqa: E402
from nexus_alpha.nlp import sec_impact as SI  # noqa: E402
from nexus_alpha.nlp import transformer_text as TT  # noqa: E402
from nexus_alpha.seal import SEAL_UTC  # noqa: E402

FEAT = paths.ARTIFACTS / "features"
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id", "ticker", "decision_utc", "block"])
    fr["decision_utc"] = pd.to_datetime(fr.decision_utc, utc=True)
    ev = S.load_events(dev_only=True)

    # ---- encoder record
    meta = json.loads((FEAT / "sec_tx" / "meta.json").read_text(encoding="utf-8"))
    check("encoder is ChronoBERT-2010 at the pinned revision", meta["model"] == TT.MODEL and meta["revision"] == TT.REVISION,
          f"{meta['model']}@{meta['revision'][:10]}")
    em = pd.read_parquet(FEAT / "sec_tx" / "events.parquet")
    E = np.load(FEAT / "sec_tx" / "event_emb.npy")
    check("one vector per encoded filing", len(E) == len(em) == meta["events"], f"{len(E):,}")
    check("no encoded filing on or after the seal", pd.to_datetime(em.available_utc, utc=True).max() < SEAL_UTC)
    empty = em.n_chars_clean.to_numpy() == 0
    check("filings without text have no vector (NaN), all others finite",
          bool(np.isnan(E[empty].astype(float)).all() and np.isfinite(E[~empty].astype(float)).all()),
          f"{int(empty.sum())} without text")

    # ---- PCA fitted on pre-2016 filings only (refit independently and compare)
    pz = np.load(FEAT / "sec_tx_pca.npz", allow_pickle=True)
    fit = (~empty) & (pd.to_datetime(em.available_utc, utc=True) < pd.Timestamp("2016-01-01", tz="UTC")).to_numpy()
    mu = E[fit].astype(np.float64).mean(0)
    check("PCA centre equals the mean of pre-2016 filings", np.allclose(mu, pz["mean"], atol=1e-9))

    # ---- per-sample sec_tx: brute force on random samples
    stx = pd.read_parquet(FEAT / "sec_tx_features.parquet")
    pos = {e: i for i, e in enumerate(em.event_id)}
    smp = fr.sample(a.n, random_state=5)
    bad, n_has = 0, 0
    for r in smp.itertuples():
        e = ev[(ev.ticker == r.ticker) & (ev.available_utc <= r.decision_utc)]
        for kind, days, types, short in (("8k", 30, ["8K"], "8k"), ("per", 120, ["periodic_10Q", "periodic_10K"], "per")):
            k = e[e.event_type.isin(types)]
            k = k[(r.decision_utc - k.available_utc) <= pd.Timedelta(days=days)]
            got = stx.loc[r.sample_id, f"sec_tx_{short}_present"]
            if k.empty:
                bad += int(got != 0)
                continue
            vec = E[pos[k.event_id.iloc[-1]]].astype(np.float64)
            want = (vec - pz["mean"]) @ pz["components"][:3].T if np.isfinite(vec).all() else np.full(3, np.nan)
            have = stx.loc[r.sample_id, [f"sec_tx_{short}_pc{i}" for i in range(3)]].to_numpy(float)
            n_has += 1
            bad += int(not (np.allclose(want, have, atol=1e-6, equal_nan=True)))
    check("sec_tx of random samples = PCA of the latest filing available at the decision time", bad == 0,
          f"{a.n:,} samples, {n_has:,} filing slots, {bad} mismatches")

    # ---- SEC expert: labels read, point-in-time evidence
    tab, cols = SI.load_table(ev)
    mx = max(tab[f"exit_h{h}"].max() for h in SI.HORIZONS)
    check("expert labels: no exit on or after 2019-01-01 was read", mx < pd.Timestamp("2019-01-01", tz="UTC"), str(mx))
    check("expert features: no year or accounting family",
          not [c for c in cols if c == "meta_year" or c.startswith(("acc_", "sue_", "ear_"))], f"{len(cols)} features")
    # the features the STORED expert was trained on (its meta.json), checked by type: a timestamp read as a number is
    # calendar time (audit 7, D-054); names alone missed three of them
    import pyarrow.parquet as pq
    used = json.loads((FEAT / "sec_impact" / "meta.json").read_text(encoding="utf-8"))["features"]
    sch = pq.read_schema(SI._stage("S09", "features.parquet"))
    non_num = [c for c in used if not str(sch.field(c).type).startswith(("double", "float", "int", "uint", "bool"))]
    check("stored expert used only numeric, non-time features", not non_num and used == cols,
          f"{len(used)} features; non-numeric: {non_num}")
    pred = pd.read_parquet(FEAT / "sec_impact" / "event_predictions.parquet")
    sif = pd.read_parquet(FEAT / "sec_impact_features.parquet")
    bad = 0
    for r in smp.itertuples():
        e = tab[(tab.ticker == r.ticker) & (tab.available_utc <= r.decision_utc)].sort_values("available_utc", kind="stable")
        e = e[(r.decision_utc - e.available_utc) <= SI.LOOKBACK]
        got = sif.loc[r.sample_id]
        if e.empty:
            bad += int(got.si_present != 0)
            continue
        want = 1 - pred.loc[e.index[-1], "pside_h5"]
        bad += int(got.si_present != 1 or not np.isclose(got.si_pmove_h5, want))
    check("sec_impact of random samples = the expert's prediction for the latest filing within 14 days", bad == 0,
          f"{bad} mismatches")
    m = json.loads((FEAT / "sec_impact" / "meta.json").read_text(encoding="utf-8"))
    check("expert trained only on pre-2016 events", all(v["train_events"] > 0 for v in m["fit"].values()))

    # ---- FinBERT record
    fbm = json.loads((FEAT / "finbert_features_meta.json").read_text(encoding="utf-8"))
    check("FinBERT pinned revision", fbm["revision"] == "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031")

    # ---- the decision, recomputed from the saved probabilities
    if a.run:
        from nexus_alpha.evaluation.metrics import log_loss
        from nexus_alpha.evaluation.stats import block_bootstrap, month_blocks
        from nexus_alpha.targets import HORIZONS
        M = json.loads((REPO / "experiments" / a.run / "metrics.json").read_text(encoding="utf-8"))
        pdir = paths.ARTIFACTS / "predictions" / a.run
        rows = pd.read_parquet(pdir / "rows.parquet")
        full = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id"] + [f"y_{h}" for h in HORIZONS])
        Y = full.set_index("sample_id").reindex(rows.sample_id)[[f"y_{h}" for h in HORIZONS]].fillna(-1).to_numpy(int)
        P = {k: np.load(pdir / f"{k}__probs.npy").astype(np.float64) for k in ("BASE", "MM", "PLACEBO")}
        val = (rows.block == "valid").to_numpy()

        def mll(Yv, Pv):
            return float(np.mean([log_loss(Yv[Yv[:, j] >= 0, j], Pv[Yv[:, j] >= 0, j]) for j in range(Yv.shape[1])]))
        for k, other in (("MM_vs_BASE", "BASE"), ("MM_vs_PLACEBO", "PLACEBO")):
            est = mll(Y[val], P[other][val]) - mll(Y[val], P["MM"][val])
            check(f"{k} validation gain recomputed", abs(est - M["validation"][k]["estimate"]) < 1e-5,
                  f"{est:+.6f} vs {M['validation'][k]['estimate']:+.6f}")
            bs = block_bootstrap(lambda yy, pa, pb: mll(yy, pb) - mll(yy, pa), (Y[val], P["MM"][val], P[other][val]),
                                 month_blocks(rows.asof_date[val]), n_boot=1000)
            check(f"{k} validation interval recomputed", np.allclose(bs["ci95"], M["validation"][k]["ci95"], atol=1e-5),
                  f"{bs['ci95']}")
        kept = all(M["validation"][k]["ci95"][0] > 0 for k in ("MM_vs_BASE", "MM_vs_PLACEBO"))
        check("decision follows the pre-registered rule", M["decision"] == ("KEPT" if kept else "NOT KEPT"), M["decision"])
    failed = [c for c in CHECKS if not c[1]]
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
