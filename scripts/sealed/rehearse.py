"""Rehearsal of the sealed scoring path on the last development year (acceptance test, no sealed data).

    python scripts/sealed/rehearse.py --year 2022

Takes the rows built by `build_period.py --rehearsal <year>` (bars, labels, x-features, rules, fresh chart embeddings),
adds news and SEC features rebuilt with the frozen TF-IDF models, the GMD/FRED context and the sequence windows, scores
them with the frozen v1.1 and v2 bundles through `nexus_alpha.sealed.apply`, and compares with:
* the development frame's features (exact) and TriFrame's stored embeddings (cosine);
* v1.1's stored deployed predictions (E08) and v2's stored calibrated predictions (E14) for the same rows.
Writes reports/verification/sealed_rehearsal_<year>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
warnings.filterwarnings("ignore")

from nexus_alpha import paths, shadow as SH  # noqa: E402
from nexus_alpha.sealed import apply as AP, assemble as AS  # noqa: E402

E08 = "20261106T020411_e08_final_model_5c4327"


def diff_stats(A: np.ndarray, B: np.ndarray, horizons: list[str]) -> dict:
    d = np.abs(A - B).max(-1)
    return {"max_abs_diff": float(d.max()), "median_max_abs_diff": float(np.median(d)),
            "p99_max_abs_diff": float(np.quantile(d, 0.99)),
            "lean_sign_agreement": float((np.sign(A[..., 2] - A[..., 0]) == np.sign(B[..., 2] - B[..., 0])).mean()),
            "by_horizon_max": {h: float(d[:, j].max()) for j, h in enumerate(horizons)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    a = ap.parse_args()
    t0 = time.time()
    tag = paths.ARTIFACTS / "sealed" / f"rehearsal_{a.year}"
    fr = pd.read_parquet(tag / "frame.parquet")
    emb = {v: np.load(tag / f"emb_{v}.npy") for v in ("W", "D", "H")}
    rep = {"year": a.year, "rows": len(fr)}
    full, info = AS.assemble(fr, beyond_seal=False, log=lambda m: print(m, flush=True))
    rep["assembly"] = info
    # ---- feature parity with the development frames
    dev = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet").set_index("sample_id")
    for name in ("news_features", "sec_features", "rule_features"):
        dev = dev.join(pd.read_parquet(paths.ARTIFACTS / "features" / f"{name}.parquet"), how="left", rsuffix="_dup")
    D = dev.loc[full.sample_id]
    bad, ncols = [], 0
    for c in full.columns:
        if c not in D.columns or c in ("ticker", "asof_date", "decision_utc", "sample_id") or c.startswith("mkt_"):
            continue
        x, y = pd.to_numeric(D[c], errors="coerce").to_numpy(float), pd.to_numeric(full[c], errors="coerce").to_numpy(float)
        tol = 2e-6 if str(D[c].dtype) == "float32" else 1e-6
        ok = (np.isnan(x) == np.isnan(y)) & (np.isnan(x) | np.isclose(x, y, rtol=tol, atol=1e-7))
        ncols += 1
        if not ok.all():
            bad.append({"column": c, "rows": int((~ok).sum())})
    rep["feature_parity"] = {"columns_compared": ncols, "mismatching_columns": bad}
    from nexus_alpha.cv import store as ST
    rows = D.row.to_numpy().astype(int)
    rep["embedding_cosine_vs_triframe"] = {}
    for v in ("W", "D", "H"):
        T = np.asarray(ST.view_matrix("convnext_tiny", v)[rows], np.float32)
        cos = (T * emb[v]).sum(1) / np.linalg.norm(T, axis=1) / np.linalg.norm(emb[v], axis=1)
        rep["embedding_cosine_vs_triframe"][v] = {"min": float(cos.min()), "median": float(np.median(cos))}
    print("features:", rep["feature_parity"]["columns_compared"], "compared,", len(bad), "mismatching;",
          "embeddings:", rep["embedding_cosine_vs_triframe"], flush=True)
    # ---- v1.1
    from nexus_alpha.inference.pipeline import Bundle
    b = Bundle(SH.verify_frozen("v1.1")[0])
    r1 = AP.v11(full, emb, b)
    s5 = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet", columns=["sample_id"])
    pos = pd.Series(np.arange(len(s5)), index=s5.sample_id)
    P08 = np.load(paths.ARTIFACTS / "predictions" / E08 / "final_deployed_calibrated__probs.npy")[pos.loc[full.sample_id].to_numpy()]
    rep["v1.1_vs_stored_E08"] = diff_stats(r1["probs"], P08, r1["horizons"])
    print("v1.1:", {k: v for k, v in rep["v1.1_vs_stored_E08"].items() if k != "by_horizon_max"}, flush=True)
    # ---- v2
    from nexus_alpha.inference.pipeline_v2 import NexusPredictorV2
    bdir, _ = SH.verify_frozen("v2")
    P2 = NexusPredictorV2(bdir)
    import e13_sequence as E13
    seq = E13.build_X(full[["ticker", "asof_date"]].reset_index(drop=True))
    r2 = AP.v2(full, emb, seq, P2)
    rows14 = pd.read_parquet(paths.ARTIFACTS / "predictions" / bdir.name / "rows.parquet")
    P14 = np.load(paths.ARTIFACTS / "predictions" / bdir.name / "probs_calibrated.npy")
    p14 = pd.Series(np.arange(len(rows14)), index=rows14.sample_id)
    have = full.sample_id.isin(p14.index).to_numpy()
    rep["v2_rows_with_stored_predictions"] = int(have.sum())
    rep["v2_vs_stored_E14"] = diff_stats(r2["probs"][have], P14[p14.loc[full.sample_id[have]].to_numpy()], r2["horizons"])
    print("v2:", {k: v for k, v in rep["v2_vs_stored_E14"].items() if k != "by_horizon_max"}, flush=True)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    out = REPO / "reports" / "verification" / f"sealed_rehearsal_{a.year}.json"
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
