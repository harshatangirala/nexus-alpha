"""E28 vision + NLP multimodal fusion (pre-registered in docs/V3_RESEARCH.md before any E28 feature or fit; D-053).

BASE = the six current blocks (three chart views through frozen ConvNeXt-T, indicators, classical news, classical SEC).
MM = BASE + news_tx (FinBERT) + sec_tx (ChronoBERT-2010 filings) + sec_impact (SEC price-impact expert).
PLACEBO = BASE + the three new blocks permuted across tickers within each as-of date (values and flags together).
Gated fusion, notebook section-8 settings, seeds 7/11/23 averaged. Keep rule: the month-block bootstrap intervals of
MM's validation mean six-horizon log-loss gain over BASE and over PLACEBO both lie above 0. Test 2020-22 is reported
once, after the decision, and selects nothing.

Run: .venv/Scripts/python -I scripts/experiments/e28_vision_nlp_fusion.py [--quick]
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nexus_alpha import paths  # noqa: E402
from nexus_alpha.evaluation.metrics import classification_report, log_loss  # noqa: E402
from nexus_alpha.evaluation.stats import block_bootstrap, month_blocks  # noqa: E402
from nexus_alpha.experiments import Run  # noqa: E402
from nexus_alpha.multimodal.dataset import NLP_BLOCKS, build, load_tables  # noqa: E402
from nexus_alpha.multimodal.fusion import FusionConfig  # noqa: E402
from nexus_alpha.multimodal.trainer import ModalityData, predict, train  # noqa: E402
from nexus_alpha.targets import HORIZONS  # noqa: E402

H = list(HORIZONS)
VISION = ("img_W", "img_D", "img_H")
BASE = VISION + ("num", "news", "sec")
NEW = tuple(NLP_BLOCKS)                                     # news_tx, sec_tx, sec_impact
MM = BASE + NEW
CONFIGS = {
    "BASE": BASE, "MM": MM, "PLACEBO": MM,
    "BASE+news_tx": BASE + ("news_tx",), "BASE+sec_tx": BASE + ("sec_tx",), "BASE+sec_impact": BASE + ("sec_impact",),
    "text_only": ("news", "sec") + NEW, "vision_only": VISION,
}
SEEDS = (7, 11, 23)
PLACEBO_SEED = 11
TRAIN = dict(epochs=30, batch=1024, lr=3e-4, weight_decay=1e-4, patience=4, schedule="cosine")
FCFG = dict(variant="gated", d_model=64, hidden=128, dropout=0.3, modality_dropout=0.15, quantile_weight=0.2)


def prereg_sha256() -> str:
    s = (REPO / "docs" / "V3_RESEARCH.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    start = s.index("## E28 Vision + NLP multimodal fusion (pre-registered")
    nxt = s.find("\n## ", start + 10)
    return hashlib.sha256(s[start: nxt if nxt > 0 else len(s)].encode("utf-8")).hexdigest()


def mean_ll(Y, P) -> float:
    vals = [log_loss(Y[Y[:, j] >= 0, j], P[Y[:, j] >= 0, j]) for j in range(Y.shape[1]) if (Y[:, j] >= 0).any()]
    return float(np.mean(vals))


def rank_ic_daily(dates, score, ret, min_n: int = 20) -> pd.Series:
    d = pd.DataFrame({"d": dates, "s": score, "r": ret}).dropna()
    d = d[d.groupby("d").s.transform("size") >= min_n]
    d["rs"] = d.groupby("d").s.rank()
    d["rr"] = d.groupby("d").r.rank()
    return d.groupby("d").apply(lambda x: np.corrcoef(x.rs, x.rr)[0, 1])


def date_bootstrap_mean(x: pd.Series, n_boot: int = 1000, seed: int = 7) -> dict:
    v = x.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    return {"estimate": float(v.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "n_dates": int(len(v))}


def permute_within_date(d: ModalityData, names: tuple, dates: np.ndarray, seed: int) -> ModalityData:
    """Placebo: each new block's rows (values and presence flag) move to another ticker of the same as-of date."""
    rng = np.random.default_rng(seed)
    perm = np.arange(len(d))
    for idx in pd.Series(np.arange(len(d))).groupby(dates).indices.values():
        perm[idx] = idx[rng.permutation(len(idx))]
    x = dict(d.x)
    mask = d.mask.copy()
    for n in names:
        j = d.names.index(n)
        x[n] = d.x[n][perm]
        mask[:, j] = d.mask[perm, j]
    return ModalityData(x, mask, d.y, d.z, d.names)


def select(d: ModalityData, mods) -> ModalityData:
    names = [m for m in d.names if m in mods]
    return ModalityData({k: d.x[k] for k in names}, d.mask[:, [d.names.index(k) for k in names]], d.y, d.z, names)


def join_nlp(fr: pd.DataFrame, quick: bool) -> pd.DataFrame:
    feat = paths.ARTIFACTS / "features"
    fb = pd.read_parquet(feat / "finbert_features.parquet").set_index("sample_id")
    parts = [fb, pd.read_parquet(feat / "sec_impact_features.parquet")]
    p = feat / "sec_tx_features.parquet"
    if p.exists():
        parts.append(pd.read_parquet(p))
    elif quick:                                   # dry run before the encoder finishes: an absent (all-missing) block
        parts.append(pd.DataFrame({"sec_tx_present": 0.0, "sec_tx_8k_pc0": np.nan}, index=fr.sample_id.values))
    else:
        raise FileNotFoundError(p)
    for t in parts:
        fr = fr.merge(t, left_on="sample_id", right_index=True, how="left", validate="one_to_one")
    return fr


def main() -> int:
    quick = "--quick" in sys.argv
    import torch
    torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 10)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)      # the nightly forward test keeps precedence
    except Exception:
        pass
    t0 = time.time()
    fr = join_nlp(load_tables(), quick)
    blocks = fr.block.to_numpy()
    tr_m, es_m, val_m, te_m = (blocks == b for b in ("train", "inner", "valid", "test"))
    seeds = (7,) if quick else SEEDS
    configs = {k: v for k, v in CONFIGS.items() if not quick or k in ("BASE", "MM", "PLACEBO")}
    train_kw = {**TRAIN, **({"epochs": 2} if quick else {})}
    cfg = {"configs": {k: list(v) for k, v in configs.items()}, "seeds": list(seeds), "train": train_kw, "fusion": FCFG,
           "placebo_seed": PLACEBO_SEED, "prereg_sha256": prereg_sha256(), "quick": quick}
    run = None if quick else Run("e28_vision_nlp_fusion", cfg)
    rid = run.run_id if run else "e28_quick"
    pred_dir = paths.ARTIFACTS / "predictions" / rid
    pred_dir.mkdir(parents=True, exist_ok=True)
    try:
        full, pre = build(fr, tr_m, MM, "convnext_tiny")
        Y = full.y
        dates = fr.asof_date.astype(str).to_numpy()
        placebo = permute_within_date(full, NEW, dates, PLACEBO_SEED)
        presence = {n: {b: float(full.mask[blocks == b, full.names.index(n)].mean()) for b in ("train", "valid", "test")}
                    for n in full.names}
        print("block presence:", json.dumps(presence), flush=True)
        probs, hist, gates, results = {}, {}, {}, {}
        models = {}
        for name, mods in configs.items():
            d = select(placebo if name == "PLACEBO" else full, mods)
            fc = FusionConfig(modality_dims={k: v.shape[1] for k, v in d.x.items()}, n_horizons=len(H), **FCFG)
            Ps, G = [], []
            for s in seeds:
                ck = paths.ARTIFACTS / "models" / rid / name.replace("+", "_") / f"seed{s}"
                model, h = train(fc, d.subset(np.flatnonzero(tr_m)), d.subset(np.flatnonzero(es_m)), ck, seed=s,
                                 monitor={"valid": d.subset(np.flatnonzero(val_m))}, log=lambda m: None, **train_kw)
                P, _, diag = predict(model, d)
                Ps.append(P)
                if "gates" in diag:
                    G.append(diag["gates"])
                hist[f"{name}|{s}"] = h
                models[(name, s)] = (model, fc)
                print(f"{name:<16} seed {s}: {len(h)} epochs, best inner {min(r['inner_ce'] for r in h):.5f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            P = np.mean(Ps, 0)
            probs[name] = P
            np.save(pred_dir / f"{name.replace('+', '_')}__probs.npy", P.astype(np.float32))
            if G:
                g = np.mean(G, 0)
                gates[name] = {"names": d.names, **{b: g[m].mean(0).tolist() for b, m in (("valid", val_m), ("test", te_m))}}
            results[name] = {"modalities": d.names, "valid_mean_log_loss": mean_ll(Y[val_m], P[val_m]),
                             "test_mean_log_loss": mean_ll(Y[te_m], P[te_m]),
                             "valid_by_horizon": {h_: classification_report(Y[val_m & (Y[:, j] >= 0), j],
                                                                             P[val_m & (Y[:, j] >= 0), j])
                                                  for j, h_ in enumerate(H)}}
            print(f"{name:<16} valid {results[name]['valid_mean_log_loss']:.5f}  test {results[name]['test_mean_log_loss']:.5f}",
                  flush=True)
        # ---------- decision (validation only)

        def gain(a, b, rows):
            """mean LL(b) - mean LL(a): positive = a is better; paired month-block bootstrap."""
            r = rows
            bl = month_blocks(fr.asof_date[r])
            return block_bootstrap(lambda yy, pa, pb: mean_ll(yy, pb) - mean_ll(yy, pa), (Y[r], probs[a][r], probs[b][r]),
                                   bl, n_boot=200 if quick else 1000)
        dec = {"MM_vs_BASE": gain("MM", "BASE", val_m), "MM_vs_PLACEBO": gain("MM", "PLACEBO", val_m)}
        kept = all(v["ci95"][0] > 0 for v in dec.values())
        text_rows = (full.mask[:, [full.names.index(n) for n in ("news", "news_tx", "sec_tx", "sec_impact")]].max(1) > 0)
        reported = {"valid_text_rows_MM_vs_BASE": gain("MM", "BASE", val_m & text_rows)}
        for n in ("BASE+news_tx", "BASE+sec_tx", "BASE+sec_impact"):
            if n in probs:
                reported[f"valid_{n}_vs_BASE"] = gain(n, "BASE", val_m)
        ric = {}
        for h_ in ("5D", "20D"):
            j = H.index(h_)
            for split, m in (("valid", val_m), ("test", te_m)):
                ics = {k: rank_ic_daily(dates[m], probs[k][m, j, 2] - probs[k][m, j, 0], fr[f"ret_{h_}"].to_numpy()[m])
                       for k in ("BASE", "MM")}
                ric[f"{split}_{h_}"] = {"BASE": date_bootstrap_mean(ics["BASE"]), "MM": date_bootstrap_mean(ics["MM"]),
                                        "MM_minus_BASE": date_bootstrap_mean((ics["MM"] - ics["BASE"]).dropna())}
        # ---------- one test report, after the decision
        test = {"MM_vs_BASE": gain("MM", "BASE", te_m), "MM_vs_PLACEBO": gain("MM", "PLACEBO", te_m),
                "text_rows_MM_vs_BASE": gain("MM", "BASE", te_m & text_rows)}
        expert = json.loads((paths.ARTIFACTS / "features" / "sec_impact" / "meta.json").read_text(encoding="utf-8"))["diagnostics"]
        summary = {"decision": "KEPT" if kept else "NOT KEPT", "decision_rule": "both validation intervals above 0",
                   "validation": dec, "reported_validation": reported, "rank_ic": ric, "test_after_decision": test,
                   "results": results, "gates": gates, "block_presence": presence, "sec_expert": expert,
                   "elapsed_s": round(time.time() - t0, 1)}
        (pred_dir / "rows.parquet").parent.mkdir(parents=True, exist_ok=True)
        fr[["sample_id", "ticker", "asof_date", "block"]].to_parquet(pred_dir / "rows.parquet", index=False)
        (pred_dir / "history.json").write_text(json.dumps(hist), encoding="utf-8")
        if kept and run:
            bdir = paths.ARTIFACTS / "bundles" / rid
            bdir.mkdir(parents=True, exist_ok=True)
            import torch
            for s in seeds:
                m, fc = models[("MM", s)]
                torch.save({"model": m.state_dict(), "cfg": asdict(fc)}, bdir / f"fusion_seed{s}.pt")
            with open(bdir / "preprocessor.pkl", "wb") as f:
                pickle.dump(pre, f)
            (bdir / "bundle_meta.json").write_text(json.dumps({
                "kind": "e28_mm", "bundle_id": rid, "modalities": list(MM), "seeds": list(seeds), "horizons": list(H),
                "trained_through_block": "train 2003-2015", "valid_mean_log_loss": results["MM"]["valid_mean_log_loss"],
                "frozen": False, "note": "not in the forward test; freezing needs a further decision (D-053)"},
                indent=1), encoding="utf-8")
        line = (f"{summary['decision']}: MM vs BASE {dec['MM_vs_BASE']['estimate']:+.5f} "
                f"[{dec['MM_vs_BASE']['ci95'][0]:+.5f},{dec['MM_vs_BASE']['ci95'][1]:+.5f}]; MM vs PLACEBO "
                f"{dec['MM_vs_PLACEBO']['estimate']:+.5f} [{dec['MM_vs_PLACEBO']['ci95'][0]:+.5f},"
                f"{dec['MM_vs_PLACEBO']['ci95'][1]:+.5f}]; test MM vs BASE {test['MM_vs_BASE']['estimate']:+.5f}")
        if run:
            run.save("gates.json", gates)
            run.finish("COMPLETED", line, summary)
        else:
            (pred_dir / "metrics_quick.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
        print(line, flush=True)
    except Exception as e:
        if run:
            run.finish("FAILED", f"{type(e).__name__}: {e}")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
