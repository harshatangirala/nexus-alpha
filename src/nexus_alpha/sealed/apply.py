"""Frozen bundles applied to a batch of evaluation rows, with the arithmetic of `pipeline.Bundle.probs` (v1.1) and
`pipeline_v2.NexusPredictorV2.predict` (v2), vectorised over rows. Nothing is refitted or recalibrated.

Inputs: an evaluation frame with the development frame's columns (bar features, labels, news/SEC/rule features, and for
v2 the GMD/FRED context columns), and fresh chart embeddings {view: (n, dim)} aligned to its rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fusion_forward(models, x: dict, mask: np.ndarray, chunk: int = 4096):
    import torch
    P, Q = [], []
    with torch.no_grad():
        for i in range(0, len(mask), chunk):
            xb = {k: torch.from_numpy(np.ascontiguousarray(v[i:i + chunk])) for k, v in x.items()}
            mb = torch.from_numpy(np.ascontiguousarray(mask[i:i + chunk]))
            outs = [m(xb, mb) for m in models]
            P.append(np.mean([torch.softmax(o[0], -1).numpy() for o in outs], 0))
            Q.append(np.mean([o[1].numpy() for o in outs], 0))
    return np.concatenate(P), np.sort(np.concatenate(Q), axis=-1)


def v11(frame: pd.DataFrame, emb: dict, bundle) -> dict:
    """bundle: pipeline.Bundle of the frozen v1.1 record. Returns calibrated probabilities (n, 6, 3) and quantiles."""
    from ..multimodal.dataset import build
    from ..targets import HORIZONS
    H = list(HORIZONS)
    names = bundle.names
    data, _ = build(frame, None, modalities=tuple(names), pre=bundle.pre, embeddings=emb)
    assert list(data.names) == list(names), (data.names, names)
    parts, w = [], []
    Q = None
    if bundle.models and bundle.w_fus > 0:
        pf, Q = _fusion_forward(bundle.models, data.x, data.mask)
        parts.append(pf); w.append(bundle.w_fus)
    raw = frame.reindex(columns=bundle.pre.num_cols).astype(float)
    if bundle.lgbm and bundle.w_lgbm > 0:
        parts.append(np.stack([bundle.lgbm[h].predict(raw) for h in H], 1)); w.append(bundle.w_lgbm)
    if bundle.logistic is not None and bundle.w_log > 0:
        xn = data.x["num"].astype(np.float64)
        parts.append(np.stack([bundle.logistic[h].predict_proba(xn) for h in H], 1)); w.append(bundle.w_log)
    w = np.array(w) / np.sum(w)
    P = sum(wi * pi for wi, pi in zip(w, parts))
    Pc = np.stack([bundle.calibrators[h].transform(P[:, j]) for j, h in enumerate(H)], 1)
    Pc = Pc / Pc.sum(-1, keepdims=True)
    return {"horizons": H, "probs": Pc, "probs_uncalibrated": P, "quantiles": Q}


def v2(frame: pd.DataFrame, emb: dict, seq_X: np.ndarray | None, predictor) -> dict:
    """predictor: pipeline_v2.NexusPredictorV2 of the frozen v2 record (its loaded components are used).
    frame needs the tabular columns, the fusion modalities' columns and the context columns; seq_X: raw (n, 63, C)
    windows from e13_sequence.build_X (standardised here with the bundle's channel statistics)."""
    from .. import seqmodels as SM
    from ..multimodal.dataset import build
    from ..targets_v2 import HORIZONS_V2
    H = list(HORIZONS_V2)
    n = len(frame)
    comp = {}
    X = frame.reindex(columns=predictor.cols).astype(float)
    if predictor.lgbm:
        comp["lgbm"] = {h: predictor.lgbm[h].predict(X) for h in predictor.lgbm}
    if predictor.logit:
        out = {}
        for h, m in predictor.logit.items():
            Z = ((X[m["cols"]].fillna(m["median"]) - m["mu"]) / m["sd"]).clip(-8, 8).to_numpy(np.float64)
            out[h] = m["model"].predict_proba(Z)
        comp["logistic"] = out
    Q = None
    if predictor.fusion is not None:
        fp = predictor.fpre
        mods = tuple(nm for nm in fp["names"] if nm != "ctx")
        data, _ = build(frame, None, modalities=mods, pre=fp["pre"], embeddings=emb)
        x, mask = dict(data.x), data.mask
        if "ctx" in fp["names"]:
            cols = fp["ctx_cols"]
            num = [c for c in cols if c != "rel_sector_id"]
            Z = np.clip(np.nan_to_num((frame.reindex(columns=num).to_numpy(np.float64) - fp["ctx_mu"]) / fp["ctx_sd"],
                                      nan=0.0), -8, 8)
            if "rel_sector_id" in cols:
                sid = frame.rel_sector_id.to_numpy()
                Z = np.concatenate([Z, np.stack([(sid == k).astype(float) for k in range(9)], 1)], 1)
            x["ctx"] = Z.astype(np.float32)
            mask = np.concatenate([mask, np.ones((n, 1), np.float32)], 1)
        x = {k: x[k] for k in fp["names"]}
        pf, Q = _fusion_forward(predictor.fusion, x, mask)
        comp["fusion"] = {h: pf[:, j] for j, h in enumerate(H)}
    if predictor.seq and seq_X is not None:
        Xs = np.clip(np.nan_to_num((seq_X - predictor.seq_mu) / predictor.seq_sd, nan=0.0), -8, 8).astype(np.float32)
        for arch, ms in predictor.seq.items():
            ps = np.mean([SM.predict(m, Xs) for m in ms], 0)
            comp[arch] = {h: ps[:, j] for j, h in enumerate(H)}
    P = np.zeros((n, len(H), 3))
    for j, h in enumerate(H):
        for c, wc in predictor.weights[h].items():
            if wc and c in comp and h in comp[c]:
                P[:, j] += wc * comp[c][h]
    Pc = np.stack([predictor.cal[h].transform(P[:, j]) for j, h in enumerate(H)], 1)
    Pc = Pc / Pc.sum(-1, keepdims=True)
    return {"horizons": H, "probs": Pc, "probs_uncalibrated": P, "quantiles": Q}
