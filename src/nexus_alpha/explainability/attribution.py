"""Model diagnostics for one prediction. They are attributions of THIS model's output, not causal explanations.

* leave_one_modality_out: change in P(UP) - P(DOWN) and in P(SIDEWAYS) per horizon when one present modality is masked
  (the model was trained with modality dropout, so masking is an in-distribution operation).
* chart_occlusion: for each timeframe, the chart is split into time strips; each strip is blanked (set to background),
  re-embedded and re-scored; the change in the score is the strip's attribution. Model-agnostic and cheap on CPU.
"""
from __future__ import annotations

import numpy as np


def leave_one_modality_out(score_fn, x: dict, mask: np.ndarray, names: list[str]) -> dict:
    base = score_fn(x, mask)                                          # (H, 3) calibrated probabilities
    out = {}
    for i, n in enumerate(names):
        if mask[0, i] == 0:
            continue
        m2 = mask.copy()
        m2[0, i] = 0
        if m2.sum() == 0:
            continue
        p = score_fn(x, m2)
        out[n] = {"delta_up_minus_down": (base[:, 2] - base[:, 0] - (p[:, 2] - p[:, 0])).tolist(),
                  "delta_sideways": (base[:, 1] - p[:, 1]).tolist()}
    return out


def chart_occlusion(images: dict, embed_view, score_from_views, n_strips: int = 8, horizon_index: int = 2) -> dict:
    """images: {'W','D','H'} binary arrays; embed_view(view, img) -> embedding; score_from_views(dict) -> (H,3) probs.
    Returns per view the strip attributions of (P(UP) - P(DOWN)) at the chosen horizon (default 5D)."""
    base_emb = {v: embed_view(v, images[v]) for v in ("W", "D", "H")}
    p0 = score_from_views(base_emb)
    s0 = p0[horizon_index, 2] - p0[horizon_index, 0]
    out = {}
    for v in ("W", "D", "H"):
        img = images[v]
        w = img.shape[1]
        edges = np.linspace(0, w, n_strips + 1).astype(int)
        attr = []
        for a, b in zip(edges[:-1], edges[1:]):
            occ = img.copy()
            occ[:, a:b] = 0
            emb = dict(base_emb)
            emb[v] = embed_view(v, occ)
            p = score_from_views(emb)
            attr.append(float(s0 - (p[horizon_index, 2] - p[horizon_index, 0])))
        out[v] = {"strip_edges_px": edges.tolist(), "attribution": attr,
                  "note": "positive = this time strip pushed the 5D score towards UP; diagnostic only"}
    return out
