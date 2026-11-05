"""Assemble masked modality blocks for the fusion models from the s500 frame and the feature tables.

Every transformation (PCA of chart embeddings, imputation, standardisation) is fitted on TRAINING rows only and frozen
in the returned Preprocessor, which the inference bundle reuses. Missing evidence is zero-filled after standardisation
and flagged in the mask: img_* masks are 1 when that timeframe's chart exists, news when at least one article was
available, sec when the sample is inside SEC coverage. Labels missing because of the seal are -1 (ignored by the loss).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths
from ..cv.features import ViewPCA
from ..indicators.core import STRUCTURED_ALL
from ..targets import HORIZONS
from .trainer import ModalityData

ALL_MODALITIES = ("img_W", "img_D", "img_H", "num", "news", "sec", "rules")


@dataclass
class Preprocessor:
    backbone: str
    num_cols: list
    news_cols: list
    sec_cols: list
    rule_cols: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    vpca: object = None

    def _std(self, name, X: np.ndarray, fit_rows: np.ndarray | None):
        if fit_rows is not None:
            mu = np.nanmean(X[fit_rows], 0)
            sd = np.nanstd(X[fit_rows], 0)
            self.stats[name] = (np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0))
        mu, sd = self.stats[name]
        Z = (X - mu) / sd
        return np.clip(np.nan_to_num(Z, nan=0.0), -8, 8).astype(np.float32)


def load_tables() -> pd.DataFrame:
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet")
    feat = paths.ARTIFACTS / "features"
    for name in ("news_features", "sec_features", "rule_features"):
        p = feat / f"{name}.parquet"
        if p.exists():
            fr = fr.merge(pd.read_parquet(p), left_on="sample_id", right_index=True, how="left")
    return fr


def build(fr: pd.DataFrame, train_mask: np.ndarray, modalities=ALL_MODALITIES, backbone: str = "convnext_tiny",
          pca_k: int = 64, pre: Preprocessor | None = None) -> tuple[ModalityData, Preprocessor]:
    fit = pre is None
    x33 = [c for c in fr.columns if c.startswith("x_")]
    if fit:
        pre = Preprocessor(backbone=backbone, num_cols=list(STRUCTURED_ALL) + x33,
                           news_cols=[c for c in fr.columns if c.startswith("news_") and c != "news_present"],
                           sec_cols=[c for c in fr.columns if c.startswith("sec_") and c not in ("sec_covered",)],
                           rule_cols=[c for c in fr.columns if c.startswith("rule_")])
    rows_fit = np.flatnonzero(train_mask) if fit else None
    x, masks = {}, []
    img_views = [m.split("_")[1] for m in modalities if m.startswith("img_")]
    if img_views:
        if fit:
            pre.vpca = ViewPCA(backbone, ("W", "D", "H"), pca_k).fit(fr.row.to_numpy()[rows_fit])
        Z = pre.vpca.transform(fr.row.to_numpy())
        for v in img_views:
            cols = [c for c in Z.columns if f"_{v}_" in c]
            x[f"img_{v}"] = pre._std(f"img_{v}", Z[cols].to_numpy(np.float32), rows_fit)
            masks.append(np.ones(len(fr), np.float32))
    if "num" in modalities:
        X = fr[pre.num_cols].to_numpy(np.float64)
        x["num"] = pre._std("num", X, rows_fit)          # NaN (warm-up, missing hourly session) -> 0 after scaling
        masks.append(np.ones(len(fr), np.float32))
    if "news" in modalities:
        present = fr.get("news_present", pd.Series(0.0, index=fr.index)).fillna(0).to_numpy(np.float32)
        X = fr.reindex(columns=pre.news_cols).to_numpy(np.float64)
        fit_rows_news = rows_fit[present[rows_fit] > 0] if fit else None
        x["news"] = pre._std("news", X, fit_rows_news) * present[:, None]
        masks.append(present)
    if "sec" in modalities:
        present = fr.get("sec_covered", pd.Series(0.0, index=fr.index)).fillna(0).to_numpy(np.float32)
        X = fr.reindex(columns=pre.sec_cols).to_numpy(np.float64)
        fit_rows_sec = rows_fit[present[rows_fit] > 0] if fit else None
        x["sec"] = pre._std("sec", X, fit_rows_sec) * present[:, None]
        masks.append(present)
    if "rules" in modalities and pre.rule_cols:
        X = fr[pre.rule_cols].to_numpy(np.float64)
        x["rules"] = pre._std("rules", X, rows_fit)
        masks.append(np.ones(len(fr), np.float32))
    names = list(x)
    y = np.stack([fr[f"y_{h}"].fillna(-1).to_numpy(np.int64) for h in HORIZONS], 1)
    z = np.stack([fr[f"z_{h}"].to_numpy(np.float32) for h in HORIZONS], 1)
    return ModalityData(x, np.stack(masks, 1).astype(np.float32), y, z, names), pre
