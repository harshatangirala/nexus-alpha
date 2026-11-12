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
# E28 (D-053) language blocks: FinBERT news, ChronoBERT filings, SEC price-impact expert. Each: (column prefix, flag)
NLP_BLOCKS = {"news_tx": ("fb_", "fb_n"), "sec_tx": ("sec_tx_", "sec_tx_present"), "sec_impact": ("si_", "si_present")}


@dataclass
class Preprocessor:
    backbone: str
    num_cols: list
    news_cols: list
    sec_cols: list
    rule_cols: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    vpca: object = None
    nlp_cols: dict = field(default_factory=dict)

    def _std(self, name, X: np.ndarray, fit_rows: np.ndarray | None):
        if fit_rows is not None:
            mu = np.nanmean(X[fit_rows], 0)
            sd = np.nanstd(X[fit_rows], 0)
            self.stats[name] = (np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0))
        mu, sd = self.stats[name]
        Z = (X - mu) / sd
        return np.clip(np.nan_to_num(Z, nan=0.0), -8, 8).astype(np.float32)


def check_feature_types(fr: pd.DataFrame, cols: list[str]) -> None:
    """Every model input must be a plain number. A timestamp or duration read as a number is calendar time, and a name
    rule cannot see it (audit 7: three `_avail_*` timestamps reached the SEC expert, D-054)."""
    bad = [c for c in cols if c in fr.columns and (pd.api.types.is_datetime64_any_dtype(fr[c])
                                                    or pd.api.types.is_timedelta64_dtype(fr[c])
                                                    or not pd.api.types.is_numeric_dtype(fr[c]))]
    if bad:
        raise TypeError(f"non-numeric or time-typed model inputs: {bad[:10]}")


def load_tables(nlp: bool = False) -> pd.DataFrame:
    """nlp: also join the E28 language blocks (FinBERT news, ChronoBERT filings, SEC price-impact expert)."""
    fr = pd.read_parquet(paths.ARTIFACTS / "samples" / "s500_frame.parquet")
    feat = paths.ARTIFACTS / "features"
    for name in ("news_features", "sec_features", "rule_features"):
        p = feat / f"{name}.parquet"
        if p.exists():
            fr = fr.merge(pd.read_parquet(p), left_on="sample_id", right_index=True, how="left")
    if nlp:
        fb = pd.read_parquet(feat / "finbert_features.parquet").set_index("sample_id")
        for t in (fb, pd.read_parquet(feat / "sec_tx_features.parquet"), pd.read_parquet(feat / "sec_impact_features.parquet")):
            fr = fr.merge(t, left_on="sample_id", right_index=True, how="left", validate="one_to_one")
    return fr


def build(fr: pd.DataFrame, train_mask: np.ndarray, modalities=ALL_MODALITIES, backbone: str = "convnext_tiny",
          pca_k: int = 64, pre: Preprocessor | None = None,
          embeddings: dict | None = None) -> tuple[ModalityData, Preprocessor]:
    """embeddings: freshly computed {view: (n, dim)} for rows without stored chart embeddings (the sealed evaluation);
    they pass through the same frozen view transform as stored ones. Only with a fitted `pre`."""
    fit = pre is None
    x33 = [c for c in fr.columns if c.startswith("x_")]
    if fit:
        pre = Preprocessor(backbone=backbone, num_cols=list(STRUCTURED_ALL) + x33,
                           news_cols=[c for c in fr.columns if c.startswith("news_")
                                      and c not in ("news_present", "news_n_selected_no_text")],
                           sec_cols=[c for c in fr.columns if c.startswith("sec_") and not c.startswith("sec_tx_")
                                     and c not in ("sec_covered",)],
                           rule_cols=[c for c in fr.columns if c.startswith("rule_")],
                           nlp_cols={m: [c for c in fr.columns if c.startswith(pfx) and c != flag and c != "sec_tx_present"]
                                     for m, (pfx, flag) in NLP_BLOCKS.items() if m in modalities})
        check_feature_types(fr, pre.num_cols + pre.news_cols + pre.sec_cols + pre.rule_cols
                            + [c for cols in pre.nlp_cols.values() for c in cols])
    rows_fit = np.flatnonzero(train_mask) if fit else None
    x, masks = {}, []
    img_views = [m.split("_")[1] for m in modalities if m.startswith("img_")]
    if img_views:
        if fit:
            pre.vpca = ViewPCA(backbone, ("W", "D", "H"), pca_k).fit(fr.row.to_numpy()[rows_fit])
        if embeddings is not None:
            assert not fit, "fresh embeddings need a fitted preprocessor"
            Zv = pre.vpca.transform_embeddings({v: embeddings[v] for v in img_views})
        else:
            Z = pre.vpca.transform(fr.row.to_numpy())
            Zv = {v: Z[[c for c in Z.columns if f"_{v}_" in c]].to_numpy(np.float32) for v in img_views}
        for v in img_views:
            x[f"img_{v}"] = pre._std(f"img_{v}", Zv[v], rows_fit)
            need = {"W": 42, "D": 51, "H": 56}[v]                      # same >= 40%-of-window rule as inference (A-03)
            present = (fr[f"n_valid_{v}"].to_numpy() >= need).astype(np.float32) if f"n_valid_{v}" in fr else np.ones(len(fr), np.float32)
            x[f"img_{v}"] *= present[:, None]
            masks.append(present)
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
    for m, (_, flag) in NLP_BLOCKS.items():
        if m not in modalities:
            continue
        f = fr.get(flag, pd.Series(0.0, index=fr.index)).fillna(0).to_numpy(np.float64)
        present = (f > 0).astype(np.float32)
        X = fr.reindex(columns=pre.nlp_cols[m]).to_numpy(np.float64)
        x[m] = pre._std(m, X, rows_fit[present[rows_fit] > 0] if fit else None) * present[:, None]
        masks.append(present)
    names = list(x)
    y = np.stack([fr[f"y_{h}"].fillna(-1).to_numpy(np.int64) for h in HORIZONS], 1)
    z = np.stack([fr[f"z_{h}"].to_numpy(np.float32) for h in HORIZONS], 1)
    return ModalityData(x, np.stack(masks, 1).astype(np.float32), y, z, names), pre
