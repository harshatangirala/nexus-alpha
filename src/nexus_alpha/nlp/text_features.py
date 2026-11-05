"""Classical text representations for the CPU profile: Loughran-McDonald tone and TF-IDF -> truncated SVD.

* LM lexicon: the master dictionary (1993-2025) is read in place from the SEC study's inputs (licence restricts
  redistribution, so it is never copied). Category membership = a positive year in the category column. Caveat: the 2025
  vintage was compiled with filings up to 2025, a mild look-ahead for early samples, as accepted in the SEC study.
* TF-IDF/SVD: vocabulary, IDF weights and SVD axes are fitted on TRAINING-period text only, then frozen.
These are the CPU stand-ins for the transformer adapters (FinBERT, BGE, SEC-BERT, ChronoBERT) registered for GPU runs;
the SEC study found transformer embeddings added nothing over classical text (L7 - L6 = -0.001 on its locked test).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import paths

LM_CSV = paths.DOWNLOADS / "sec_report_mini" / "inputs" / "lexicons" / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
LM_CATEGORIES = ("Negative", "Positive", "Uncertainty", "Litigious", "Constraining")
_WORD = re.compile(r"[A-Za-z]+")


@lru_cache(maxsize=1)
def lm_sets() -> dict[str, frozenset]:
    d = pd.read_csv(LM_CSV, usecols=["Word"] + list(LM_CATEGORIES), keep_default_na=False)
    return {c: frozenset(d.Word[pd.to_numeric(d[c], errors="coerce").fillna(0) > 0].str.upper()) for c in LM_CATEGORIES}


def lm_tone(text: str) -> dict:
    words = [w.upper() for w in _WORD.findall(text or "")]
    n = len(words)
    s = lm_sets()
    out = {f"lm_{c.lower()}": (sum(w in s[c] for w in words) / n if n else np.nan) for c in LM_CATEGORIES}
    pos, neg = out["lm_positive"], out["lm_negative"]
    out["lm_net_tone"] = (pos - neg) / (pos + neg + 1e-3) if n else np.nan
    out["n_words"] = n
    return out


def lm_tone_frame(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([lm_tone(t) for t in texts.fillna("")], index=texts.index)


class TfidfSvd:
    def __init__(self, n_components: int = 64, max_features: int = 50000, min_df: int = 5, seed: int = 7):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(max_features=max_features, min_df=min_df, sublinear_tf=True, ngram_range=(1, 2),
                                   stop_words="english", token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b", dtype=np.float32)
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)

    def fit(self, train_texts: pd.Series):
        X = self.vec.fit_transform(train_texts.fillna(""))
        self.svd.fit(X)
        self.explained = float(self.svd.explained_variance_ratio_.sum())
        return self

    def transform(self, texts: pd.Series) -> np.ndarray:
        return self.svd.transform(self.vec.transform(texts.fillna(""))).astype(np.float32)
