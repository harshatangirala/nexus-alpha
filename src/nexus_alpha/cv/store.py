"""Read-only access to the fetched TriFrame s500 chart embeddings, with the fine-tuning governance rule.

ft_* backbones were fine-tuned on TriFrame targets with data to 2014 and early-stopped on 2015-2016. Their embeddings may
be used as fixed features for training at any date, but any EVALUATION row dated before FT_CLEAN_FROM would score a
representation that has seen those years' outcomes. `assert_eval_allowed` enforces this.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import paths

STORE = paths.ARTIFACTS / "cv_embeddings" / "s500"
VIEWS = ("W", "D", "H")
FT_CLEAN_FROM = pd.Timestamp("2017-01-01")


class EmbeddingGovernanceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def sample_index() -> pd.DataFrame:
    return pd.read_parquet(STORE / "sample_index.parquet")


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))


def available_backbones() -> list[str]:
    return sorted({k.split("__")[0] for k in manifest()["arrays"]})


def view_matrix(backbone: str, view: str) -> np.ndarray:
    return np.load(STORE / f"{backbone}__{view}.npy", mmap_mode="r")


def assert_eval_allowed(backbone: str, eval_dates) -> None:
    if backbone.startswith("ft_"):
        d = pd.to_datetime(pd.Series(eval_dates))
        if (d < FT_CLEAN_FROM).any():
            raise EmbeddingGovernanceError(
                f"{backbone}: fine-tuned on TriFrame targets through 2016 (early stopping); it cannot be evaluated on "
                f"rows before {FT_CLEAN_FROM.date()} ({int((d < FT_CLEAN_FROM).sum())} rows).")
