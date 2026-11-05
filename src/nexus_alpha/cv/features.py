"""Chart-embedding features for tabular/linear heads: per-view PCA fitted on training rows only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import store as ST


class ViewPCA:
    def __init__(self, backbone: str, views=("W", "D", "H"), n_components: int = 64, seed: int = 7):
        self.backbone, self.views, self.k, self.seed = backbone, tuple(views), n_components, seed
        self.pca, self.mu, self.sd = {}, {}, {}

    def fit(self, rows_train: np.ndarray):
        for v in self.views:
            X = np.asarray(ST.view_matrix(self.backbone, v)[np.sort(rows_train)], dtype=np.float32)
            self.mu[v] = X.mean(0)
            self.sd[v] = X.std(0) + 1e-6
            self.pca[v] = PCA(self.k, svd_solver="randomized", random_state=self.seed).fit((X - self.mu[v]) / self.sd[v])
        return self

    def transform(self, rows: np.ndarray) -> pd.DataFrame:
        cols, mats = [], []
        for v in self.views:
            X = np.asarray(ST.view_matrix(self.backbone, v)[rows], dtype=np.float32)
            Z = self.pca[v].transform((X - self.mu[v]) / self.sd[v]).astype(np.float32)
            mats.append(Z)
            cols += [f"img_{self.backbone}_{v}_{i}" for i in range(self.k)]
        return pd.DataFrame(np.concatenate(mats, axis=1), columns=cols)

    def transform_embeddings(self, emb: dict) -> dict:
        """Freshly computed embeddings {view: (n, dim)} -> {view: (n, k)} with the frozen training transform."""
        return {v: self.pca[v].transform((np.asarray(e, np.float32) - self.mu[v]) / self.sd[v]).astype(np.float32)
                for v, e in emb.items()}

    def explained(self) -> dict:
        return {v: float(p.explained_variance_ratio_.sum()) for v, p in self.pca.items()}
