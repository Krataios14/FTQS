"""Applicability-domain scoring: when can a prediction be trusted?

A model trained on 131 literature datapoints will happily extrapolate
into composition/processing regimes it has never seen, silently. This
module makes that visible. Each query is scored by its distance to the
training distribution (k-nearest-neighbour distance in standardized
feature space, calibrated against the training set's own leave-one-out
neighbour distances) and assigned a trust tier:

- Tier A (interpolation): closer than 80% of training self-distances.
- Tier B (boundary): between the 80th and 97.5th percentile.
- Tier C (extrapolation): beyond the 97.5th percentile. The model is
  guessing here, and the conformal exchangeability assumption is weak.

Every query can also be traced to its nearest training specimens,
including the source publication, so a prediction is auditable:
"anchored by these measured datapoints from these papers".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

TIER_BOUNDS = {"A": 80.0, "B": 97.5}  # percentile thresholds


class TrustModel:
    """kNN applicability-domain model with provenance lookup."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.median_: Optional[np.ndarray] = None
        self.train_self_dist_: Optional[np.ndarray] = None
        self.metadata_: Optional[pd.DataFrame] = None
        self._nn: Optional[NearestNeighbors] = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        filled = np.where(np.isnan(X), self.median_, X)
        return (filled - self.center_) / self.scale_

    def fit(self, X: np.ndarray, metadata: Optional[pd.DataFrame] = None) -> "TrustModel":
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isnan(X), self.median_, X)
        self.center_ = np.median(filled, axis=0)
        q75, q25 = np.percentile(filled, [75, 25], axis=0)
        scale = q75 - q25
        std = np.std(filled, axis=0)
        scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
        self.scale_ = scale

        Z = (filled - self.center_) / self.scale_
        k = min(self.n_neighbors, len(Z) - 1)
        if k < 1:
            raise ValueError("TrustModel needs at least 2 training rows")
        self._nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        dists, _ = self._nn.kneighbors(Z)
        # Drop self-distance (column 0) -> mean distance to k nearest others
        self.train_self_dist_ = dists[:, 1:].mean(axis=1)
        self.metadata_ = metadata.reset_index(drop=True) if metadata is not None else None
        return self

    def _query_distances(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("fit must be called first")
        Z = self._standardize(X)
        k = min(self.n_neighbors, len(self.train_self_dist_))
        dists, _ = self._nn.kneighbors(Z, n_neighbors=k)
        return dists.mean(axis=1)

    def score(self, X: np.ndarray) -> pd.DataFrame:
        """Trust scores in [0, 100] plus tier labels for each query row."""
        d = self._query_distances(X)
        ref = np.sort(self.train_self_dist_)
        # Percentile of each query distance within the training distribution
        pct = np.searchsorted(ref, d, side="right") / len(ref) * 100.0
        score = np.clip(100.0 - pct, 0.0, 100.0)
        tiers = np.where(
            pct <= TIER_BOUNDS["A"], "A", np.where(pct <= TIER_BOUNDS["B"], "B", "C")
        )
        return pd.DataFrame(
            {
                "knn_distance": d,
                "distance_percentile": pct,
                "trust_score": score,
                "trust_tier": tiers,
            }
        )

    def neighbors(self, X: np.ndarray, k: int = 3) -> List[pd.DataFrame]:
        """Nearest training specimens (with metadata) for each query row."""
        if self._nn is None:
            raise RuntimeError("fit must be called first")
        Z = self._standardize(X)
        k = min(k, len(self.train_self_dist_))
        dists, idx = self._nn.kneighbors(Z, n_neighbors=k)
        out: List[pd.DataFrame] = []
        for row_d, row_i in zip(dists, idx):
            if self.metadata_ is not None:
                block = self.metadata_.iloc[row_i].reset_index(drop=True).copy()
            else:
                block = pd.DataFrame({"train_index": row_i})
            block.insert(0, "distance", row_d)
            out.append(block)
        return out

    def to_dict(self) -> Dict:
        return {
            "n_neighbors": self.n_neighbors,
            "center": self.center_,
            "scale": self.scale_,
            "median": self.median_,
            "train_self_dist": self.train_self_dist_,
            "metadata": self.metadata_,
            "nn": self._nn,
        }

    @classmethod
    def from_dict(cls, blob: Dict) -> "TrustModel":
        model = cls(n_neighbors=blob["n_neighbors"])
        model.center_ = blob["center"]
        model.scale_ = blob["scale"]
        model.median_ = blob["median"]
        model.train_self_dist_ = blob["train_self_dist"]
        model.metadata_ = blob["metadata"]
        model._nn = blob["nn"]
        return model
