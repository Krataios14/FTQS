"""Group-aware conformal prediction (CV+) for finite-sample-valid intervals.

Implements the CV+ / cross-conformal method of Barber, Candes, Ramdas &
Tibshirani, "Predictive inference with the jackknife+", Ann. Statist. 49
(2021): prediction intervals with a distribution-free coverage guarantee
of at least 1 - 2*alpha (and typically close to 1 - alpha in practice),
valid at any sample size and for any underlying regressor.

The group-aware variant folds the data by *group* (here: source
publication / alloy) rather than by row. Literature-mined materials
datasets are clustered -- several specimens per paper share lab, method
and material -- so row-level exchangeability is violated and naive
conformal intervals are overconfident for genuinely new alloys.
Group-level folding restores exchangeability at the level that matters:
"a material system we have never seen".
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


ModelFactory = Callable[[], object]


def _fold_assignment(groups: np.ndarray, n_folds: int, seed: int) -> Tuple[np.ndarray, int]:
    """Assign each row to a fold so that a group never straddles folds."""
    unique = np.unique(groups)
    n_folds = max(2, min(n_folds, len(unique)))
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique)
    group_to_fold = {g: i % n_folds for i, g in enumerate(shuffled)}
    folds = np.array([group_to_fold[g] for g in groups], dtype=int)
    return folds, n_folds


class GroupCVPlus:
    """CV+ conformal regressor with group-level folds.

    Parameters
    ----------
    model_factory : zero-argument callable returning an unfitted
        sklearn-style regressor (fresh instance per call).
    n_folds : number of cross-conformal folds (reduced automatically if
        there are fewer groups).
    seed : fold-assignment seed.
    """

    def __init__(self, model_factory: ModelFactory, n_folds: int = 8, seed: int = 42):
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.seed = seed
        self.fold_models_: list = []
        self.full_model_ = None
        self.residuals_: Optional[np.ndarray] = None
        self.fold_of_sample_: Optional[np.ndarray] = None
        self.n_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[Sequence] = None) -> "GroupCVPlus":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        if groups is None:
            groups = np.arange(n)
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError("groups must have the same length as y")

        folds, n_folds = _fold_assignment(groups, self.n_folds, self.seed)

        self.fold_models_ = []
        residuals = np.empty(n, dtype=np.float64)
        for k in range(n_folds):
            held = folds == k
            model = self.model_factory()
            model.fit(X[~held], y[~held])
            preds = np.asarray(model.predict(X[held]), dtype=np.float64).ravel()
            residuals[held] = np.abs(y[held] - preds)
            self.fold_models_.append(model)

        self.full_model_ = self.model_factory()
        self.full_model_.fit(X, y)
        self.residuals_ = residuals
        self.fold_of_sample_ = folds
        self.n_ = n
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.full_model_ is None:
            raise RuntimeError("fit must be called before predict")
        return np.asarray(self.full_model_.predict(np.asarray(X, dtype=np.float64))).ravel()

    def _fold_predictions(self, X: np.ndarray) -> np.ndarray:
        """Matrix of shape (n_folds, m): each fold model's predictions."""
        X = np.asarray(X, dtype=np.float64)
        return np.stack(
            [np.asarray(m.predict(X), dtype=np.float64).ravel() for m in self.fold_models_]
        )

    def predict_interval(self, X: np.ndarray, alpha: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """CV+ interval [lower, upper] with >= 1 - 2*alpha guaranteed coverage.

        For each test point x the bounds are order statistics of
        { mu_{-k(i)}(x) -/+ R_i } over the n training points, where
        mu_{-k(i)} is the model trained without sample i's fold and
        R_i its out-of-fold absolute residual.
        """
        if self.residuals_ is None:
            raise RuntimeError("fit must be called before predict_interval")
        if not 0.0 < alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5)")
        fold_preds = self._fold_predictions(X)  # (K, m)
        per_sample = fold_preds[self.fold_of_sample_, :]  # (n, m)
        v_lo = per_sample - self.residuals_[:, None]
        v_hi = per_sample + self.residuals_[:, None]

        n = self.n_
        i_lo = max(int(math.floor(alpha * (n + 1))), 1) - 1
        i_hi = min(int(math.ceil((1.0 - alpha) * (n + 1))), n) - 1
        lower = np.sort(v_lo, axis=0)[i_lo]
        upper = np.sort(v_hi, axis=0)[i_hi]
        return lower, upper


def interval_metrics(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(upper - lower)),
        "median_width": float(np.median(upper - lower)),
    }


def evaluate_group_coverage(
    model_factory: ModelFactory,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    alpha: float = 0.1,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    n_folds: int = 8,
    seed: int = 42,
) -> Dict[str, float]:
    """Held-out-group calibration evidence.

    Repeatedly splits off whole groups as a test set, fits a fresh
    GroupCVPlus on the remainder, and measures empirical coverage,
    interval width and point accuracy on the unseen groups. This is the
    honest estimate of how the tool behaves on a material system it has
    never encountered.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.RandomState(seed)

    coverages, widths, maes, all_true, all_pred = [], [], [], [], []
    for split in range(n_splits):
        test_groups = rng.choice(
            unique, size=max(1, int(round(test_fraction * len(unique)))), replace=False
        )
        test_mask = np.isin(groups, test_groups)
        if test_mask.all() or not test_mask.any():
            continue
        model = GroupCVPlus(model_factory, n_folds=n_folds, seed=seed + split)
        model.fit(X[~test_mask], y[~test_mask], groups=groups[~test_mask])
        lower, upper = model.predict_interval(X[test_mask], alpha=alpha)
        preds = model.predict(X[test_mask])
        m = interval_metrics(y[test_mask], lower, upper)
        coverages.append(m["coverage"])
        widths.append(m["mean_width"])
        maes.append(mean_absolute_error(y[test_mask], preds))
        all_true.append(y[test_mask])
        all_pred.append(preds)

    if not coverages:
        raise ValueError("No valid evaluation splits could be formed")
    y_cat = np.concatenate(all_true)
    p_cat = np.concatenate(all_pred)
    return {
        "alpha": float(alpha),
        "nominal_coverage": float(1.0 - alpha),
        "guaranteed_coverage": float(1.0 - 2.0 * alpha),
        "empirical_coverage": float(np.mean(coverages)),
        "coverage_std": float(np.std(coverages)),
        "mean_interval_width": float(np.mean(widths)),
        "mae": float(np.mean(maes)),
        "r2": float(r2_score(y_cat, p_cat)) if len(y_cat) > 1 else float("nan"),
        "n_splits": int(len(coverages)),
    }
