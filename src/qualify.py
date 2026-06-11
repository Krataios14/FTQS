"""Train a qualification-grade conformal model and emit its artifact.

One command takes the processed training table to a self-contained
artifact directory containing:

- conformal_model.joblib : pre-committed Mondrian conformal model
  (per-bin group-aware CV+ for the brittle and ductile phase classes,
  pooled fallback), built around the base regressor selected by median
  out-of-fold group-CV MAE across several fold seeds
- trust.joblib           : applicability-domain model with provenance
- scaler/encoder.joblib  : preprocessing artifacts
- features.json          : exact feature lists used
- model_card.json        : dataset summary, selection results,
  selection-inclusive nested calibration evidence with honest provable
  floors, a conditional coverage audit, out-of-fold permutation
  importance, and the replicate-scatter decomposition

Calibration claims are deliberately conservative in wording: the
Barber et al. (2021) Theorem 4 floor (1 - 2*alpha minus an O(K/n)
excess) is reported instead of the often-quoted 1 - 2*alpha, the
group-level extension is labeled heuristic, and the reported evidence
re-runs the model selection inside every held-out-group split so the
numbers describe the full pipeline, selection included.

Usage:
    python -m src.qualify --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from src.applicability import TrustModel
from src.config import load_config
from src.conformal import (
    GroupCVPlus,
    MondrianCVPlus,
    clopper_pearson,
    interval_metrics,
    provable_floor,
    subsample_one_per_group,
)
from src.data import fit_preprocessor, transform_df, transform_target, save_artifacts
from src.physics import brittleness_bin
from src.schema import apply_schema
from src.utils import ensure_dir, save_json, set_seed, timestamp

METADATA_CANDIDATES = [
    "composition_at_percent",
    "reference",
    "material_condition",
    "phase",
    "processing_history",
    "testing_temperature_k",
    "toughness_measure",
    "test_geometry",
]

ALPHA_GRID = [0.32, 0.20, 0.10, 0.05]
PRIMARY_ALPHA = 0.10


class SkModelFactory:
    """Picklable zero-arg factory for the candidate regressors.

    early_stopping is off by default: HistGradientBoostingRegressor
    would otherwise carve a ~12-row, non-group-aware internal validation
    set out of each ~125-row fold, making stopping noisy and optimistic.
    """

    def __init__(self, name: str, cfg: Dict):
        self.name = name
        self.seed = int(cfg.get("seed", 42))
        self.gbdt_params = dict(cfg.get("model", {}).get("gbdt", {}))
        self.et_params = dict(cfg.get("model", {}).get("extra_trees", {}))

    def _gbdt(self):
        p = self.gbdt_params
        return HistGradientBoostingRegressor(
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", None),
            max_leaf_nodes=p.get("max_leaf_nodes", 63),
            max_bins=p.get("max_bins", 255),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 0.1),
            max_iter=p.get("max_iter", 500),
            early_stopping=p.get("early_stopping", False),
            random_state=self.seed,
        )

    def _extra_trees(self):
        p = self.et_params
        return ExtraTreesRegressor(
            n_estimators=p.get("n_estimators", 400),
            max_depth=p.get("max_depth", None),
            min_samples_leaf=p.get("min_samples_leaf", 5),
            max_features=p.get("max_features", 1.0),
            bootstrap=p.get("bootstrap", False),
            random_state=self.seed,
            n_jobs=-1,
        )

    def __call__(self):
        if self.name == "gbdt":
            return self._gbdt()
        if self.name == "extra_trees":
            return self._extra_trees()
        if self.name == "blend":
            return VotingRegressor(
                [("gbdt", self._gbdt()), ("extra_trees", self._extra_trees())]
            )
        if self.name == "random_forest":
            return RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=5,
                random_state=self.seed,
                n_jobs=-1,
            )
        if self.name == "ridge":
            return Ridge(alpha=1.0, random_state=self.seed)
        raise ValueError(f"Unknown candidate: {self.name}")


def build_group_key(df: pd.DataFrame, cfg: Dict) -> np.ndarray:
    """Group rows by source publication + alloy so conformal folds and
    calibration splits never leak a material system across the split."""
    preferred = ["reference", "composition_at_percent"]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = [c for c in cfg.get("data", {}).get("group_columns", []) if c in df.columns]
    if not cols:
        return np.arange(len(df))
    key = df[cols].astype(str).fillna("UNK").agg("|".join, axis=1)
    return key.to_numpy()


def prune_numeric_features(df: pd.DataFrame, num_features: List[str], cfg: Dict) -> List[str]:
    data_cfg = cfg.get("data", {})
    num_features = [c for c in num_features if c in df.columns and not df[c].isna().all()]
    min_frac = data_cfg.get("min_non_null_fraction")
    if min_frac is not None:
        num_features = [c for c in num_features if df[c].notna().mean() >= min_frac]
    min_nz = data_cfg.get("min_non_zero_fraction")
    if min_nz is not None:
        num_features = [
            c
            for c in num_features
            if not c.startswith("elem_") or (df[c].fillna(0.0) != 0.0).mean() >= min_nz
        ]
    return num_features


def build_matrix(
    df: pd.DataFrame, num_features: List[str], cat_features: List[str], artifacts
) -> np.ndarray:
    x_num, x_num_mask, x_cat = transform_df(df, num_features, cat_features, artifacts)
    return np.concatenate([x_num, x_num_mask, x_cat.astype(np.float32)], axis=1)


def make_inverse_transform(artifacts, standardize: bool):
    """Monotone inverse of the target transform, without clipping.

    The training-range clipping used for point prediction must NOT be
    applied to interval bounds: clipping a lower bound upward would be
    anti-conservative. Infinite bounds pass through as infinite.
    """

    def _inv(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if standardize:
            y = y * artifacts.target_std + artifacts.target_mean
        if artifacts.target_transform == "log1p":
            finite = np.isfinite(y)
            out = np.array(y, dtype=np.float64)
            out[finite] = np.expm1(np.clip(y[finite], -700.0, 700.0))
            return out
        return y

    return _inv


def _dataset_fingerprint(csv_path: str) -> str:
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_bins(df: pd.DataFrame) -> np.ndarray:
    """Pre-committed Mondrian bins from the reported phase structure."""
    phase = df["phase"] if "phase" in df.columns else pd.Series(np.nan, index=df.index)
    return phase.map(brittleness_bin).to_numpy()


def select_candidate(
    candidates: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: Dict,
    n_folds: int,
    n_seeds: int,
) -> Tuple[str, Dict[str, List[float]]]:
    """Pick the base regressor by median OOF group-CV MAE across seeds.

    Single-seed argmin over several candidates is close to a coin flip
    at this n (between-candidate gaps are smaller than seed-to-seed
    noise), so the median over multiple fold assignments is used and
    every per-seed value is recorded for the model card.
    """
    base_seed = int(cfg.get("seed", 42))
    per_seed: Dict[str, List[float]] = {name: [] for name in candidates}
    for name in candidates:
        for s in range(n_seeds):
            model = GroupCVPlus(
                SkModelFactory(name, cfg), n_folds=n_folds, seed=base_seed + 101 * s
            )
            model.fit(X, y, groups=groups)
            per_seed[name].append(float(np.mean(model.residuals_)))
    medians = {name: float(np.median(v)) for name, v in per_seed.items()}
    best = min(medians, key=medians.get)
    return best, per_seed


def _strata_for(df: pd.DataFrame, bins: np.ndarray) -> Dict[str, np.ndarray]:
    """Fixed row attributes used by the conditional coverage audit."""
    n = len(df)

    def col(name: str, default: str) -> np.ndarray:
        if name in df.columns:
            return df[name].fillna(default).astype(str).to_numpy()
        return np.full(n, default, dtype=object)

    temp = pd.to_numeric(
        df.get("testing_temperature_k", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    temp_class = np.where(
        temp.isna(), "unknown", np.where(temp < 293, "below_RT", np.where(temp > 303, "above_RT", "RT"))
    )
    geometry = col("test_geometry", "unknown")
    geometry_class = np.where(
        geometry == "indentation", "indentation", np.where(geometry == "unknown", "unknown", "specimen")
    )
    return {
        "phase_bin": bins.astype(str),
        "geometry": geometry_class.astype(str),
        "measure": col("toughness_measure", "manual"),
        "temperature": temp_class.astype(str),
    }


def nested_evidence(
    df: pd.DataFrame,
    X: np.ndarray,
    y_t: np.ndarray,
    groups: np.ndarray,
    bins: np.ndarray,
    cfg: Dict,
    candidates: Sequence[str],
    inverse_transform: Callable[[np.ndarray], np.ndarray],
    n_splits: int,
    n_folds: int,
    selection_seeds: int,
    test_fraction: float = 0.2,
) -> Dict[str, object]:
    """Selection-inclusive held-out-group calibration evidence.

    Each outer split holds out whole groups, re-runs the candidate
    selection on the remaining groups only, fits the pre-committed
    Mondrian conformal model, and scores the held-out groups. Covered
    indicators are pooled across splits (never averaged as per-split
    rates), with exact Clopper-Pearson bands. The conditional audit is
    reported at the single pre-registered alpha of 0.10 on fixed row
    attributes, with thin strata flagged as insufficient rather than
    colored as pass/fail.
    """
    seed = int(cfg.get("seed", 42))
    rng = np.random.RandomState(seed + 7919)
    unique = np.unique(groups)
    strata = _strata_for(df, bins)

    covered = {a: [] for a in ALPHA_GRID}
    widths = {a: [] for a in ALPHA_GRID}
    test_rows_idx: List[np.ndarray] = []
    all_true, all_pred = [], []
    selected_per_split: List[str] = []

    for split in range(n_splits):
        test_groups = rng.choice(
            unique, size=max(1, int(round(test_fraction * len(unique)))), replace=False
        )
        test_mask = np.isin(groups, test_groups)
        if test_mask.all() or not test_mask.any():
            continue
        tr, te = ~test_mask, test_mask

        name, _ = select_candidate(
            candidates, X[tr], y_t[tr], groups[tr], cfg, n_folds, selection_seeds
        )
        selected_per_split.append(name)
        model = MondrianCVPlus(
            SkModelFactory(name, cfg), n_folds=n_folds, seed=seed + split
        ).fit(X[tr], y_t[tr], groups[tr], bins[tr])

        intervals = model.predict_interval_multi(X[te], ALPHA_GRID, bins[te])
        preds = inverse_transform(model.predict(X[te], bins[te]))
        y_true = inverse_transform(y_t[te])
        for a in ALPHA_GRID:
            lo, hi = inverse_transform(intervals[a][0]), inverse_transform(intervals[a][1])
            covered[a].append((y_true >= lo) & (y_true <= hi))
            widths[a].append(hi - lo)
        test_rows_idx.append(np.flatnonzero(te))
        all_true.append(y_true)
        all_pred.append(preds)

    if not all_true:
        raise ValueError("No valid evaluation splits could be formed")

    y_cat = np.concatenate(all_true)
    p_cat = np.concatenate(all_pred)
    rows_cat = np.concatenate(test_rows_idx)
    n_train_typ = int(len(y_t) * (1 - test_fraction))

    per_alpha: Dict[str, Dict[str, object]] = {}
    for a in ALPHA_GRID:
        cov = np.concatenate(covered[a])
        wid = np.concatenate(widths[a])
        finite = np.isfinite(wid)
        k, n = int(cov.sum()), int(len(cov))
        lo_ci, hi_ci = clopper_pearson(k, n)
        per_alpha[f"alpha_{a:.2f}"] = {
            "alpha": a,
            "nominal_coverage": 1.0 - a,
            "provable_floor_rowlevel": provable_floor(a, n_train_typ, n_folds),
            "n": n,
            "covered": k,
            "empirical_coverage": k / n,
            "coverage_ci95": [lo_ci, hi_ci],
            "mean_width": float(np.mean(wid[finite])) if finite.any() else float("inf"),
            "median_width": float(np.median(wid[finite])) if finite.any() else float("inf"),
            "n_unbounded": int((~finite).sum()),
        }

    cov_primary = np.concatenate(covered[PRIMARY_ALPHA])
    audit: Dict[str, Dict[str, object]] = {}
    for stratum_name, labels in strata.items():
        levels: Dict[str, object] = {}
        test_labels = labels[rows_cat]
        test_groups_arr = groups[rows_cat]
        for level in sorted(set(test_labels)):
            mask = test_labels == level
            k, n = int(cov_primary[mask].sum()), int(mask.sum())
            n_grp = len(np.unique(test_groups_arr[mask]))
            lo_ci, hi_ci = clopper_pearson(k, n)
            levels[level] = {
                "n": n,
                "n_groups": n_grp,
                "covered": k,
                "coverage": k / n if n else float("nan"),
                "coverage_ci95": [lo_ci, hi_ci],
                "insufficient": bool(n < 15 or n_grp < 5),
            }
        audit[stratum_name] = levels

    return {
        "protocol": (
            "selection-inclusive nested evaluation: candidate selection is "
            "re-run inside every held-out-group split, so these numbers "
            "describe the deployed pipeline including its data-driven choices"
        ),
        "audit_alpha": PRIMARY_ALPHA,
        "per_alpha": per_alpha,
        "audit": audit,
        "mae": float(mean_absolute_error(y_cat, p_cat)),
        "r2": float(r2_score(y_cat, p_cat)) if len(y_cat) > 1 else float("nan"),
        "n_splits": len(all_true),
        "selected_per_split": selected_per_split,
        "note_rows_vs_groups": (
            "coverage counts pool rows; rows within one publication are "
            "correlated, so effective sample sizes are closer to the group "
            "counts shown in the audit"
        ),
    }


def reference_subsampled_evidence(
    factory: SkModelFactory,
    X: np.ndarray,
    y_t: np.ndarray,
    groups: np.ndarray,
    inverse_transform,
    seed: int,
    n_splits: int = 4,
) -> Dict[str, object]:
    """Provably valid hierarchical reference: one row per group.

    Subsampling one row per publication restores exchangeability for a
    new-group test point (Dunn, Wasserman & Ramdas 2022), at the price
    of discarding replicates. Reported beside the main evidence so the
    cost of the group-folding heuristic is visible.
    """
    Xs, ys, gs = subsample_one_per_group(X, y_t, groups, seed=seed)
    rng = np.random.RandomState(seed + 13)
    unique = np.unique(gs)
    cov, wid = [], []
    for split in range(n_splits):
        test_groups = rng.choice(unique, size=max(1, len(unique) // 5), replace=False)
        te = np.isin(gs, test_groups)
        if te.all() or not te.any():
            continue
        model = GroupCVPlus(factory, n_folds=8, seed=seed + split).fit(
            Xs[~te], ys[~te], groups=gs[~te]
        )
        lo, hi = model.predict_interval(Xs[te], alpha=PRIMARY_ALPHA)
        y_true = inverse_transform(ys[te])
        lo, hi = inverse_transform(lo), inverse_transform(hi)
        cov.append((y_true >= lo) & (y_true <= hi))
        wid.append(hi - lo)
    cov = np.concatenate(cov)
    wid = np.concatenate(wid)
    k, n = int(cov.sum()), len(cov)
    lo_ci, hi_ci = clopper_pearson(k, n)
    return {
        "alpha": PRIMARY_ALPHA,
        "n_rows_used": int(len(ys)),
        "n": n,
        "empirical_coverage": k / n,
        "coverage_ci95": [lo_ci, hi_ci],
        "median_width": float(np.median(wid[np.isfinite(wid)])),
        "note": "one subsampled row per publication; provably valid under the two-layer hierarchical model",
    }


def replicate_scatter(df: pd.DataFrame, y_raw: np.ndarray) -> Dict[str, object]:
    """Within-lab vs between-lab scatter on replicated test conditions.

    Clusters are (composition, test temperature) pairs, so a single
    paper's temperature sweep does not masquerade as scatter. Computed
    in log1p units on clusters reported by more than one publication.
    The between-lab component is irreducible at query time: no
    composition-based model can resolve which lab's value a new
    measurement will reproduce.
    """
    comp = df.get("composition_at_percent", pd.Series("", index=df.index)).fillna("")
    ref = df.get("reference", pd.Series("", index=df.index)).fillna("").astype(str).str[:60]
    temp = pd.to_numeric(
        df.get("testing_temperature_k", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).round(0)
    ylog = pd.Series(np.log1p(np.clip(y_raw, 0, None)))
    cluster = comp.astype(str) + "@" + temp.astype(str)
    valid = (comp != "") & temp.notna()

    within, between = [], []
    n_clusters = 0
    for c in sorted(set(cluster[valid])):
        mask = ((cluster == c) & valid).to_numpy()
        refs = ref[mask]
        if refs.nunique() < 2:
            continue
        n_clusters += 1
        means = ylog[mask].groupby(refs.to_numpy()).mean()
        between.append(float(np.std(means.to_numpy(), ddof=1)))
        stds = ylog[mask].groupby(refs.to_numpy()).std(ddof=1).dropna()
        if len(stds):
            within.append(float(stds.mean()))
    return {
        "n_replicated_conditions": n_clusters,
        "within_lab_std_log": float(np.mean(within)) if within else float("nan"),
        "between_lab_std_log": float(np.mean(between)) if between else float("nan"),
        "units": "log(1 + K) with K in MPa m^0.5",
        "note": (
            "clusters share composition and test temperature; between-lab "
            "scatter on nominally identical conditions is irreducible "
            "aleatoric uncertainty at query time"
        ),
    }


def oof_permutation_importance(
    cvplus: GroupCVPlus,
    X: np.ndarray,
    y_t: np.ndarray,
    num_features: List[str],
    cat_features: List[str],
    seed: int,
    n_repeats: int = 5,
    top: int = 15,
) -> Dict[str, object]:
    """Out-of-fold permutation importance with paired value/mask columns.

    Each numeric feature occupies a (value, mask) column pair; permuting
    them jointly avoids creating value-present/mask-absent states that
    never occur in training. Permutations happen within each fold's
    held-out rows and are scored by that fold's model, so the numbers
    are out-of-fold rather than memorization.
    """
    rng = np.random.RandomState(seed)
    n_num = len(num_features)
    folds = cvplus.fold_of_sample_

    def oof_mae(Xq: np.ndarray) -> float:
        errs = np.empty(len(y_t))
        for k, model in enumerate(cvplus.fold_models_):
            held = folds == k
            errs[held] = np.abs(y_t[held] - np.asarray(model.predict(Xq[held])).ravel())
        return float(errs.mean())

    baseline = oof_mae(X)
    names = list(num_features) + list(cat_features)
    col_sets = [[j, n_num + j] for j in range(n_num)] + [
        [2 * n_num + i] for i in range(len(cat_features))
    ]
    importances: Dict[str, float] = {}
    for name, cols in zip(names, col_sets):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            for k in range(len(cvplus.fold_models_)):
                idx = np.flatnonzero(folds == k)
                perm = rng.permutation(idx)
                Xp[np.ix_(idx, cols)] = X[np.ix_(perm, cols)]
            deltas.append(oof_mae(Xp) - baseline)
        importances[name] = float(np.mean(deltas))
    ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return {
        "baseline_oof_mae_transformed": baseline,
        "note": (
            "increase in out-of-fold MAE (transformed target units) when the "
            "feature's value+mask pair is permuted within held-out folds; "
            "correlated features dilute each other"
        ),
        "top": [[name, delta] for name, delta in ranked],
    }


def qualify_from_config(cfg: Dict) -> str:
    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    target = data_cfg["target"]
    conf_cfg = cfg.get("conformal", {})
    n_folds = conf_cfg.get("n_folds", 8)
    n_splits = conf_cfg.get("eval_splits", 8)
    selection_seeds = conf_cfg.get("selection_seeds", 3)

    df = pd.read_csv(data_cfg["train_csv"])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset")
    df = df[pd.to_numeric(df[target], errors="coerce").notna()].reset_index(drop=True)
    y_raw = df[target].to_numpy(dtype=np.float64)

    num_features = prune_numeric_features(df, list(data_cfg.get("numerical_features", [])), cfg)
    cat_features = [c for c in data_cfg.get("categorical_features", []) if c in df.columns]
    groups = build_group_key(df, cfg)
    bins = compute_bins(df)

    standardize = bool(data_cfg.get("target_standardize", False))
    artifacts = fit_preprocessor(
        df, num_features, cat_features, y_raw,
        data_cfg.get("target_transform", "none"), standardize,
    )
    X = build_matrix(df, num_features, cat_features, artifacts)
    y_t = transform_target(y_raw, artifacts, standardize)
    inv = make_inverse_transform(artifacts, standardize)

    candidates = cfg.get("model", {}).get("auto", {}).get("candidates") or [
        "gbdt", "extra_trees", "blend", "random_forest", "ridge",
    ]

    # --- deployed model: selection on full data, pre-committed Mondrian
    best_name, per_seed = select_candidate(
        candidates, X, y_t, groups, cfg, n_folds, selection_seeds
    )
    factory = SkModelFactory(best_name, cfg)
    conformal_model = MondrianCVPlus(factory, n_folds=n_folds, seed=cfg["seed"]).fit(
        X, y_t, groups, bins
    )

    # --- selection-inclusive nested calibration evidence
    evidence = nested_evidence(
        df, X, y_t, groups, bins, cfg, candidates, inv,
        n_splits=n_splits, n_folds=n_folds, selection_seeds=selection_seeds,
    )
    reference = reference_subsampled_evidence(
        factory, X, y_t, groups, inv, seed=cfg["seed"]
    )
    scatter = replicate_scatter(df, y_raw)
    importance = oof_permutation_importance(
        conformal_model.pooled_, X, y_t, num_features, cat_features, seed=cfg["seed"]
    )

    # --- applicability domain with provenance metadata
    meta_cols = [c for c in METADATA_CANDIDATES if c in df.columns]
    meta = df[meta_cols].copy()
    meta["measured_" + target] = y_raw
    trust = TrustModel(n_neighbors=conf_cfg.get("trust_neighbors", 5)).fit(X, metadata=meta)

    # --- save artifact
    run_dir = os.path.join(cfg["outputs"]["run_dir"], f"qualify_{timestamp()}")
    ensure_dir(run_dir)
    joblib.dump(conformal_model, os.path.join(run_dir, "conformal_model.joblib"))
    joblib.dump(trust.to_dict(), os.path.join(run_dir, "trust.joblib"))
    save_artifacts(
        artifacts,
        scaler_path=os.path.join(run_dir, cfg["outputs"]["scaler"]),
        encoder_path=os.path.join(run_dir, cfg["outputs"]["encoder"]),
    )
    save_json(
        {
            "numerical_features": num_features,
            "categorical_features": cat_features,
            "metadata_columns": meta_cols,
            "target": target,
            "target_standardize": standardize,
            "mondrian_bin_source": "phase (brittle vs ductile class, src.physics.brittleness_bin)",
        },
        os.path.join(run_dir, "features.json"),
    )

    bin_counts = pd.Series(bins).value_counts().to_dict()
    model_card = {
        "tool": "Fracture Toughness Qualification Suite (FTQS)",
        "created": timestamp(),
        "training_data": {
            "path": data_cfg["train_csv"],
            "sha256_16": _dataset_fingerprint(data_cfg["train_csv"]),
            "n_specimens": int(len(df)),
            "n_groups": int(len(np.unique(groups))),
            "target": target,
            "target_range": [float(np.min(y_raw)), float(np.max(y_raw))],
            "n_numerical_features": len(num_features),
            "n_categorical_features": len(cat_features),
            "phase_bins": {str(k): int(v) for k, v in bin_counts.items()},
            "measures": df.get("toughness_measure", pd.Series(dtype=str)).fillna("manual").value_counts().to_dict(),
            "geometries": df.get("test_geometry", pd.Series(dtype=str)).fillna("unknown").value_counts().to_dict(),
        },
        "model_selection": {
            "criterion": f"median OOF group-CV MAE across {selection_seeds} fold seeds (transformed target)",
            "per_seed_mae": per_seed,
            "selected": best_name,
        },
        "conformal": {
            "method": (
                "pre-committed Mondrian over group-aware CV+ (Barber et al. 2021): "
                "separate calibration for the brittle and ductile phase classes, "
                "pooled fallback for thin or unknown bins"
            ),
            "bin_rule": "fixed a priori on DBTT physics; never tuned against calibration results",
            "bins": conformal_model.bin_summary(alpha=PRIMARY_ALPHA),
            "guarantee": (
                "distribution-free coverage floor max(0, 1 - 2*alpha - c(K,n)) per bin "
                "under row exchangeability (Barber et al. 2021, Thm 4), extended "
                "heuristically to group-level folds; group-level validity is supported "
                "empirically by the held-out-publication evidence below and by the "
                "subsampled reference"
            ),
            "grouping": "source publication + composition",
        },
        "calibration_evidence": evidence,
        "reference_subsampled": reference,
        "replicate_scatter": scatter,
        "permutation_importance": importance,
        "intended_use": (
            "Screening-level fracture toughness estimation with "
            "conservative bounds for test prioritization and material "
            "down-selection. Not a substitute for ASTM E399/E1820 "
            "testing or MMPDS/CMH-17 allowables."
        ),
        "seed": cfg["seed"],
    }
    save_json(model_card, os.path.join(run_dir, "model_card.json"))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = apply_schema(load_config(args.config))
    run_dir = qualify_from_config(cfg)
    print(json.dumps({"run_dir": run_dir}, indent=2))


if __name__ == "__main__":
    # Run through the canonical module so SkModelFactory instances are
    # pickled as src.qualify.SkModelFactory, not __main__.SkModelFactory,
    # and stay loadable from src.certify.
    import src.qualify as _canonical

    _canonical.main()
