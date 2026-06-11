"""Train a qualification-grade conformal model and emit its artifact.

One command takes the processed training table to a self-contained
artifact directory containing:

- conformal_model.joblib : group-aware CV+ ensemble around the best
  base regressor (selected by out-of-fold group-CV MAE)
- trust.joblib           : applicability-domain model with provenance
- scaler/encoder.joblib  : preprocessing artifacts
- features.json          : exact feature lists used
- model_card.json        : dataset summary, selection results and
  held-out-group calibration evidence (empirical coverage vs nominal)

Usage:
    python -m src.qualify --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from src.applicability import TrustModel
from src.config import load_config
from src.conformal import GroupCVPlus, evaluate_group_coverage
from src.data import fit_preprocessor, transform_df, transform_target, save_artifacts
from src.schema import apply_schema
from src.utils import ensure_dir, save_json, set_seed, timestamp

METADATA_CANDIDATES = [
    "composition_at_percent",
    "reference",
    "material_condition",
    "phase",
    "processing_history",
    "testing_temperature_k",
]


class SkModelFactory:
    """Picklable zero-arg factory for the candidate regressors."""

    def __init__(self, name: str, cfg: Dict):
        self.name = name
        self.seed = int(cfg.get("seed", 42))
        self.gbdt_params = dict(cfg.get("model", {}).get("gbdt", {}))
        self.et_params = dict(cfg.get("model", {}).get("extra_trees", {}))

    def __call__(self):
        if self.name == "gbdt":
            p = self.gbdt_params
            return HistGradientBoostingRegressor(
                learning_rate=p.get("learning_rate", 0.05),
                max_depth=p.get("max_depth", None),
                max_leaf_nodes=p.get("max_leaf_nodes", 63),
                max_bins=p.get("max_bins", 255),
                min_samples_leaf=p.get("min_samples_leaf", 20),
                l2_regularization=p.get("l2_regularization", 0.1),
                max_iter=p.get("max_iter", 500),
                early_stopping=p.get("early_stopping", True),
                random_state=self.seed,
            )
        if self.name == "extra_trees":
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
    anti-conservative.
    """

    def _inv(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if standardize:
            y = y * artifacts.target_std + artifacts.target_mean
        if artifacts.target_transform == "log1p":
            y = np.expm1(np.clip(y, -700.0, 700.0))
        return y

    return _inv


def _dataset_fingerprint(csv_path: str) -> str:
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def qualify_from_config(cfg: Dict) -> str:
    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    target = data_cfg["target"]

    df = pd.read_csv(data_cfg["train_csv"])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset")
    df = df[pd.to_numeric(df[target], errors="coerce").notna()].reset_index(drop=True)
    y_raw = df[target].to_numpy(dtype=np.float64)

    num_features = prune_numeric_features(df, list(data_cfg.get("numerical_features", [])), cfg)
    cat_features = [c for c in data_cfg.get("categorical_features", []) if c in df.columns]
    groups = build_group_key(df, cfg)

    standardize = bool(data_cfg.get("target_standardize", False))
    artifacts = fit_preprocessor(
        df, num_features, cat_features, y_raw,
        data_cfg.get("target_transform", "none"), standardize,
    )
    X = build_matrix(df, num_features, cat_features, artifacts)
    y_t = transform_target(y_raw, artifacts, standardize)
    inv = make_inverse_transform(artifacts, standardize)

    # --- model selection by out-of-fold group-CV MAE (transformed space)
    candidates = cfg.get("model", {}).get("auto", {}).get("candidates") or [
        "gbdt", "extra_trees", "random_forest", "ridge",
    ]
    selection: Dict[str, float] = {}
    fitted: Dict[str, GroupCVPlus] = {}
    for name in candidates:
        factory = SkModelFactory(name, cfg)
        model = GroupCVPlus(factory, n_folds=cfg.get("conformal", {}).get("n_folds", 8), seed=cfg["seed"])
        model.fit(X, y_t, groups=groups)
        selection[name] = float(np.mean(model.residuals_))
        fitted[name] = model
    best_name = min(selection, key=selection.get)
    conformal_model = fitted[best_name]

    # --- held-out-group calibration evidence in original units
    evidence = {}
    for alpha in (0.10, 0.05):
        evidence[f"alpha_{alpha:.2f}"] = evaluate_group_coverage(
            SkModelFactory(best_name, cfg),
            X, y_t, groups,
            alpha=alpha,
            n_splits=cfg.get("conformal", {}).get("eval_splits", 8),
            seed=cfg["seed"],
            inverse_transform=inv,
        )

    # --- applicability domain with provenance metadata
    meta_cols = [c for c in METADATA_CANDIDATES if c in df.columns]
    meta = df[meta_cols].copy()
    meta["measured_" + target] = y_raw
    trust = TrustModel(
        n_neighbors=cfg.get("conformal", {}).get("trust_neighbors", 5)
    ).fit(X, metadata=meta)

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
        },
        os.path.join(run_dir, "features.json"),
    )

    model_card = {
        "tool": "ToughCert",
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
        },
        "model_selection": {
            "criterion": "out-of-fold group-CV MAE (transformed target)",
            "candidates": selection,
            "selected": best_name,
        },
        "conformal": {
            "method": "group-aware CV+ (Barber et al. 2021)",
            "n_folds": len(conformal_model.fold_models_),
            "grouping": "source publication + composition",
        },
        "calibration_evidence": evidence,
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
    main()
