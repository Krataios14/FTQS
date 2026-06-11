"""Batch certification: predictions with guarantees, trust and provenance.

Takes a qualification artifact (from `python -m src.qualify`) and a CSV
of candidate materials, and emits:

- a predictions CSV: point estimate, conformal 90% and 95% intervals
  (finite-sample-valid lower/upper bounds), trust score and tier, and
  the nearest training specimens with their source publications
- a self-contained HTML qualification report for design review

Usage:
    python -m src.certify --run runs/qualify_YYYYMMDD_HHMMSS \
        --data data/processed_unseen.csv --out predictions.csv \
        --report reports/qualification_report.html
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.applicability import TrustModel
from src.conformal import MondrianCVPlus
from src.data import load_artifacts
from src.physics import add_physics_features
from src.qualify import build_matrix, compute_bins, make_inverse_transform

IDENTITY_COLS = [
    "composition_at_percent",
    "material_condition",
    "phase",
    "testing_temperature_k",
]


def latest_qualify_run(run_root: str = "runs") -> str:
    runs = sorted(glob.glob(os.path.join(run_root, "qualify_*")))
    if not runs:
        raise FileNotFoundError(
            f"No qualify_* artifact found under '{run_root}'. Run `python -m src.qualify` first."
        )
    return runs[-1]


def load_run(run_dir: str) -> Dict:
    with open(os.path.join(run_dir, "features.json"), "r", encoding="utf-8") as f:
        features = json.load(f)
    with open(os.path.join(run_dir, "model_card.json"), "r", encoding="utf-8") as f:
        model_card = json.load(f)
    artifacts = load_artifacts(
        os.path.join(run_dir, "scaler.joblib"), os.path.join(run_dir, "encoder.joblib")
    )
    conformal_model = joblib.load(os.path.join(run_dir, "conformal_model.joblib"))
    trust = TrustModel.from_dict(joblib.load(os.path.join(run_dir, "trust.joblib")))
    return {
        "features": features,
        "model_card": model_card,
        "artifacts": artifacts,
        "conformal": conformal_model,
        "trust": trust,
    }


def _ensure_columns(df: pd.DataFrame, features: Dict) -> pd.DataFrame:
    df = df.copy()
    needed_phys = [c for c in features["numerical_features"] if c.startswith("phys_")]
    if needed_phys and not all(c in df.columns for c in needed_phys):
        df, _ = add_physics_features(df)
    for col in features["numerical_features"]:
        if col not in df.columns:
            df[col] = np.nan
    for col in features["categorical_features"]:
        if col not in df.columns:
            df[col] = "UNK"
    return df


def certify_dataframe(
    df: pd.DataFrame,
    run: Dict,
    alphas: Tuple[float, ...] = (0.10, 0.05),
    k_neighbors: int = 3,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    features = run["features"]
    artifacts = run["artifacts"]
    df = _ensure_columns(df, features)
    X = build_matrix(
        df, features["numerical_features"], features["categorical_features"], artifacts
    )
    inv = make_inverse_transform(artifacts, features.get("target_standardize", False))

    model = run["conformal"]
    out = pd.DataFrame(index=df.index)
    for col in IDENTITY_COLS:
        if col in df.columns:
            out[col] = df[col]
    if isinstance(model, MondrianCVPlus):
        bins = compute_bins(df)
        out["phase_bin"] = bins
        preds = inv(model.predict(X, bins))
        intervals = model.predict_interval_multi(X, list(alphas), bins)
    else:
        preds = inv(model.predict(X))
        intervals = model.predict_interval_multi(X, list(alphas))
    out["predicted_toughness_mpa_m0_5"] = preds
    unbounded = np.zeros(len(df), dtype=bool)
    for alpha in alphas:
        lo, hi = intervals[alpha]
        level = int(round((1 - alpha) * 100))
        lo_inv, hi_inv = inv(lo), inv(hi)
        unbounded |= ~np.isfinite(lo_inv) | ~np.isfinite(hi_inv)
        out[f"lower_{level}"] = np.where(np.isfinite(lo_inv), np.maximum(lo_inv, 0.0), 0.0)
        out[f"upper_{level}"] = hi_inv
    if unbounded.any():
        out["interval_unbounded"] = unbounded

    trust_df = run["trust"].score(X)
    out = pd.concat([out, trust_df[["trust_score", "trust_tier"]]], axis=1)

    neighbor_blocks = run["trust"].neighbors(X, k=k_neighbors)
    anchors = []
    for block in neighbor_blocks:
        refs = []
        for _, row in block.iterrows():
            ref = str(row.get("reference", ""))[:80]
            comp = str(row.get("composition_at_percent", ""))[:40]
            refs.append(f"{comp} [{ref}]" if comp and comp != "nan" else f"[{ref}]")
        anchors.append(" ; ".join(refs))
    out["nearest_training_anchors"] = anchors

    if features["target"] in df.columns:
        measured = pd.to_numeric(df[features["target"]], errors="coerce")
        if measured.notna().any():
            out["measured_toughness_mpa_m0_5"] = measured
    return out, neighbor_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default=None, help="qualify_* artifact dir (default: latest)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="predictions.csv")
    parser.add_argument("--report", type=str, default=None, help="optional HTML report path")
    parser.add_argument("--neighbors", type=int, default=3)
    args = parser.parse_args()

    run_dir = args.run or latest_qualify_run()
    run = load_run(run_dir)
    df = pd.read_csv(args.data, skip_blank_lines=True)
    out, neighbor_blocks = certify_dataframe(df, run, k_neighbors=args.neighbors)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(json.dumps({"run_dir": run_dir, "predictions": args.out, "n": len(out)}, indent=2))

    if args.report:
        from src.report import render_report

        render_report(args.report, out, run["model_card"], neighbor_blocks)
        print(json.dumps({"report": args.report}, indent=2))


if __name__ == "__main__":
    main()
