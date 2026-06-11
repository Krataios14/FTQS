from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.certify import certify_dataframe, load_run
from src.prepare_data import METADATA_COLS, prepare_dataframe
from src.qualify import qualify_from_config
from src.report import render_report

ROOT = Path(__file__).resolve().parents[1]
TARGET = "fracture_toughness_mpa_m0_5"


def _feature_lists(df: pd.DataFrame):
    num = [
        c
        for c in df.columns
        if c.startswith("elem_")
        or c.startswith("phys_")
        or any(x in c for x in ["_um", "_g_cm3", "_gpa", "_mpa", "_percent", "_k"])
    ]
    if TARGET in num:
        num.remove(TARGET)
    meta = [c for c in METADATA_COLS if c in df.columns]
    cat = [c for c in df.columns if c not in num + meta + [TARGET]]
    return sorted(num), sorted(cat)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("toughcert")
    raw_train = pd.read_csv(ROOT / "assets" / "combined_fracture_training.csv", skip_blank_lines=True)
    raw_unseen = pd.read_csv(ROOT / "assets" / "combined_fracture_unseen.csv", skip_blank_lines=True)

    train_df, _, _ = prepare_dataframe(raw_train, TARGET, [])
    unseen_df, _, _ = prepare_dataframe(raw_unseen, TARGET, [])
    all_cols = sorted(set(train_df.columns) | set(unseen_df.columns))
    train_df = train_df.reindex(columns=all_cols)
    unseen_df = unseen_df.reindex(columns=all_cols)
    train_csv = tmp / "train.csv"
    train_df.to_csv(train_csv, index=False)

    num, cat = _feature_lists(train_df)
    cfg = {
        "seed": 42,
        "data": {
            "train_csv": str(train_csv),
            "target": TARGET,
            "numerical_features": num,
            "categorical_features": cat,
            "target_transform": "log1p",
            "target_standardize": True,
            "min_non_null_fraction": 0.05,
            "min_non_zero_fraction": 0.02,
        },
        "model": {"auto": {"candidates": ["ridge", "gbdt"]}},
        "conformal": {"n_folds": 5, "eval_splits": 3, "trust_neighbors": 5},
        "outputs": {
            "run_dir": str(tmp / "runs"),
            "scaler": "scaler.joblib",
            "encoder": "encoder.joblib",
        },
    }
    run_dir = qualify_from_config(cfg)
    return tmp, run_dir, unseen_df


def test_qualify_writes_artifact(pipeline):
    _, run_dir, _ = pipeline
    run_dir = Path(run_dir)
    for name in [
        "conformal_model.joblib",
        "trust.joblib",
        "scaler.joblib",
        "encoder.joblib",
        "features.json",
        "model_card.json",
    ]:
        assert (run_dir / name).exists(), name


def test_model_card_contents(pipeline):
    _, run_dir, _ = pipeline
    run = load_run(run_dir)
    card = run["model_card"]
    assert card["training_data"]["n_specimens"] > 100
    assert card["training_data"]["n_groups"] > 10
    assert card["model_selection"]["selected"] in {"ridge", "gbdt"}
    ev = card["calibration_evidence"]["alpha_0.10"]
    # Finite-sample guarantee is >= 0.8; allow slack for small eval splits
    assert ev["empirical_coverage"] >= 0.7
    assert ev["mean_interval_width"] > 0


def test_certify_outputs(pipeline):
    tmp, run_dir, unseen_df = pipeline
    run = load_run(run_dir)
    out, blocks = certify_dataframe(unseen_df, run)
    assert len(out) == len(unseen_df)
    for col in [
        "predicted_toughness_mpa_m0_5",
        "lower_90",
        "upper_90",
        "lower_95",
        "upper_95",
        "trust_score",
        "trust_tier",
        "nearest_training_anchors",
    ]:
        assert col in out.columns, col
    assert (out["lower_90"] <= out["upper_90"]).all()
    # 95% bounds are at least as wide as 90% bounds
    assert (out["lower_95"] <= out["lower_90"] + 1e-9).all()
    assert (out["upper_95"] >= out["upper_90"] - 1e-9).all()
    assert (out["lower_90"] >= 0).all()
    assert out["trust_tier"].isin(["A", "B", "C"]).all()
    # Provenance anchors carry source citations
    assert out["nearest_training_anchors"].str.len().gt(5).all()
    assert len(blocks) == len(out)


def test_report_renders(pipeline):
    tmp, run_dir, unseen_df = pipeline
    run = load_run(run_dir)
    out, blocks = certify_dataframe(unseen_df, run)
    path = render_report(str(tmp / "report.html"), out, run["model_card"], blocks)
    text = Path(path).read_text(encoding="utf-8")
    assert "ToughCert" in text
    assert "data:image/png;base64," in text
    assert "Intended use" in text
    assert len(text) > 20000  # embedded figures present


def test_certify_handles_raw_columns(pipeline):
    """Certify accepts a frame missing physics columns and rebuilds them."""
    tmp, run_dir, unseen_df = pipeline
    run = load_run(run_dir)
    stripped = unseen_df.drop(columns=[c for c in unseen_df.columns if c.startswith("phys_")])
    out, _ = certify_dataframe(stripped, run)
    assert np.isfinite(out["predicted_toughness_mpa_m0_5"]).all()
