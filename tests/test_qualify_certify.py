import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.certify import certify_dataframe, load_run
from src.report import render_report

ROOT = Path(__file__).resolve().parents[1]


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
    ev = card["calibration_evidence"]
    e90 = ev["per_alpha"]["alpha_0.10"]
    assert e90["empirical_coverage"] >= 0.7
    assert e90["mean_width"] > 0
    # Honest floor: strictly below the often-quoted 1 - 2*alpha
    assert 0 < e90["provable_floor_rowlevel"] < 0.80
    lo_ci, hi_ci = e90["coverage_ci95"]
    assert lo_ci <= e90["empirical_coverage"] <= hi_ci
    # Audit exists at the pre-registered alpha and flags thin strata
    audit = ev["audit"]
    assert "phase_bin" in audit and "geometry" in audit
    for levels in audit.values():
        for entry in levels.values():
            assert {"n", "n_groups", "covered", "insufficient"} <= set(entry)
    # Selection-inclusive protocol recorded per split
    assert len(ev["selected_per_split"]) == ev["n_splits"]
    # Mondrian bins with per-bin honest floors
    bins = card["conformal"]["bins"]
    assert "__pooled__" in bins
    assert all(0 <= b["provable_floor"] < 0.8 for b in bins.values())
    # Replicate scatter decomposition present and physically sane
    scatter = card["replicate_scatter"]
    assert scatter["between_lab_any_condition"]["n_clusters"] >= 3
    assert scatter["between_lab_any_condition"]["std_log"] > 0.1
    # Pure repeatability must be smaller than the cross-lab spread
    assert (
        scatter["within_lab"]["std_log"]
        < scatter["between_lab_any_condition"]["std_log"]
    )
    # OOF permutation importance ranked
    top = card["permutation_importance"]["top"]
    assert len(top) >= 5
    assert top[0][1] >= top[-1][1]
    # Subsampled hierarchical reference present
    assert 0 <= card["reference_subsampled"]["empirical_coverage"] <= 1


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
    assert "FTQS" in text
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


def test_cli_artifact_loads_across_processes(pipeline, tmp_path):
    """Regression: an artifact written by `python -m src.qualify` must be
    loadable from another process. Module-as-__main__ pickling broke this."""
    tmp, _, _ = pipeline
    from tests.conftest import TARGET, feature_lists

    train_csv = tmp / "train.csv"
    num, cat = feature_lists(pd.read_csv(train_csv))
    cfg = {
        "seed": 0,
        "data": {
            "train_csv": str(train_csv),
            "target": TARGET,
            "numerical_features": num,
            "categorical_features": cat,
            "target_transform": "log1p",
            "target_standardize": True,
        },
        "model": {"auto": {"candidates": ["ridge"]}},
        "conformal": {"n_folds": 3, "eval_splits": 1},
        "outputs": {
            "run_dir": str(tmp_path / "runs"),
            "scaler": "scaler.joblib",
            "encoder": "encoder.joblib",
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "src.qualify", "--config", str(cfg_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    run_dir = json.loads(proc.stdout)["run_dir"]
    run = load_run(str(ROOT / run_dir) if not Path(run_dir).is_absolute() else run_dir)
    assert run["model_card"]["model_selection"]["selected"] == "ridge"
