import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGET = "fracture_toughness_mpa_m0_5"


def feature_lists(df: pd.DataFrame):
    from src.prepare_data import METADATA_COLS

    meta = [c for c in METADATA_COLS if c in df.columns]
    num = [
        c
        for c in df.columns
        if c not in meta
        and (
            c.startswith("elem_")
            or c.startswith("phys_")
            or any(x in c for x in ["_um", "_g_cm3", "_gpa", "_mpa", "_percent", "_k"])
        )
    ]
    if TARGET in num:
        num.remove(TARGET)
    cat = [c for c in df.columns if c not in num + meta + [TARGET]]
    return sorted(num), sorted(cat)


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """Prepare the real assets and train a small qualification artifact."""
    from src.prepare_data import prepare_dataframe
    from src.qualify import qualify_from_config

    tmp = tmp_path_factory.mktemp("toughcert")
    raw_train = pd.read_csv(
        ROOT / "assets" / "combined_fracture_training.csv", skip_blank_lines=True
    )
    raw_unseen = pd.read_csv(
        ROOT / "assets" / "combined_fracture_unseen.csv", skip_blank_lines=True
    )

    train_df, _, _ = prepare_dataframe(raw_train, TARGET, [])
    unseen_df, _, _ = prepare_dataframe(raw_unseen, TARGET, [])
    all_cols = sorted(set(train_df.columns) | set(unseen_df.columns))
    train_df = train_df.reindex(columns=all_cols)
    unseen_df = unseen_df.reindex(columns=all_cols)
    train_csv = tmp / "train.csv"
    train_df.to_csv(train_csv, index=False)

    num, cat = feature_lists(train_df)
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
        "conformal": {
            "n_folds": 5,
            "eval_splits": 3,
            "trust_neighbors": 5,
            "selection_seeds": 1,
        },
        "outputs": {
            "run_dir": str(tmp / "runs"),
            "scaler": "scaler.joblib",
            "encoder": "encoder.joblib",
        },
    }
    run_dir = qualify_from_config(cfg)
    return tmp, run_dir, unseen_df
