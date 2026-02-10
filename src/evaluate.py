"""Evaluation utilities."""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import inverse_transform_target, load_artifacts, transform_df, transform_target, TabularDataset
from src.gbdt import load_gbdt
from src.model import FTTransformerModel
from src.schema import apply_schema
from src.utils import get_device, load_state_dict_flexible, save_json


def _mae_from_arrays(
    model: torch.nn.Module,
    x_num: np.ndarray,
    x_num_mask: np.ndarray,
    x_cat: np.ndarray,
    y_true: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    if len(y_true) == 0:
        return 0.0
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(y_true), batch_size):
            xb_num = torch.tensor(x_num[i : i + batch_size], dtype=torch.float32, device=device)
            xb_mask = torch.tensor(x_num_mask[i : i + batch_size], dtype=torch.float32, device=device)
            xb_cat = torch.tensor(x_cat[i : i + batch_size], dtype=torch.long, device=device)
            preds.append(model(xb_num, xb_mask, xb_cat).cpu().numpy())
    preds = np.vstack(preds).squeeze()
    return float(mean_absolute_error(y_true, preds))


def _permutation_importance(
    model: torch.nn.Module,
    x_num: np.ndarray,
    x_num_mask: np.ndarray,
    x_cat: np.ndarray,
    y_true: np.ndarray,
    num_features: List[str],
    cat_features: List[str],
    device: torch.device,
    batch_size: int,
) -> List[Dict]:
    baseline = _mae_from_arrays(model, x_num, x_num_mask, x_cat, y_true, device, batch_size)
    importances = []

    for i, name in enumerate(num_features):
        x_num_perm = x_num.copy()
        rng = np.random.default_rng(0)
        rng.shuffle(x_num_perm[:, i])
        mae = _mae_from_arrays(model, x_num_perm, x_num_mask, x_cat, y_true, device, batch_size)
        importances.append({"feature": name, "type": "numerical", "mae_increase": mae - baseline})

    for i, name in enumerate(cat_features):
        x_cat_perm = x_cat.copy()
        rng = np.random.default_rng(0)
        rng.shuffle(x_cat_perm[:, i])
        mae = _mae_from_arrays(model, x_num, x_num_mask, x_cat_perm, y_true, device, batch_size)
        importances.append({"feature": name, "type": "categorical", "mae_increase": mae - baseline})

    importances.sort(key=lambda x: x["mae_increase"], reverse=True)
    return importances


def evaluate_from_config(cfg: Dict, run_dir: str) -> Dict:
    device = get_device()

    df = pd.read_csv(cfg["data"]["train_csv"])
    target = cfg["data"]["target"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataset.")
    y = df[target].to_numpy(dtype=np.float32)
    df = df.drop(columns=[target])

    group_columns = cfg["data"].get("group_columns") or []
    valid_groups = [c for c in group_columns if c in df.columns]
    if valid_groups:
        group_key = (
            df[valid_groups]
            .astype(str)
            .fillna("UNK")
            .agg(lambda row: "|".join(row.astype(str)), axis=1)
        )
        gss = GroupShuffleSplit(
            n_splits=1, test_size=cfg["data"]["val_split"], random_state=cfg["seed"]
        )
        _, val_idx = next(gss.split(df, y, groups=group_key))
        val_df = df.iloc[val_idx]
        y_val = y[val_idx]
    else:
        _, val_df, _, y_val = train_test_split(
            df, y, test_size=cfg["data"]["val_split"], random_state=cfg["seed"]
        )

    artifacts = load_artifacts(
        scaler_path=os.path.join(run_dir, cfg["outputs"]["scaler"]),
        encoder_path=os.path.join(run_dir, cfg["outputs"]["encoder"]),
    )

    num_features = cfg["data"]["numerical_features"]
    cat_features = cfg["data"]["categorical_features"]
    features_path = os.path.join(run_dir, "features.json")
    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        num_features = features.get("numerical_features", num_features)
        cat_features = features.get("categorical_features", cat_features)
    x_num_val, x_num_mask_val, x_cat_val = transform_df(
        val_df, num_features, cat_features, artifacts
    )
    y_val_t = transform_target(
        y_val,
        artifacts,
        cfg["data"].get("target_standardize", False),
    )
    val_ds = TabularDataset(x_num_val, x_num_mask_val, x_cat_val, y_val_t)

    cat_cardinalities = artifacts.cat_encoder.cardinalities(cat_features)

    model_type = cfg.get("model", {}).get("type", "transformer")
    model_path = os.path.join(run_dir, cfg["outputs"]["best_model"])

    if model_type in ("gbdt", "auto", "extra_trees"):
        model = load_gbdt(model_path)
        x_val = np.concatenate(
            [
                val_ds.x_num.numpy(),
                val_ds.x_num_mask.numpy(),
                val_ds.x_cat.numpy().astype(np.float32),
            ],
            axis=1,
        )
        preds = model.predict(x_val).squeeze()
        targets = val_ds.y.numpy().squeeze()
    else:
        model = FTTransformerModel(
            num_features=len(num_features),
            cat_cardinalities=cat_cardinalities,
            d_model=cfg["model"]["d_model"],
            n_heads=cfg["model"]["n_heads"],
            n_layers=cfg["model"]["n_layers"],
            dropout=cfg["model"]["dropout"],
            mlp_hidden=cfg["model"]["mlp_hidden"],
            mlp_layers=cfg["model"]["mlp_layers"],
        ).to(device)

        model.load_state_dict(load_state_dict_flexible(model_path, device))
        model.eval()

        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

        preds = []
        targets = []
        with torch.no_grad():
            for x_num, x_num_mask, x_cat, y in val_loader:
                x_num = x_num.to(device)
                x_num_mask = x_num_mask.to(device)
                x_cat = x_cat.to(device)
                y = y.to(device)
                pred = model(x_num, x_num_mask, x_cat)
                preds.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())

        preds = np.vstack(preds).squeeze()
        targets = np.vstack(targets).squeeze()

    preds = inverse_transform_target(
        preds, artifacts, cfg["data"].get("target_standardize", False)
    )
    targets = inverse_transform_target(
        targets, artifacts, cfg["data"].get("target_standardize", False)
    )

    metrics = {
        "mae": float(mean_absolute_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "r2": float(r2_score(targets, preds)),
    }

    if model_type == "transformer":
        importances = _permutation_importance(
            model,
            val_ds.x_num.numpy(),
            val_ds.x_num_mask.numpy(),
            val_ds.x_cat.numpy(),
            (val_ds.y.numpy().squeeze()),
            num_features,
            cat_features,
            device,
            cfg["train"]["batch_size"],
        )
        metrics["permutation_importance"] = importances

    preds_path = os.path.join(run_dir, "predictions.csv")
    pd.DataFrame({"y_true": targets, "y_pred": preds}).to_csv(preds_path, index=False)

    metrics_path = os.path.join(run_dir, cfg["outputs"]["metrics"])
    save_json(metrics, metrics_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_schema(cfg)
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = cfg["outputs"]["run_dir"]
        run_dir = os.path.join(run_dir, sorted(os.listdir(run_dir))[-1])

    metrics = evaluate_from_config(cfg, run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
