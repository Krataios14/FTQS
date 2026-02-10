"""Inference entrypoint."""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data import inverse_transform_target, load_artifacts, transform_df
from src.gbdt import load_gbdt
from src.model import FTTransformerModel
from src.schema import apply_schema
from src.utils import get_device, load_state_dict_flexible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, default="predictions.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_schema(cfg)
    device = get_device()

    run_dir = os.path.dirname(args.model)
    scaler_path = os.path.join(run_dir, cfg["outputs"]["scaler"])
    encoder_path = os.path.join(run_dir, cfg["outputs"]["encoder"])

    artifacts = load_artifacts(scaler_path, encoder_path)
    num_features = cfg["data"]["numerical_features"]
    cat_features = cfg["data"]["categorical_features"]
    features_path = os.path.join(run_dir, "features.json")
    if os.path.exists(features_path):
        import json

        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        num_features = features.get("numerical_features", num_features)
        cat_features = features.get("categorical_features", cat_features)

    df = pd.read_csv(args.data)
    for col in num_features:
        if col not in df.columns:
            df[col] = float("nan")
    for col in cat_features:
        if col not in df.columns:
            df[col] = "UNK"
    x_num, x_num_mask, x_cat = transform_df(df, num_features, cat_features, artifacts)

    cat_cardinalities = artifacts.cat_encoder.cardinalities(cat_features)
    model_type = cfg.get("model", {}).get("type", "transformer")
    if model_type in ("gbdt", "auto", "extra_trees"):
        gbdt = load_gbdt(args.model)
        x = np.concatenate(
            [x_num, x_num_mask, x_cat.astype(np.float32)],
            axis=1,
        )
        preds = gbdt.predict(x).squeeze()
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
        model.load_state_dict(load_state_dict_flexible(args.model, device))
        model.eval()

        with torch.no_grad():
            preds = model(
                torch.tensor(x_num, dtype=torch.float32, device=device),
                torch.tensor(x_num_mask, dtype=torch.float32, device=device),
                torch.tensor(x_cat, dtype=torch.long, device=device),
            ).cpu().numpy().squeeze()

    preds = inverse_transform_target(
        preds, artifacts, cfg["data"].get("target_standardize", False)
    )

    out_df = df.copy()
    out_df["fracture_toughness_pred"] = preds
    out_df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
