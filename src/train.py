"""Training entrypoint."""

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR

from src.config import load_config
from src.data import datasets_to_arrays, load_data, save_artifacts
from src.gbdt import save_gbdt, train_gbdt
from src.models_auto import train_auto, train_extra_trees
from src.model import FTTransformerModel
from src.schema import apply_schema
from src.utils import ensure_dir, get_device, save_json, set_seed, timestamp


def _lr_schedule_fn(warmup_epochs: int, max_epochs: int):
    def _fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return _fn


def _mixup(
    x: torch.Tensor, x_mask: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, x_mask, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    x_mask_mix = torch.minimum(x_mask, x_mask[idx])
    return x_mix, x_mask_mix, y_mix


def train_from_config(cfg: Dict) -> str:
    set_seed(cfg["seed"])
    device = get_device()

    run_root = cfg["outputs"]["run_dir"]
    run_name = f"run_{timestamp()}"
    run_dir = os.path.join(run_root, run_name)
    ensure_dir(run_dir)

    train_ds, val_ds, artifacts, num_features, cat_features = load_data(
        csv_path=cfg["data"]["train_csv"],
        target=cfg["data"]["target"],
        num_features=cfg["data"]["numerical_features"],
        cat_features=cfg["data"]["categorical_features"],
        val_split=cfg["data"]["val_split"],
        seed=cfg["seed"],
        target_transform=cfg["data"].get("target_transform", "none"),
        target_standardize=cfg["data"].get("target_standardize", False),
        group_columns=cfg["data"].get("group_columns"),
        min_non_null_fraction=cfg["data"].get("min_non_null_fraction"),
        min_known_numeric=cfg["data"].get("min_known_numeric"),
        min_non_zero_fraction=cfg["data"].get("min_non_zero_fraction"),
    )

    num_features_len = len(num_features)
    cat_cardinalities = artifacts.cat_encoder.cardinalities(cat_features)

    model_type = cfg.get("model", {}).get("type", "transformer")
    features_path = os.path.join(run_dir, "features.json")
    save_json({"numerical_features": num_features, "categorical_features": cat_features}, features_path)

    if model_type == "auto":
        x_train, y_train, x_val, y_val = datasets_to_arrays(train_ds, val_ds)
        model, model_name, model_mae = train_auto(x_train, y_train, x_val, y_val, cfg)
        best_path = os.path.join(run_dir, cfg["outputs"]["best_model"])
        save_gbdt(model, best_path)
        save_artifacts(
            artifacts,
            scaler_path=os.path.join(run_dir, cfg["outputs"]["scaler"]),
            encoder_path=os.path.join(run_dir, cfg["outputs"]["encoder"]),
        )
        metrics_path = os.path.join(run_dir, cfg["outputs"]["metrics"])
        save_json(
            {"best_val_loss": None, "model_type": model_name, "val_mae": model_mae},
            metrics_path,
        )
        return run_dir

    if model_type == "extra_trees":
        x_train, y_train, x_val, y_val = datasets_to_arrays(train_ds, val_ds)
        model = train_extra_trees(x_train, y_train, cfg)
        best_path = os.path.join(run_dir, cfg["outputs"]["best_model"])
        save_gbdt(model, best_path)
        save_artifacts(
            artifacts,
            scaler_path=os.path.join(run_dir, cfg["outputs"]["scaler"]),
            encoder_path=os.path.join(run_dir, cfg["outputs"]["encoder"]),
        )
        metrics_path = os.path.join(run_dir, cfg["outputs"]["metrics"])
        save_json({"best_val_loss": None, "model_type": "extra_trees"}, metrics_path)
        return run_dir

    if model_type == "gbdt":
        x_train, y_train, x_val, y_val = datasets_to_arrays(train_ds, val_ds)
        gbdt = train_gbdt(x_train, y_train, cfg)
        best_path = os.path.join(run_dir, cfg["outputs"]["best_model"])
        save_gbdt(gbdt, best_path)
        save_artifacts(
            artifacts,
            scaler_path=os.path.join(run_dir, cfg["outputs"]["scaler"]),
            encoder_path=os.path.join(run_dir, cfg["outputs"]["encoder"]),
        )
        metrics_path = os.path.join(run_dir, cfg["outputs"]["metrics"])
        save_json({"best_val_loss": None, "model_type": "gbdt"}, metrics_path)
        return run_dir

    model = FTTransformerModel(
        num_features=num_features_len,
        cat_cardinalities=cat_cardinalities,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
        mlp_hidden=cfg["model"]["mlp_hidden"],
        mlp_layers=cfg["model"]["mlp_layers"],
    ).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    loss_fn = nn.HuberLoss(delta=cfg["train"]["huber_delta"])
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    lr_scheduler = LambdaLR(optimizer, _lr_schedule_fn(cfg["train"]["warmup_epochs"], cfg["train"]["max_epochs"]))

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=cfg["train"]["mixed_precision"] and device.type == "cuda")

    use_swa = cfg["train"]["use_swa"]
    swa_model = AveragedModel(model) if use_swa else None
    swa_start = int(cfg["train"]["swa_start"] * cfg["train"]["max_epochs"])
    swa_scheduler = SWALR(optimizer, swa_lr=cfg["train"]["lr"] * 0.1) if use_swa else None

    best_val = float("inf")
    best_path = os.path.join(run_dir, cfg["outputs"]["best_model"])
    patience = cfg["train"]["patience"]
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg["train"]["max_epochs"]):
        model.train()
        train_losses = []
        for x_num, x_num_mask, x_cat, y in train_loader:
            x_num = x_num.to(device)
            x_num_mask = x_num_mask.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            x_num, x_num_mask, y = _mixup(
                x_num, x_num_mask, y, cfg["train"]["mixup_alpha"]
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=scaler.is_enabled()):
                preds = model(x_num, x_num_mask, x_cat)
                loss = loss_fn(preds, y)

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_num, x_num_mask, x_cat, y in val_loader:
                x_num = x_num.to(device)
                x_num_mask = x_num_mask.to(device)
                x_cat = x_cat.to(device)
                y = y.to(device)
                preds = model(x_num, x_num_mask, x_cat)
                val_losses.append(loss_fn(preds, y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            lr_scheduler.step()

        if epochs_no_improve >= patience:
            break

    if use_swa:
        torch.save(swa_model.state_dict(), best_path)

    save_artifacts(
        artifacts,
        scaler_path=os.path.join(run_dir, cfg["outputs"]["scaler"]),
        encoder_path=os.path.join(run_dir, cfg["outputs"]["encoder"]),
    )

    metrics_path = os.path.join(run_dir, cfg["outputs"]["metrics"])
    save_json({"best_val_loss": best_val, "model_type": "transformer"}, metrics_path)
    history_path = os.path.join(run_dir, "history.json")
    save_json(history, history_path)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_schema(cfg)
    run_dir = train_from_config(cfg)
    print(json.dumps({"run_dir": run_dir}, indent=2))


if __name__ == "__main__":
    main()
