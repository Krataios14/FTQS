"""Prepare raw fracture toughness data into model-ready form."""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

from src.physics import add_physics_features, parse_composition


COMPOSITION_COL = "composition_at_percent"
# Provenance and measurement metadata: carried through for traceability,
# never used as model features.
METADATA_COLS = ["reference", COMPOSITION_COL, "toughness_uncertainty_mpa_m0_5"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = []
    for c in df.columns:
        s = c.strip().lower()
        s = s.replace("%", "percent")
        s = s.replace("(", " ").replace(")", " ")
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        cols.append(s)
    df.columns = cols
    return df


def _expand_composition(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, List[str]]:
    parsed = df[col].apply(parse_composition)
    elements = sorted({k for d in parsed for k in d.keys()})
    comp_df = pd.DataFrame([{e: d.get(e, 0.0) for e in elements} for d in parsed])
    comp_df.columns = [f"elem_{e.lower()}" for e in comp_df.columns]
    # Keep the raw composition string as provenance metadata.
    df = pd.concat([df.reset_index(drop=True), comp_df.reset_index(drop=True)], axis=1)
    return df, list(comp_df.columns)


def prepare_dataframe(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
    physics: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = _normalize_columns(df)
    df = df.dropna(how="all")

    if COMPOSITION_COL in df.columns:
        df, elem_cols = _expand_composition(df, COMPOSITION_COL)
    else:
        elem_cols = []

    phys_cols: List[str] = []
    if physics:
        df, phys_cols = add_physics_features(df, composition_col=COMPOSITION_COL)

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Coerce known numeric columns to numeric (metadata stays as text)
    for col in df.columns:
        if col in METADATA_COLS:
            continue
        if col == target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif col.startswith("elem_") or col.startswith("phys_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif any(x in col for x in ["_um", "_g_cm3", "_gpa", "_mpa", "_percent", "_k"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, elem_cols, phys_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--unseen", type=str, required=True)
    parser.add_argument("--out_train", type=str, required=True)
    parser.add_argument("--out_unseen", type=str, required=True)
    parser.add_argument("--schema", type=str, default="data/schema.json")
    parser.add_argument("--target", type=str, default="fracture_toughness_mpa_m0_5")
    parser.add_argument("--no-physics", action="store_true", help="Skip physics-informed descriptors")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, skip_blank_lines=True)
    unseen_df = pd.read_csv(args.unseen, skip_blank_lines=True)

    physics = not args.no_physics
    # Reference and the raw composition string are kept as provenance
    # metadata; they are excluded from the feature lists below.
    train_df, elem_cols, phys_cols = prepare_dataframe(train_df, args.target, [], physics)
    unseen_df, _, _ = prepare_dataframe(unseen_df, args.target, [], physics)

    # Align columns
    all_cols = sorted(set(train_df.columns) | set(unseen_df.columns))
    train_df = train_df.reindex(columns=all_cols)
    unseen_df = unseen_df.reindex(columns=all_cols)

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    train_df.to_csv(args.out_train, index=False)
    unseen_df.to_csv(args.out_unseen, index=False)

    # Infer feature types
    num_features = [
        c
        for c in train_df.columns
        if c not in METADATA_COLS
        and (
            c.startswith("elem_")
            or c.startswith("phys_")
            or any(x in c for x in ["_um", "_g_cm3", "_gpa", "_mpa", "_percent", "_k"])
        )
    ]
    if args.target in num_features:
        num_features.remove(args.target)
    metadata_cols = [c for c in METADATA_COLS if c in train_df.columns]
    cat_features = [
        c
        for c in train_df.columns
        if c not in num_features + metadata_cols + [args.target]
    ]

    schema = {
        "target": args.target,
        "numerical_features": sorted(num_features),
        "categorical_features": sorted(cat_features),
        "element_features": sorted(elem_cols),
        "physics_features": sorted(phys_cols),
        "metadata_columns": metadata_cols,
    }

    os.makedirs(os.path.dirname(args.schema), exist_ok=True)
    with open(args.schema, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


if __name__ == "__main__":
    main()
