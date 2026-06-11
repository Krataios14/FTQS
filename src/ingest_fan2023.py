"""Rebuild the combined assets from the Fan et al. (2023) source dataset.

The bundled training/unseen CSVs originally carried 129 of the 148
fracture toughness records in "Dataset for Fracture and Impact
Toughness of High-Entropy Alloys" (Fan, Chen, Steingrimsson, Xiong,
Li, Liaw; Scientific Data 10, 2023; Materials Cloud
doi:10.24435/materialscloud:d6-pf), keeping only rows with a plain
K_IC value. This script ingests the source spreadsheet directly so the
remaining records become usable as well:

- K_Q records (refractory NbTaTiZr-Mo and NbTaTiV series, including
  tests at 77 K and 134-226 K, regions where the data is thinnest)
- J_IC / J_Q records, converted to K via K = sqrt(J * E / (1 - nu^2))
  with nu = 0.3 where the record reports a Young's modulus
- the toughness measure type (KIC / KQ / KJIC / KJQ) is kept as a
  column so the model can account for the difference instead of
  treating all values as identical, and the reported measurement
  uncertainty (the +/- part) is kept as metadata

Hand-collected records that are not part of the source dataset (steel
datapoints from Ritchie 1976 and supplier datasheets, cryogenic CoCrNi
and CoCrFeMnNi, WC-Co hardmetal compositions) live in
assets/manual_records.csv and are merged in unchanged. The unseen
split is pinned by assets/unseen_keys.json so evaluation stays
comparable across dataset revisions.

Usage:
    python -m src.ingest_fan2023 ^
      --xlsx assets/fan2023_hea_toughness.xlsx ^
      --manual assets/manual_records.csv ^
      --unseen-keys assets/unseen_keys.json ^
      --out-train assets/combined_fracture_training.csv ^
      --out-unseen assets/combined_fracture_unseen.csv
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

POISSON_RATIO = 0.3

# Corrections for typos in the source spreadsheet, applied to the
# composition string before parsing. "(NbTaTiZr)90M10" sits between the
# Mo5 and Mo20 rows of the same Mo-substitution series in the source,
# so "M" is read as Mo.
COMPOSITION_FIXES = {
    "(NbTaTiZr)90M10": "(NbTaTiZr)90Mo10",
}

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_GROUP = re.compile(r"\(([A-Za-z]+)\)(\d*\.?\d*)")
_ANNOTATION = re.compile(r"\s*\([^)0-9]*\)\s*$")


def parse_formula(s: str) -> Dict[str, float]:
    """Parse a molar-ratio formula like Al0.2CrFeNiTi0.2 into ratios.

    Parenthesized groups with a multiplier, e.g. (NbTaTiZr)95Mo5, split
    the multiplier equally over the group members. A trailing
    parenthesized annotation without digits, e.g. "(single
    crystalline)", is ignored.
    """
    if not isinstance(s, str) or not s.strip():
        return {}
    s = s.strip()
    s = COMPOSITION_FIXES.get(s, s)
    s = _ANNOTATION.sub("", s)

    ratios: Dict[str, float] = {}

    def add(el: str, r: float) -> None:
        ratios[el] = ratios.get(el, 0.0) + r

    pos = 0
    for m in _GROUP.finditer(s):
        # elements before the group
        for el, num in _ELEMENT.findall(s[pos:m.start()]):
            add(el, float(num) if num else 1.0)
        members = _ELEMENT.findall(m.group(1))
        group_total = float(m.group(2)) if m.group(2) else 1.0
        member_ratios = [(el, float(num) if num else 1.0) for el, num in members]
        weight_sum = sum(r for _, r in member_ratios)
        for el, r in member_ratios:
            add(el, group_total * r / weight_sum)
        pos = m.end()
    for el, num in _ELEMENT.findall(s[pos:]):
        add(el, float(num) if num else 1.0)
    return ratios


def to_at_percent_string(ratios: Dict[str, float]) -> str:
    total = sum(ratios.values())
    if total <= 0:
        return ""
    return "-".join(f"{el}{100.0 * r / total:.2f}" for el, r in sorted(ratios.items()))


_RANGE = re.compile(r"^\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*$")


def parse_value_pm(v: object) -> Tuple[float, float]:
    """Parse '5.8±0.2', '295 (200K)', '10-20' or a number.

    Returns (value, uncertainty); either may be NaN.
    """
    if pd.isna(v):
        return float("nan"), float("nan")
    if isinstance(v, (int, float)):
        return float(v), float("nan")
    s = str(v).strip().replace("±", "+-")
    s = re.sub(r"\([^)]*\)", "", s).strip()  # drop parenthetical notes
    m = _RANGE.match(s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2.0, (hi - lo) / 2.0
    parts = s.split("+-")
    try:
        value = float(parts[0].strip())
    except ValueError:
        return float("nan"), float("nan")
    unc = float("nan")
    if len(parts) > 1:
        try:
            unc = float(parts[1].strip())
        except ValueError:
            pass
    return value, unc


def j_to_k(j_kj_m2: float, e_gpa: float, nu: float = POISSON_RATIO) -> float:
    """K [MPa m^0.5] from J [kJ/m^2] under plane strain."""
    if not np.isfinite(j_kj_m2) or not np.isfinite(e_gpa) or j_kj_m2 <= 0 or e_gpa <= 0:
        return float("nan")
    e_prime = e_gpa * 1e9 / (1.0 - nu**2)
    return float(np.sqrt(j_kj_m2 * 1e3 * e_prime) / 1e6)


def _col(ft: pd.DataFrame, prefix: str) -> str:
    matches = [c for c in ft.columns if str(c).strip().startswith(prefix)]
    if not matches:
        raise KeyError(f"No column starting with '{prefix}'")
    return matches[0]


def convert_fracture_sheet(xlsx_path: str) -> pd.DataFrame:
    ft = pd.read_excel(xlsx_path, sheet_name="Fracture toughness", header=0)
    ft.columns = [str(c).strip() for c in ft.columns]

    rows = []
    skipped = []
    for _, r in ft.iterrows():
        kic, kic_u = parse_value_pm(r[_col(ft, "KIC")])
        kq, kq_u = parse_value_pm(r[_col(ft, "KQ")])
        jic, _ = parse_value_pm(r[_col(ft, "JIC")])
        jq, _ = parse_value_pm(r[_col(ft, "JQ")])
        e_gpa, _ = parse_value_pm(r[_col(ft, "Young")])

        if np.isfinite(kic):
            k, k_u, measure = kic, kic_u, "KIC"
        elif np.isfinite(kq):
            k, k_u, measure = kq, kq_u, "KQ"
        elif np.isfinite(jic) and np.isfinite(e_gpa):
            k, k_u, measure = j_to_k(jic, e_gpa), float("nan"), "KJIC"
        elif np.isfinite(jq) and np.isfinite(e_gpa):
            k, k_u, measure = j_to_k(jq, e_gpa), float("nan"), "KJQ"
        else:
            skipped.append(r["ID"])
            continue

        hv, _ = parse_value_pm(r[_col(ft, "Hardness (HV)")])
        h_gpa, _ = parse_value_pm(r[_col(ft, "Hardness (GPa)")])
        if not np.isfinite(h_gpa) and np.isfinite(hv):
            h_gpa = hv * 0.009807  # HV (kgf/mm^2) to GPa

        grain, _ = parse_value_pm(r[_col(ft, "Grain size")])
        density, _ = parse_value_pm(r[_col(ft, "Density")])
        ys, _ = parse_value_pm(r[_col(ft, "Tensile YS")])
        uts, _ = parse_value_pm(r[_col(ft, "UTS")])
        elong, _ = parse_value_pm(r[_col(ft, "Final elongation")])
        temp, _ = parse_value_pm(r[_col(ft, "Testing temperature")])

        rows.append(
            {
                "Composition (at. %)": to_at_percent_string(parse_formula(r["Composition"])),
                "Material condition": r["Material condition"],
                "Processing history": r["Processing history"],
                "Phase": r["Phase"],
                "Grain_size_um": grain,
                "Density_g_cm3": density,
                "Hardness_GPa": h_gpa,
                "Youngs_modulus_GPa": e_gpa,
                "Yield_strength_MPa": ys,
                "UTS_MPa": uts,
                "Final_elongation_percent": elong,
                "Testing_temperature_K": temp,
                "Fracture_toughness_MPa_m0.5": k,
                "Toughness_measure": measure,
                "Toughness_uncertainty_MPa_m0.5": k_u,
                "Reference": r["Reference"],
            }
        )
    out = pd.DataFrame(rows)
    if skipped:
        print(f"skipped {len(skipped)} source records with no usable toughness value: IDs {skipped}")
    return out


def split_unseen(df: pd.DataFrame, unseen_keys: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pin the unseen split: match each key once by (K, T, reference prefix)."""
    df = df.reset_index(drop=True)
    used = set()
    unseen_idx = []
    for key in unseen_keys:
        for i, row in df.iterrows():
            if i in used:
                continue
            k = row["Fracture_toughness_MPa_m0.5"]
            t = row["Testing_temperature_K"]
            if not (np.isfinite(k) and np.isfinite(t)):
                continue
            if round(k, 2) == key["k"] and round(t, 0) == key["temp"] and str(
                row["Reference"]
            ).startswith(key["ref_prefix"][:20]):
                used.add(i)
                unseen_idx.append(i)
                break
    unseen = df.loc[unseen_idx]
    train = df.drop(index=unseen_idx)
    return train.reset_index(drop=True), unseen.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default="assets/fan2023_hea_toughness.xlsx")
    parser.add_argument("--manual", default="assets/manual_records.csv")
    parser.add_argument("--unseen-keys", default="assets/unseen_keys.json")
    parser.add_argument("--out-train", default="assets/combined_fracture_training.csv")
    parser.add_argument("--out-unseen", default="assets/combined_fracture_unseen.csv")
    args = parser.parse_args()

    converted = convert_fracture_sheet(args.xlsx)
    manual = pd.read_csv(args.manual, skip_blank_lines=True)
    combined = pd.concat([converted, manual], ignore_index=True)
    combined = combined[combined["Fracture_toughness_MPa_m0.5"].notna()].reset_index(drop=True)

    with open(args.unseen_keys, "r", encoding="utf-8") as f:
        unseen_keys = json.load(f)
    train, unseen = split_unseen(combined, unseen_keys)

    train.to_csv(args.out_train, index=False)
    unseen.to_csv(args.out_unseen, index=False)
    print(
        json.dumps(
            {
                "converted_from_xlsx": int(len(converted)),
                "manual": int(len(manual)),
                "train": int(len(train)),
                "unseen": int(len(unseen)),
                "measures": combined["Toughness_measure"].fillna("manual").value_counts().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
