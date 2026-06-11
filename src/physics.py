"""Physics-informed featurization for alloy fracture-toughness modeling.

Converts raw composition strings and basic test metadata into descriptors
with established physical meaning:

Thermodynamic / electronic (high-entropy-alloy literature):
- Ideal mixing entropy  dS_mix/R = -sum(c_i ln c_i)
- Miedema-model mixing enthalpy  dH_mix = sum_{i<j} 4 H_ij c_i c_j
  (pairwise H_ij after Takeuchi & Inoue, Mater. Trans. 46 (2005) 2817;
  values are rounded model estimates, not measurements)
- Atomic size mismatch  delta = sqrt(sum c_i (1 - r_i / r_bar)^2)
- Electronegativity spread  d_chi = sqrt(sum c_i (chi_i - chi_bar)^2)
- Valence electron concentration  VEC = sum c_i VEC_i
- Composition-weighted melting point  Tm_mix = sum c_i Tm_i
- Phase-stability parameter  Omega = Tm_mix * dS_mix / |dH_mix|
  (Yang & Zhang, Mater. Chem. Phys. 132 (2012) 233)
- Rule-of-mixtures density estimate (mass-weighted harmonic mean)

Mechanics-derived (from measured columns where available):
- Homologous test temperature  T / Tm_mix
- Hall-Petch term  1 / sqrt(grain size)
- Elastic yield strain  YS / E
- Strain-hardening capacity  (UTS - YS) / YS
- H/E and H^3/E^2 indentation indices

All composition descriptors are NaN for rows without a parsable
composition; the downstream pipeline already handles missing values via
masking and median imputation. Pairs missing from the enthalpy table are
excluded from dH_mix and reported through `phys_dhmix_coverage` so the
model can discount low-coverage estimates.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PHYS_PREFIX = "phys_"

# Per-element properties:
# (metallic/covalent radius pm, Pauling electronegativity, VEC,
#  melting point K, atomic mass g/mol, density g/cm3 or None for gases)
ELEMENT_PROPS: Dict[str, Tuple[float, float, float, float, float, Optional[float]]] = {
    "Al": (143.2, 1.61, 3, 933.5, 26.98, 2.70),
    "B": (85.0, 2.04, 3, 2349.0, 10.81, 2.34),
    "C": (77.0, 2.55, 4, 3823.0, 12.01, 2.27),
    "Co": (125.1, 1.88, 9, 1768.0, 58.93, 8.86),
    "Cr": (124.9, 1.66, 6, 2180.0, 52.00, 7.19),
    "Cu": (127.8, 1.90, 11, 1357.8, 63.55, 8.96),
    "Fe": (124.1, 1.83, 8, 1811.0, 55.85, 7.87),
    "Hf": (157.8, 1.30, 4, 2506.0, 178.49, 13.31),
    "Mg": (160.1, 1.31, 2, 923.0, 24.31, 1.74),
    "Mn": (127.0, 1.55, 7, 1519.0, 54.94, 7.21),
    "Mo": (136.3, 2.16, 6, 2896.0, 95.95, 10.22),
    "N": (75.0, 3.04, 5, 63.2, 14.01, None),
    "Nb": (142.9, 1.60, 5, 2750.0, 92.91, 8.57),
    "Ni": (124.6, 1.91, 10, 1728.0, 58.69, 8.91),
    "O": (73.0, 3.44, 6, 54.4, 16.00, None),
    "Pb": (175.0, 2.33, 4, 600.6, 207.20, 11.34),
    "S": (102.0, 2.58, 6, 388.4, 32.06, 2.07),
    "Si": (115.3, 1.90, 4, 1687.0, 28.09, 2.33),
    "Sn": (162.3, 1.96, 4, 505.1, 118.71, 7.29),
    "Ta": (143.0, 1.50, 5, 3290.0, 180.95, 16.65),
    "Ti": (146.2, 1.54, 4, 1941.0, 47.87, 4.51),
    "V": (131.6, 1.63, 5, 2183.0, 50.94, 6.11),
    "W": (136.7, 2.36, 6, 3695.0, 183.84, 19.25),
    "Y": (180.1, 1.22, 3, 1799.0, 88.91, 4.47),
    "Zn": (139.4, 1.65, 12, 692.7, 65.38, 7.13),
    "Zr": (160.2, 1.33, 4, 2128.0, 91.22, 6.51),
}

# Pairwise Miedema mixing enthalpies H_ij in kJ/mol (rounded estimates,
# Takeuchi & Inoue 2005). Pairs not listed are treated as unknown and
# excluded from dH_mix (tracked via phys_dhmix_coverage). Interstitial
# pairs (N, O, S) and Pb are deliberately omitted: Miedema estimates are
# unreliable there.
_H_PAIRS: Dict[Tuple[str, str], float] = {
    ("Al", "B"): 0, ("Al", "C"): -36, ("Al", "Co"): -19, ("Al", "Cr"): -10,
    ("Al", "Cu"): -1, ("Al", "Fe"): -11, ("Al", "Hf"): -39, ("Al", "Mn"): -19,
    ("Al", "Mo"): -5, ("Al", "Nb"): -18, ("Al", "Ni"): -22, ("Al", "Si"): -19,
    ("Al", "Ta"): -19, ("Al", "Ti"): -30, ("Al", "V"): -16, ("Al", "W"): -2,
    ("Al", "Y"): -38, ("Al", "Zn"): 1, ("Al", "Zr"): -44,
    ("B", "Co"): -24, ("B", "Cr"): -31, ("B", "Cu"): 0, ("B", "Fe"): -26,
    ("B", "Hf"): -66, ("B", "Mn"): -32, ("B", "Mo"): -34, ("B", "Nb"): -54,
    ("B", "Ni"): -24, ("B", "Si"): -14, ("B", "Ta"): -54, ("B", "Ti"): -58,
    ("B", "V"): -42, ("B", "W"): -31, ("B", "Zr"): -71,
    ("C", "Co"): -42, ("C", "Cr"): -61, ("C", "Fe"): -50, ("C", "Hf"): -123,
    ("C", "Mn"): -66, ("C", "Mo"): -67, ("C", "Nb"): -102, ("C", "Ni"): -39,
    ("C", "Si"): -39, ("C", "Ta"): -101, ("C", "Ti"): -109, ("C", "V"): -75,
    ("C", "W"): -60, ("C", "Zr"): -131,
    ("Co", "Cr"): -4, ("Co", "Cu"): 6, ("Co", "Fe"): -1, ("Co", "Hf"): -35,
    ("Co", "Mn"): -5, ("Co", "Mo"): -5, ("Co", "Nb"): -25, ("Co", "Ni"): 0,
    ("Co", "Si"): -38, ("Co", "Ta"): -24, ("Co", "Ti"): -28, ("Co", "V"): -14,
    ("Co", "W"): -1, ("Co", "Y"): -22, ("Co", "Zn"): -5, ("Co", "Zr"): -41,
    ("Cr", "Cu"): 12, ("Cr", "Fe"): -1, ("Cr", "Hf"): -9, ("Cr", "Mn"): 2,
    ("Cr", "Mo"): 0, ("Cr", "Nb"): -7, ("Cr", "Ni"): -7, ("Cr", "Si"): -37,
    ("Cr", "Ta"): -7, ("Cr", "Ti"): -7, ("Cr", "V"): -2, ("Cr", "W"): 1,
    ("Cr", "Y"): 11, ("Cr", "Zn"): 5, ("Cr", "Zr"): -12,
    ("Cu", "Fe"): 13, ("Cu", "Hf"): -17, ("Cu", "Mn"): 4, ("Cu", "Mo"): 19,
    ("Cu", "Nb"): 3, ("Cu", "Ni"): 4, ("Cu", "Si"): -19, ("Cu", "Ta"): 2,
    ("Cu", "Ti"): -9, ("Cu", "V"): 5, ("Cu", "W"): 22, ("Cu", "Y"): -22,
    ("Cu", "Zn"): 1, ("Cu", "Zr"): -23,
    ("Fe", "Hf"): -21, ("Fe", "Mn"): 0, ("Fe", "Mo"): -2, ("Fe", "Nb"): -16,
    ("Fe", "Ni"): -2, ("Fe", "Si"): -35, ("Fe", "Ta"): -15, ("Fe", "Ti"): -17,
    ("Fe", "V"): -7, ("Fe", "W"): 0, ("Fe", "Y"): -1, ("Fe", "Zn"): 4,
    ("Fe", "Zr"): -25,
    ("Hf", "Mn"): -12, ("Hf", "Mo"): -4, ("Hf", "Nb"): 4, ("Hf", "Ni"): -42,
    ("Hf", "Si"): -77, ("Hf", "Ta"): 3, ("Hf", "Ti"): 0, ("Hf", "V"): -2,
    ("Hf", "W"): -6, ("Hf", "Zr"): 0,
    ("Mn", "Mo"): 5, ("Mn", "Nb"): -4, ("Mn", "Ni"): -8, ("Mn", "Si"): -45,
    ("Mn", "Ta"): -4, ("Mn", "Ti"): -8, ("Mn", "V"): -1, ("Mn", "W"): 6,
    ("Mn", "Zn"): -6, ("Mn", "Zr"): -15,
    ("Mo", "Nb"): -6, ("Mo", "Ni"): -7, ("Mo", "Si"): -35, ("Mo", "Ta"): -5,
    ("Mo", "Ti"): -4, ("Mo", "V"): 0, ("Mo", "W"): 0, ("Mo", "Zr"): -6,
    ("Nb", "Ni"): -30, ("Nb", "Si"): -56, ("Nb", "Ta"): 0, ("Nb", "Ti"): 2,
    ("Nb", "V"): -1, ("Nb", "W"): -8, ("Nb", "Zr"): 4,
    ("Ni", "Si"): -40, ("Ni", "Ta"): -29, ("Ni", "Ti"): -35, ("Ni", "V"): -18,
    ("Ni", "W"): -3, ("Ni", "Y"): -31, ("Ni", "Zn"): -9, ("Ni", "Zr"): -49,
    ("Si", "Ta"): -56, ("Si", "Ti"): -66, ("Si", "V"): -48, ("Si", "W"): -31,
    ("Si", "Zr"): -84,
    ("Ta", "Ti"): 1, ("Ta", "V"): -1, ("Ta", "W"): -7, ("Ta", "Zr"): 3,
    ("Ti", "V"): -2, ("Ti", "W"): -6, ("Ti", "Zr"): 0,
    ("V", "W"): -1, ("V", "Zr"): -4,
    ("W", "Zr"): -9,
}

H_MIX: Dict[frozenset, float] = {frozenset(k): float(v) for k, v in _H_PAIRS.items()}

_COMP_TOKEN = re.compile(r"^([A-Z][a-z]?)([0-9]*\.?[0-9]+)?$")


def parse_composition(entry: object) -> Dict[str, float]:
    """Parse a dash-separated at.% composition string into {element: at%}."""
    if not isinstance(entry, str) or not entry.strip():
        return {}
    result: Dict[str, float] = {}
    for part in entry.split("-"):
        part = part.strip()
        if not part:
            continue
        match = _COMP_TOKEN.match(part)
        if not match:
            continue
        value = match.group(2)
        if value is None:
            continue
        try:
            result[match.group(1)] = float(value)
        except ValueError:
            continue
    return result


def composition_descriptors(comp: Dict[str, float]) -> Dict[str, float]:
    """Compute thermodynamic/electronic descriptors from an at.% dict."""
    nan = float("nan")
    out = {
        "phys_smix_r": nan,
        "phys_dhmix": nan,
        "phys_dhmix_coverage": nan,
        "phys_delta_r": nan,
        "phys_dchi": nan,
        "phys_vec": nan,
        "phys_tm_mix": nan,
        "phys_omega": nan,
        "phys_density_rom": nan,
        "phys_n_elements": nan,
        "phys_max_frac": nan,
        "phys_mass_mean": nan,
    }
    known = {e: v for e, v in comp.items() if e in ELEMENT_PROPS and v > 0}
    total = sum(known.values())
    if not known or total <= 0:
        return out
    fracs = {e: v / total for e, v in known.items()}
    elements = sorted(fracs)

    smix_r = -sum(c * math.log(c) for c in fracs.values() if c > 0)
    r_bar = sum(c * ELEMENT_PROPS[e][0] for e, c in fracs.items())
    chi_bar = sum(c * ELEMENT_PROPS[e][1] for e, c in fracs.items())
    delta_r = math.sqrt(
        sum(c * (1.0 - ELEMENT_PROPS[e][0] / r_bar) ** 2 for e, c in fracs.items())
    )
    dchi = math.sqrt(
        sum(c * (ELEMENT_PROPS[e][1] - chi_bar) ** 2 for e, c in fracs.items())
    )
    vec = sum(c * ELEMENT_PROPS[e][2] for e, c in fracs.items())
    tm_mix = sum(c * ELEMENT_PROPS[e][3] for e, c in fracs.items())
    mass_mean = sum(c * ELEMENT_PROPS[e][4] for e, c in fracs.items())

    dh = 0.0
    covered_weight = 0.0
    total_weight = 0.0
    for i, ei in enumerate(elements):
        for ej in elements[i + 1:]:
            w = fracs[ei] * fracs[ej]
            total_weight += w
            h = H_MIX.get(frozenset((ei, ej)))
            if h is not None:
                dh += 4.0 * h * w
                covered_weight += w
    if total_weight > 0:
        coverage = covered_weight / total_weight
        dhmix = dh if coverage > 0 else float("nan")
    else:  # single element
        coverage = 1.0
        dhmix = 0.0

    if not math.isnan(dhmix):
        omega = tm_mix * smix_r * 8.314e-3 / max(abs(dhmix), 1e-3)
        omega = min(omega, 1000.0)
    else:
        omega = float("nan")

    mass_total = sum(fracs[e] * ELEMENT_PROPS[e][4] for e in fracs)
    vol_total = 0.0
    density_ok = True
    for e, c in fracs.items():
        rho = ELEMENT_PROPS[e][5]
        if rho is None:
            density_ok = False
            break
        vol_total += c * ELEMENT_PROPS[e][4] / rho
    density_rom = mass_total / vol_total if density_ok and vol_total > 0 else float("nan")

    out.update(
        phys_smix_r=smix_r,
        phys_dhmix=dhmix,
        phys_dhmix_coverage=coverage,
        phys_delta_r=delta_r * 100.0,
        phys_dchi=dchi,
        phys_vec=vec,
        phys_tm_mix=tm_mix,
        phys_omega=omega,
        phys_density_rom=density_rom,
        phys_n_elements=float(len(fracs)),
        phys_max_frac=max(fracs.values()),
        phys_mass_mean=mass_mean,
    )
    return out


def brittleness_bin(phase: object) -> str:
    """Pre-committed Mondrian bin from the reported phase structure.

    FCC-containing alloys stay tough or get tougher toward cryogenic
    temperatures, while BCC-only, intermetallic, boride and carbide
    structures embrittle below their (alloy-specific) transition. The
    two classes have very different toughness scatter, so conformal
    calibration is done per class. The rule is fixed on this physical
    ground and must not be tuned against calibration results.
    """
    if phase is None or (isinstance(phase, float) and math.isnan(phase)):
        return "unknown"
    s = str(phase).strip().lower()
    if not s or s == "nan":
        return "unknown"
    return "ductile" if "fcc" in s else "brittle"


# Mechanics features: name -> (required normalized columns, function)
def _mechanics_features(df: pd.DataFrame, tm_mix: pd.Series) -> Dict[str, pd.Series]:
    nan_series = pd.Series(np.nan, index=df.index)

    def col(name: str) -> pd.Series:
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else nan_series

    grain = col("grain_size_um")
    ys = col("yield_strength_mpa")
    uts = col("uts_mpa")
    e_gpa = col("youngs_modulus_gpa")
    h_gpa = col("hardness_gpa")
    t_k = col("testing_temperature_k")

    feats: Dict[str, pd.Series] = {}
    feats["phys_t_homologous"] = t_k / tm_mix
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["phys_hall_petch"] = 1.0 / np.sqrt(grain.where(grain > 0))
        feats["phys_yield_strain"] = ys / (e_gpa * 1000.0).where(e_gpa > 0)
        feats["phys_strain_hardening"] = (uts - ys) / ys.where(ys > 0)
        feats["phys_h_over_e"] = h_gpa / e_gpa.where(e_gpa > 0)
        feats["phys_h3_e2"] = h_gpa**3 / (e_gpa**2).where(e_gpa > 0)
    return feats


def add_physics_features(
    df: pd.DataFrame, composition_col: str = "composition_at_percent"
) -> Tuple[pd.DataFrame, List[str]]:
    """Append phys_* descriptor columns to a normalized-column dataframe.

    Expects column names already normalized (lowercase snake_case, as
    produced by `prepare_data._normalize_columns`). Returns the augmented
    dataframe and the list of added column names.
    """
    df = df.copy()
    if composition_col in df.columns:
        comp_rows = [composition_descriptors(parse_composition(v)) for v in df[composition_col]]
    else:
        comp_rows = [composition_descriptors({}) for _ in range(len(df))]
    comp_df = pd.DataFrame(comp_rows, index=df.index)

    mech = _mechanics_features(df, comp_df["phys_tm_mix"])
    for name, series in mech.items():
        comp_df[name] = series.replace([np.inf, -np.inf], np.nan)

    added = list(comp_df.columns)
    df = pd.concat([df, comp_df], axis=1)
    return df, added
