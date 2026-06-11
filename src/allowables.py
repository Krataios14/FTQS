"""Design-allowable style one-sided lower tolerance bounds.

Aerospace structures are not certified against mean properties: MMPDS
defines the B-basis allowable as the value exceeded by 90% of the
population with 95% confidence, and A-basis as 99%/95%. This module
computes those bounds from measured samples:

- Normal-distribution bound: x_bar - k * s, with the exact one-sided
  tolerance factor k from the noncentral t distribution.
- Nonparametric (distribution-free) bound: an order statistic chosen so
  the coverage/confidence requirement holds for any distribution
  (requires n >= 29 for B-basis, n >= 299 for A-basis).

Intended use here: compute screening allowables from physical test
results and compare them against the model's conformal lower bounds.
Model outputs are *screening* values for test planning and material
down-selection -- they do not replace MMPDS/CMH-17 qualification.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats

BASIS_DEFINITIONS = {
    "B": {"coverage": 0.90, "confidence": 0.95},
    "A": {"coverage": 0.99, "confidence": 0.95},
}


def tolerance_factor_normal(n: int, coverage: float = 0.90, confidence: float = 0.95) -> float:
    """Exact one-sided normal tolerance factor k (noncentral t)."""
    if n < 2:
        raise ValueError("Need at least 2 samples")
    z_p = stats.norm.ppf(coverage)
    nc = z_p * np.sqrt(n)
    return float(stats.nct.ppf(confidence, df=n - 1, nc=nc) / np.sqrt(n))


def basis_value_normal(
    samples: np.ndarray, coverage: float = 0.90, confidence: float = 0.95
) -> Dict[str, float]:
    """Lower tolerance bound assuming normality: x_bar - k*s."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    k = tolerance_factor_normal(n, coverage, confidence)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    return {
        "method": "normal",
        "n": n,
        "mean": mean,
        "sd": sd,
        "k_factor": k,
        "value": mean - k * sd,
        "coverage": coverage,
        "confidence": confidence,
    }


def nonparametric_rank(n: int, coverage: float = 0.90, confidence: float = 0.95) -> Optional[int]:
    """Largest order-statistic rank r (1-based) giving a valid
    distribution-free lower tolerance bound, or None if n is too small.

    The r-th smallest sample is a valid bound when the probability that
    at least proportion `coverage` of the population exceeds it is at
    least `confidence`: Beta_cdf(1 - coverage; r, n - r + 1) >= confidence.
    """
    if n < 1:
        return None
    best = None
    for r in range(1, n + 1):
        conf = stats.beta.cdf(1.0 - coverage, r, n - r + 1)
        if conf >= confidence:
            best = r
        else:
            break
    return best


def basis_value_nonparametric(
    samples: np.ndarray, coverage: float = 0.90, confidence: float = 0.95
) -> Optional[Dict[str, float]]:
    """Distribution-free lower tolerance bound, or None if n insufficient."""
    x = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    x = x[np.isfinite(x)]
    r = nonparametric_rank(len(x), coverage, confidence)
    if r is None:
        return None
    return {
        "method": "nonparametric",
        "n": int(len(x)),
        "rank": int(r),
        "value": float(x[r - 1]),
        "coverage": coverage,
        "confidence": confidence,
    }


def basis_summary(samples: np.ndarray) -> Dict[str, Dict]:
    """A- and B-basis screening values by both methods."""
    out: Dict[str, Dict] = {}
    for name, spec in BASIS_DEFINITIONS.items():
        normal = basis_value_normal(samples, **spec)
        nonpar = basis_value_nonparametric(samples, **spec)
        out[f"{name}_basis"] = {
            "normal": normal,
            "nonparametric": nonpar,
        }
    return out
