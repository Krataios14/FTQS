"""Candidate alloy screening and physical-test prioritization.

Two workflows on top of a qualification artifact:

screen: generate composition candidates around known alloys (or take a
user-supplied CSV), predict fracture toughness with conformal bounds,
apply engineering constraints (density ceiling, trust tier) and rank.
The default objective ranks by the conservative lower bound: the alloy
you would actually commit to is the one whose guaranteed floor is
highest, not the one with the best point estimate.

advise: rank candidate test articles by how much a physical test would
be worth. The score combines relative interval width with distance from
the training distribution, so the suggested tests are the ones that
shrink model uncertainty where it is largest. Fracture toughness tests
are expensive; this is the test-matrix reduction tool. By default the
advised batch is also diversified in feature space, so a small test
budget is not spent on near-replicas of one alloy (--no-diversify
restores plain score ordering).

Usage:
    python -m src.screen --mode screen --temperature 298 --top 20
    python -m src.screen --mode advise --data data/processed_unseen.csv --top 10
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.certify import _ensure_columns, certify_dataframe, latest_qualify_run, load_run
from src.physics import add_physics_features, parse_composition
from src.qualify import build_matrix


def _format_composition(comp: Dict[str, float]) -> str:
    return "-".join(f"{e}{v:.2f}" for e, v in sorted(comp.items()) if v > 0.005)


def generate_candidates(
    base_compositions: List[Dict[str, float]],
    n_per_base: int = 20,
    jitter: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Perturb known compositions on the at.% simplex.

    Each candidate multiplies every element fraction by an independent
    factor in [1-jitter, 1+jitter] and renormalizes to 100 at.%, which
    keeps candidates inside the neighbourhood the model can support.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for base in base_compositions:
        base = {e: v for e, v in base.items() if v > 0}
        if not base:
            continue
        total = sum(base.values())
        norm = {e: 100.0 * v / total for e, v in base.items()}
        rows.append({"composition_at_percent": _format_composition(norm), "origin": "base"})
        for _ in range(n_per_base):
            perturbed = {
                e: v * (1.0 + rng.uniform(-jitter, jitter)) for e, v in norm.items()
            }
            s = sum(perturbed.values())
            perturbed = {e: 100.0 * v / s for e, v in perturbed.items()}
            rows.append(
                {"composition_at_percent": _format_composition(perturbed), "origin": "perturbed"}
            )
    df = pd.DataFrame(rows).drop_duplicates(subset="composition_at_percent")
    return df.reset_index(drop=True)


def _expand_elements(df: pd.DataFrame, num_features: List[str]) -> pd.DataFrame:
    """Fill elem_* model features from the composition strings.

    Elements absent from a parsed composition are 0.0 (matching how the
    training table encodes them), not NaN.
    """
    df = df.copy()
    elem_cols = [c for c in num_features if c.startswith("elem_")]
    parsed = [parse_composition(v) for v in df["composition_at_percent"]]
    for col in elem_cols:
        symbol = col[len("elem_"):].capitalize()
        df[col] = [p.get(symbol, 0.0) if p else np.nan for p in parsed]
    return df


def screen_candidates(
    candidates: pd.DataFrame,
    run: Dict,
    temperature_k: float = 298.0,
    objective: str = "conservative",
    max_density: Optional[float] = None,
    min_tier: str = "B",
) -> pd.DataFrame:
    df = candidates.copy()
    if "testing_temperature_k" not in df.columns:
        df["testing_temperature_k"] = temperature_k
    df = _expand_elements(df, run["features"]["numerical_features"])
    if "phys_density_rom" not in df.columns:
        df, _ = add_physics_features(df)
    out, _ = certify_dataframe(df, run)
    out["origin"] = candidates.get("origin", pd.Series(["user"] * len(df))).to_numpy()

    if max_density is not None and "phys_density_rom" in df.columns:
        out["density_estimate_g_cm3"] = df["phys_density_rom"].to_numpy()
        out = out[out["density_estimate_g_cm3"] <= max_density]

    tier_rank = {"A": 0, "B": 1, "C": 2}
    out = out[out["trust_tier"].map(tier_rank) <= tier_rank.get(min_tier, 1)]

    if objective == "conservative":
        out = out.sort_values("lower_90", ascending=False)
    elif objective == "explore":
        out = out.sort_values("upper_90", ascending=False)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return out.reset_index(drop=True)


def _standardized_features(data: pd.DataFrame, run: Dict) -> np.ndarray:
    """Feature rows in the trust model's standardized space.

    Builds the matrix exactly as certify does, then applies the same
    median-fill / robust-scale arithmetic the fitted trust model uses.
    """
    features = run["features"]
    frame = _ensure_columns(data, features)
    X = build_matrix(
        frame, features["numerical_features"], features["categorical_features"], run["artifacts"]
    )
    trust = run["trust"]
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    filled = np.where(np.isnan(X), trust.median_, X)
    return (filled - trust.center_) / trust.scale_


def advise_tests(
    data: pd.DataFrame, run: Dict, top: int = 10, diversify: bool = True
) -> pd.DataFrame:
    """Rank candidate test articles by expected value of a physical test.

    With diversify=True (the default) the batch is spread out in the trust
    model's standardized feature space: candidates are taken in score order,
    and after the first pick a candidate is only accepted if its minimum
    Euclidean distance to everything already accepted is at least tau, the
    median training self-distance. Tests are bought in small batches, and a
    batch of near-replicas of one alloy wastes most of it. If too few
    candidates satisfy the spacing rule, the batch is topped up in plain
    score order and those rows carry diversity_relaxed=True. The diversified
    frame adds 'min_distance_to_selected' (NaN for the first pick) and
    'diversity_relaxed' columns; diversify=False returns the plain ranking
    unchanged.
    """
    out, _ = certify_dataframe(data, run)
    pred = out["predicted_toughness_mpa_m0_5"].to_numpy()
    width = out["upper_90"].to_numpy() - out["lower_90"].to_numpy()
    rel_width = width / np.maximum(pred, 1e-6)
    novelty = (100.0 - out["trust_score"].to_numpy()) / 100.0
    out["relative_interval_width"] = rel_width
    out["test_value_score"] = rel_width * (1.0 + novelty)
    if not diversify:
        out = out.sort_values("test_value_score", ascending=False).head(top)
        return out.reset_index(drop=True)

    Z = _standardized_features(data, run)
    tau = float(np.median(run["trust"].train_self_dist_))
    out = out.reset_index(drop=True)  # positions now align with Z rows
    order = out.sort_values("test_value_score", ascending=False).index.to_numpy()

    # Greedy pass: skipped candidates never become eligible again because
    # adding picks can only shrink a candidate's minimum distance.
    accepted: List[int] = []
    skipped: List[int] = []
    min_dist: Dict[int, float] = {}
    for idx in order:
        if len(accepted) >= top:
            break
        idx = int(idx)
        if not accepted:
            accepted.append(idx)
            min_dist[idx] = float("nan")
            continue
        d = float(min(np.linalg.norm(Z[idx] - Z[j]) for j in accepted))
        if d >= tau:
            accepted.append(idx)
            min_dist[idx] = d
        else:
            skipped.append(idx)

    relaxed = set()
    for idx in skipped:
        if len(accepted) >= top:
            break
        d = float(min(np.linalg.norm(Z[idx] - Z[j]) for j in accepted))
        accepted.append(idx)
        min_dist[idx] = d
        relaxed.add(idx)

    result = out.loc[accepted].copy()
    result["min_distance_to_selected"] = [min_dist[i] for i in accepted]
    result["diversity_relaxed"] = [i in relaxed for i in accepted]
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["screen", "advise"], default="screen")
    parser.add_argument("--run", type=str, default=None, help="qualify_* artifact (default: latest)")
    parser.add_argument("--data", type=str, default=None, help="candidate CSV (advise/user screening)")
    parser.add_argument("--temperature", type=float, default=298.0, help="test temperature K")
    parser.add_argument("--objective", choices=["conservative", "explore"], default="conservative")
    parser.add_argument("--max-density", type=float, default=None, help="g/cm3 ceiling (rule of mixtures)")
    parser.add_argument("--min-tier", choices=["A", "B", "C"], default="B")
    parser.add_argument("--n-per-base", type=int, default=20)
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--no-diversify",
        action="store_true",
        help="advise: rank purely by score instead of spacing the batch",
    )
    parser.add_argument("--out", type=str, default="screening_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = args.run or latest_qualify_run()
    run = load_run(run_dir)

    if args.mode == "advise":
        if not args.data:
            raise SystemExit("--data is required for advise mode")
        data = pd.read_csv(args.data, skip_blank_lines=True)
        result = advise_tests(data, run, top=args.top, diversify=not args.no_diversify)
    else:
        if args.data:
            candidates = pd.read_csv(args.data, skip_blank_lines=True)
        else:
            meta = run["trust"].metadata_
            if meta is None or "composition_at_percent" not in meta.columns:
                raise SystemExit("Artifact has no stored compositions; pass --data")
            comps = [
                parse_composition(v)
                for v in meta["composition_at_percent"].dropna().unique()
            ]
            comps = [c for c in comps if c]
            candidates = generate_candidates(
                comps, n_per_base=args.n_per_base, jitter=args.jitter, seed=args.seed
            )
        result = screen_candidates(
            candidates,
            run,
            temperature_k=args.temperature,
            objective=args.objective,
            max_density=args.max_density,
            min_tier=args.min_tier,
        ).head(args.top)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(json.dumps({"run_dir": run_dir, "mode": args.mode, "out": args.out, "n": len(result)}, indent=2))


if __name__ == "__main__":
    main()
