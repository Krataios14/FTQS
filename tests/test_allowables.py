import numpy as np
import pytest

from src.allowables import (
    basis_summary,
    basis_value_nonparametric,
    basis_value_normal,
    nonparametric_rank,
    tolerance_factor_normal,
)


def test_tolerance_factor_matches_tables():
    # Published one-sided (0.90, 0.95) tolerance factors
    assert tolerance_factor_normal(10, 0.90, 0.95) == pytest.approx(2.355, abs=0.01)
    assert tolerance_factor_normal(30, 0.90, 0.95) == pytest.approx(1.777, abs=0.01)
    # A-basis factor at n=10
    assert tolerance_factor_normal(10, 0.99, 0.95) == pytest.approx(3.981, abs=0.02)


def test_basis_value_normal_below_mean():
    rng = np.random.RandomState(0)
    x = rng.normal(100.0, 10.0, size=50)
    b = basis_value_normal(x, 0.90, 0.95)
    assert b["value"] < b["mean"]
    # Roughly mean - 1.6*sd for n=50
    assert 70 < b["value"] < 90


def test_nonparametric_rank_thresholds():
    # Classic result: minimum of n=29 samples is a valid B-basis bound
    assert nonparametric_rank(29, 0.90, 0.95) == 1
    assert nonparametric_rank(28, 0.90, 0.95) is None
    # A-basis needs ~299 samples for the minimum
    assert nonparametric_rank(299, 0.99, 0.95) == 1
    assert nonparametric_rank(250, 0.99, 0.95) is None
    # Rank grows with n
    assert nonparametric_rank(100, 0.90, 0.95) > 1


def test_basis_value_nonparametric():
    rng = np.random.RandomState(1)
    x = rng.normal(100.0, 10.0, size=100)
    b = basis_value_nonparametric(x, 0.90, 0.95)
    assert b is not None
    assert b["value"] == pytest.approx(np.sort(x)[b["rank"] - 1])
    assert b["value"] < np.mean(x)
    # Too few samples -> None
    assert basis_value_nonparametric(x[:10], 0.90, 0.95) is None


def test_basis_summary_structure():
    rng = np.random.RandomState(2)
    x = rng.normal(50.0, 5.0, size=40)
    s = basis_summary(x)
    assert set(s) == {"A_basis", "B_basis"}
    assert s["B_basis"]["normal"]["value"] > s["A_basis"]["normal"]["value"]
    assert s["B_basis"]["nonparametric"] is not None
    assert s["A_basis"]["nonparametric"] is None  # n=40 too small for A-basis
