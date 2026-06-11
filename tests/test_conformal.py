import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.conformal import GroupCVPlus, evaluate_group_coverage, interval_metrics


def _make_data(n=300, n_groups=40, noise=0.5, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, size=(n, 4))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + noise * rng.randn(n)
    groups = rng.randint(0, n_groups, size=n)
    return X, y, groups


def test_cvplus_coverage_on_fresh_data():
    X, y, groups = _make_data(seed=1)
    Xt, yt, _ = _make_data(n=500, seed=2)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=8, seed=42)
    model.fit(X, y, groups=groups)
    lower, upper = model.predict_interval(Xt, alpha=0.1)
    m = interval_metrics(yt, lower, upper)
    # Guarantee is >= 0.8 at alpha=0.1; in practice ~0.9.
    assert m["coverage"] >= 0.85
    assert m["mean_width"] > 0


def test_intervals_widen_with_confidence():
    X, y, groups = _make_data(seed=3)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=5, seed=0)
    model.fit(X, y, groups=groups)
    lo90, hi90 = model.predict_interval(X[:20], alpha=0.10)
    lo95, hi95 = model.predict_interval(X[:20], alpha=0.05)
    assert np.all(lo95 <= lo90 + 1e-12)
    assert np.all(hi95 >= hi90 - 1e-12)
    assert np.all(hi90 > lo90)


def test_few_groups_reduces_folds():
    X, y, _ = _make_data(n=60, seed=4)
    groups = np.array([0, 1, 2] * 20)
    model = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=10, seed=0)
    model.fit(X, y, groups=groups)
    assert len(model.fold_models_) == 3
    lower, upper = model.predict_interval(X[:5], alpha=0.2)
    assert lower.shape == (5,)
    assert np.all(upper >= lower)


def test_deterministic_with_seed():
    X, y, groups = _make_data(seed=5)
    a = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=6, seed=7).fit(X, y, groups)
    b = GroupCVPlus(lambda: Ridge(alpha=1.0), n_folds=6, seed=7).fit(X, y, groups)
    la, ua = a.predict_interval(X[:10], alpha=0.1)
    lb, ub = b.predict_interval(X[:10], alpha=0.1)
    np.testing.assert_allclose(la, lb)
    np.testing.assert_allclose(ua, ub)


def test_requires_fit():
    model = GroupCVPlus(lambda: Ridge())
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 4)))
    with pytest.raises(RuntimeError):
        model.predict_interval(np.zeros((1, 4)))


def test_invalid_alpha():
    X, y, groups = _make_data(n=50, seed=6)
    model = GroupCVPlus(lambda: Ridge(), n_folds=4).fit(X, y, groups)
    with pytest.raises(ValueError):
        model.predict_interval(X[:2], alpha=0.7)


def test_evaluate_group_coverage():
    X, y, groups = _make_data(n=400, n_groups=50, seed=8)
    report = evaluate_group_coverage(
        lambda: Ridge(alpha=1.0), X, y, groups, alpha=0.1, n_splits=3, seed=1
    )
    assert report["empirical_coverage"] >= 0.8
    assert report["guaranteed_coverage"] == pytest.approx(0.8)
    assert report["mae"] < 1.0
    assert report["n_splits"] == 3
