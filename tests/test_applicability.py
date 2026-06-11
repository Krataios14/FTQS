import joblib
import numpy as np
import pandas as pd
import pytest

from src.applicability import TrustModel


def _train_cloud(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, 5)


def test_in_domain_vs_extrapolation():
    X = _train_cloud()
    model = TrustModel(n_neighbors=5).fit(X)
    near = np.zeros((1, 5))
    far = np.full((1, 5), 50.0)
    s_near = model.score(near)
    s_far = model.score(far)
    assert s_near.loc[0, "trust_tier"] == "A"
    assert s_near.loc[0, "trust_score"] > 50
    assert s_far.loc[0, "trust_tier"] == "C"
    assert s_far.loc[0, "trust_score"] == 0.0


def test_handles_nan_features():
    X = _train_cloud()
    X[::7, 2] = np.nan
    model = TrustModel(n_neighbors=4).fit(X)
    q = np.array([[0.1, -0.2, np.nan, 0.3, 0.0]])
    s = model.score(q)
    assert np.isfinite(s.loc[0, "knn_distance"])


def test_neighbors_return_provenance():
    X = _train_cloud(n=50)
    meta = pd.DataFrame(
        {
            "reference": [f"Paper {i}" for i in range(50)],
            "composition": [f"Alloy{i}" for i in range(50)],
            "measured": np.arange(50.0),
        }
    )
    model = TrustModel(n_neighbors=3).fit(X, metadata=meta)
    blocks = model.neighbors(X[:2], k=3)
    assert len(blocks) == 2
    assert list(blocks[0].columns) == ["distance", "reference", "composition", "measured"]
    assert blocks[0]["distance"].is_monotonic_increasing
    # The nearest neighbour of a training point is (almost) itself
    assert blocks[0].loc[0, "distance"] == pytest.approx(0.0, abs=1e-9)


def test_constant_feature_no_crash():
    X = _train_cloud(n=30)
    X[:, 4] = 7.0
    model = TrustModel().fit(X)
    s = model.score(X[:3])
    assert np.isfinite(s["knn_distance"]).all()


def test_roundtrip_serialization(tmp_path):
    X = _train_cloud(n=40)
    meta = pd.DataFrame({"reference": [f"p{i}" for i in range(40)]})
    model = TrustModel(n_neighbors=3).fit(X, metadata=meta)
    path = tmp_path / "trust.joblib"
    joblib.dump(model.to_dict(), path)
    restored = TrustModel.from_dict(joblib.load(path))
    a = model.score(X[:5])
    b = restored.score(X[:5])
    pd.testing.assert_frame_equal(a, b)


def test_needs_two_rows():
    with pytest.raises(ValueError):
        TrustModel().fit(np.zeros((1, 3)))
