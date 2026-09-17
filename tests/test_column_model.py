"""Tests for probability-column ensemble members."""

import numpy as np
import pytest

from football_predictor.models.column_model import KNOWN_SOURCES, ColumnProbabilityModel


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    n = 400
    y = rng.choice(3, n, p=[0.45, 0.27, 0.28])

    probs = 0.6 * np.eye(3)[y] + 0.4 * rng.dirichlet([2, 2, 2], n)
    probs /= probs.sum(axis=1, keepdims=True)

    X = np.column_stack([rng.normal(size=(n, 4)), probs])
    names = ["a", "b", "c", "d", "poisson_prob_home", "poisson_prob_draw", "poisson_prob_away"]
    return X, y, names


class TestColumnProbabilityModel:
    def test_reads_columns_as_probabilities(self, data):
        X, y, names = data
        model = ColumnProbabilityModel(KNOWN_SOURCES["dixon_coles"]).fit(X, y, names)

        probs = model.predict_proba(X)
        np.testing.assert_allclose(probs, X[:, 4:7], rtol=1e-9)

    def test_missing_columns_raise(self, data):
        X, y, names = data
        with pytest.raises(ValueError, match="Missing probability columns"):
            ColumnProbabilityModel(KNOWN_SOURCES["market"]).fit(X, y, names)

    def test_availability_check(self, data):
        _, _, names = data
        assert ColumnProbabilityModel.available(KNOWN_SOURCES["dixon_coles"], names)
        assert not ColumnProbabilityModel.available(KNOWN_SOURCES["market"], names)

    def test_zero_filled_rows_fall_back_to_prior(self, data):
        """Unpriced fixtures arrive as zeros after fillna, not as a distribution."""
        X, y, names = data
        model = ColumnProbabilityModel(KNOWN_SOURCES["dixon_coles"]).fit(X, y, names)

        blank = X[:1].copy()
        blank[0, 4:7] = 0.0
        probs = model.predict_proba(blank)

        assert probs.sum() == pytest.approx(1.0)
        expected_prior = np.bincount(y, minlength=3) / len(y)
        np.testing.assert_allclose(probs[0], expected_prior, rtol=1e-9)

    def test_outputs_always_valid(self, data):
        X, y, names = data
        model = ColumnProbabilityModel(KNOWN_SOURCES["dixon_coles"]).fit(X, y, names)

        broken = X[:3].copy()
        broken[0, 4:7] = [np.nan, 0.5, 0.5]
        broken[1, 4:7] = [-1.0, 1.0, 1.0]
        probs = model.predict_proba(broken)

        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()

    def test_calibration_is_applied_when_fitted(self, data):
        X, y, names = data
        model = ColumnProbabilityModel(KNOWN_SOURCES["dixon_coles"]).fit(
            X[:300], y[:300], names, calibration_set=(X[300:], y[300:])
        )

        assert model.calibrator is not None
        assert not np.allclose(model.predict_proba(X[:10]), model.predict_proba_raw(X[:10]))

    def test_roundtrip_persistence(self, data, tmp_path):
        X, y, names = data
        model = ColumnProbabilityModel(KNOWN_SOURCES["dixon_coles"]).fit(X, y, names)
        expected = model.predict_proba(X[:10])

        path = tmp_path / "member.joblib"
        model.save(path)
        restored = ColumnProbabilityModel(KNOWN_SOURCES["dixon_coles"]).load(path)

        np.testing.assert_allclose(restored.predict_proba(X[:10]), expected)
