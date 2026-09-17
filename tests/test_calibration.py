"""Tests for probability calibration."""

import numpy as np
import pytest

from football_predictor.evaluation.metrics import ranked_probability_score
from football_predictor.models.calibration import ProbabilityCalibrator


@pytest.fixture
def overconfident_data():
    """Probabilities that are systematically too confident for the true rate."""
    rng = np.random.default_rng(7)
    n = 1200
    y = rng.choice(3, n, p=[0.45, 0.27, 0.28])

    base = np.eye(3)[y] * 0.5 + rng.dirichlet([3, 3, 3], n) * 0.5
    # Sharpen: push probabilities towards 0/1 so they overstate certainty
    sharp = base ** 2.5
    sharp /= sharp.sum(axis=1, keepdims=True)
    return sharp, y


class TestProbabilityCalibrator:
    def test_isotonic_improves_overconfident_probabilities(self, overconfident_data):
        probs, y = overconfident_data
        fit, hold = slice(0, 800), slice(800, None)

        cal = ProbabilityCalibrator(method="isotonic").fit(probs[fit], y[fit])
        calibrated = cal.transform(probs[hold])

        assert cal.effective_method == "isotonic"
        assert ranked_probability_score(y[hold], calibrated) <= ranked_probability_score(
            y[hold], probs[hold]
        )

    def test_outputs_are_valid_distributions(self, overconfident_data):
        probs, y = overconfident_data
        for method in ("isotonic", "vector", "none"):
            out = ProbabilityCalibrator(method=method).fit(probs, y).transform(probs)
            assert np.allclose(out.sum(axis=1), 1.0)
            assert (out >= 0).all()

    def test_thin_holdout_degrades_to_vector_scaling(self, overconfident_data):
        probs, y = overconfident_data
        cal = ProbabilityCalibrator(method="isotonic").fit(probs[:40], y[:40])
        assert cal.effective_method == "vector"

    def test_missing_class_falls_back_to_identity(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.8, 0.1, 0.1]])
        y = np.array([0, 0, 0])  # no draws or away wins present

        cal = ProbabilityCalibrator(method="isotonic").fit(probs, y)

        assert cal.effective_method == "none"
        assert np.allclose(cal.transform(probs), probs)

    def test_roundtrip_persistence(self, overconfident_data, tmp_path):
        probs, y = overconfident_data
        cal = ProbabilityCalibrator(method="isotonic").fit(probs, y)
        expected = cal.transform(probs[:20])

        path = tmp_path / "calib.joblib"
        cal.save(path)
        restored = ProbabilityCalibrator.load(path)

        assert np.allclose(restored.transform(probs[:20]), expected)
