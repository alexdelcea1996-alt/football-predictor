"""Tests for ensemble blenders."""

import numpy as np
import pytest

from football_predictor.evaluation.metrics import ranked_probability_score
from football_predictor.models.blending import StackingBlender, WeightBlender


@pytest.fixture
def informative_and_noise():
    """One informative member and one pure-noise member."""
    rng = np.random.default_rng(3)
    n = 800
    y = rng.choice(3, n, p=[0.45, 0.27, 0.28])

    good = 0.6 * np.eye(3)[y] + 0.4 * rng.dirichlet([2, 2, 2], n)
    good /= good.sum(axis=1, keepdims=True)
    noise = rng.dirichlet([1, 1, 1], n)
    return good, noise, y


class TestWeightBlender:
    def test_downweights_noise_member(self, informative_and_noise):
        good, noise, y = informative_and_noise

        blender = WeightBlender().fit([good, noise], y)

        assert blender.weights[0] > 0.9
        assert blender.weights[1] < 0.1
        assert np.isclose(blender.weights.sum(), 1.0)

    def test_beats_equal_average(self, informative_and_noise):
        good, noise, y = informative_and_noise

        fitted = WeightBlender().fit([good, noise], y).transform([good, noise])
        equal = WeightBlender([0.5, 0.5]).transform([good, noise])

        assert ranked_probability_score(y, fitted) < ranked_probability_score(y, equal)

    def test_never_worse_than_equal_weights_on_fitting_data(self, informative_and_noise):
        good, noise, y = informative_and_noise
        members = [good, noise, (good + noise) / 2]

        fitted = WeightBlender().fit(members, y)
        equal = WeightBlender([1 / 3] * 3)

        assert ranked_probability_score(y, fitted.transform(members)) <= (
            ranked_probability_score(y, equal.transform(members)) + 1e-9
        )

    def test_single_member_is_passthrough(self, informative_and_noise):
        good, _, y = informative_and_noise
        blender = WeightBlender().fit([good], y)
        assert np.allclose(blender.transform([good]), good)

    def test_roundtrip_persistence(self, informative_and_noise, tmp_path):
        good, noise, y = informative_and_noise
        blender = WeightBlender().fit([good, noise], y)

        path = tmp_path / "blender.joblib"
        blender.save(path)
        restored = WeightBlender.load(path)

        assert np.allclose(restored.weights, blender.weights)


class TestStackingBlender:
    def test_learns_to_prefer_informative_member(self, informative_and_noise):
        good, noise, y = informative_and_noise
        fit, hold = slice(0, 600), slice(600, None)

        stacker = StackingBlender().fit([good[fit], noise[fit]], y[fit])
        blended = stacker.transform([good[hold], noise[hold]])

        equal = (good[hold] + noise[hold]) / 2
        assert ranked_probability_score(y[hold], blended) < ranked_probability_score(
            y[hold], equal
        )

    def test_rejects_wrong_member_count(self, informative_and_noise):
        good, noise, y = informative_and_noise
        stacker = StackingBlender().fit([good, noise], y)

        with pytest.raises(ValueError, match="Expected 2 members"):
            stacker.transform([good])
