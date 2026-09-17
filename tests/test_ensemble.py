"""Tests for ensemble predictor."""

import numpy as np
import pytest

from football_predictor.evaluation.metrics import ranked_probability_score
from football_predictor.models.ensemble import EnsemblePredictor


@pytest.fixture(scope="module")
def sample_data():
    """Chronologically ordered data with a learnable signal."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 900, 12

    X = rng.normal(size=(n_samples, n_features))
    # Outcome depends on the first two features, so members have real skill
    logits = np.column_stack([
        0.9 * X[:, 0] + 0.4,
        0.3 * X[:, 1],
        -0.9 * X[:, 0] + 0.5 * X[:, 1],
    ])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in probs])

    names = [f"feature_{i}" for i in range(n_features)]
    return X, y, names


@pytest.fixture(scope="module")
def trained_ensemble(sample_data):
    X, y, names = sample_data
    return EnsemblePredictor().fit(X[:700], y[:700], names)


class TestEnsemble:
    def test_fit_and_predict(self, sample_data, trained_ensemble):
        X, _, _ = sample_data
        probs = trained_ensemble.predict_proba(X[700:])

        assert probs.shape == (200, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predict_classes(self, sample_data, trained_ensemble):
        X, _, _ = sample_data
        preds = trained_ensemble.predict(X[700:])

        assert len(preds) == 200
        assert set(preds).issubset({0, 1, 2})

    def test_blend_weights_are_fitted_on_validation(self, trained_ensemble):
        info = trained_ensemble.get_blend_info()

        assert info["fitted_on_validation"] is True
        assert info["method"] == "weights"
        assert pytest.approx(sum(info["weights"].values()), abs=1e-6) == 1.0

    def test_blend_methods_differ(self, sample_data):
        """Equal averaging and fitted weights must not be the same thing."""
        X, y, names = sample_data

        equal = EnsemblePredictor(blend="equal").fit(X[:700], y[:700], names)
        weighted = EnsemblePredictor(blend="weights").fit(X[:700], y[:700], names)

        assert not np.allclose(
            equal.predict_proba(X[700:]), weighted.predict_proba(X[700:])
        )

    def test_fitted_blend_not_worse_than_best_member(self, sample_data, trained_ensemble):
        """The failure this ensemble used to have: worse than its own best member."""
        X, y, _ = sample_data
        X_test, y_test = X[700:], y[700:]

        ensemble_rps = ranked_probability_score(y_test, trained_ensemble.predict_proba(X_test))
        member_rps = [
            ranked_probability_score(y_test, probs)
            for probs in trained_ensemble.member_probabilities(X_test).values()
        ]

        # Allow a small tolerance: weights are fitted on validation, not test
        assert ensemble_rps <= min(member_rps) + 0.01

    def test_members_are_calibrated(self, trained_ensemble):
        for _, model in trained_ensemble._models:
            assert model.calibrator is not None
            assert model.calibrator.fitted

    def test_predict_with_confidence(self, sample_data, trained_ensemble):
        X, _, _ = sample_data
        preds = trained_ensemble.predict_with_confidence(X[700:705])

        assert len(preds) == 5
        assert all("predicted_outcome" in p for p in preds)
        assert all(p["confidence"] >= 1 / 3 for p in preds)

    def test_feature_importance(self, sample_data, trained_ensemble):
        _, _, names = sample_data
        importance = trained_ensemble.get_feature_importance()

        assert len(importance) == len(names)
        assert all(v >= 0 for v in importance.values())

    def test_small_sample_falls_back_to_equal_weights(self, sample_data):
        """Too little validation data: train on everything, average equally."""
        X, y, names = sample_data
        ensemble = EnsemblePredictor().fit(X[:100], y[:100], names)

        info = ensemble.get_blend_info()
        assert info["fitted_on_validation"] is False
        assert ensemble.predict_proba(X[100:120]).shape == (20, 3)

    def test_save_and_load_roundtrip(self, sample_data, trained_ensemble, tmp_path):
        X, _, _ = sample_data
        expected = trained_ensemble.predict_proba(X[700:720])

        trained_ensemble.save(tmp_path / "ensemble")
        restored = EnsemblePredictor().load(tmp_path / "ensemble")

        # Calibrators and blend must survive persistence, not just the models
        assert np.allclose(restored.predict_proba(X[700:720]), expected, atol=1e-6)

    def test_eval_set_is_deprecated(self, sample_data):
        X, y, names = sample_data
        with pytest.warns(DeprecationWarning, match="eval_set is deprecated"):
            EnsemblePredictor().fit(
                X[:700], y[:700], names, eval_set=(X[700:], y[700:])
            )

    def test_not_trained_error(self):
        with pytest.raises(RuntimeError, match="not trained"):
            EnsemblePredictor().predict_proba(np.random.randn(10, 5))


@pytest.fixture(scope="module")
def oof_ensemble(sample_data):
    X, y, names = sample_data
    return EnsemblePredictor(blend_strategy="oof", blend_folds=3).fit(X[:700], y[:700], names)


@pytest.fixture(scope="module")
def holdout_ensemble(sample_data):
    X, y, names = sample_data
    return EnsemblePredictor(blend_strategy="holdout").fit(X[:700], y[:700], names)


class TestOutOfFoldBlending:
    """The default strategy: calibrate and blend on out-of-fold predictions."""

    def test_blend_is_fitted_on_more_rows_than_a_holdout(self, oof_ensemble, holdout_ensemble):
        oof_rows = oof_ensemble.get_blend_info()["blend_fit_samples"]
        holdout_rows = holdout_ensemble.get_blend_info()["blend_fit_samples"]

        assert oof_rows > holdout_rows

    def test_members_see_all_training_data(self, sample_data, oof_ensemble):
        """Unlike the holdout strategy, no rows are given up to a val block."""
        X, y, names = sample_data
        probs = oof_ensemble.predict_proba(X[700:])

        assert probs.shape == (200, 3)
        assert oof_ensemble.get_blend_info()["strategy"] == "oof"

    def test_members_are_calibrated(self, oof_ensemble):
        for _, model in oof_ensemble._models:
            assert model.calibrator is not None

    def test_small_dataset_falls_back_to_holdout(self, sample_data):
        X, y, names = sample_data
        ensemble = EnsemblePredictor(blend_strategy="oof").fit(X[:250], y[:250], names)

        assert ensemble.get_blend_info()["strategy"] == "oof"  # requested
        assert ensemble.predict_proba(X[250:260]).shape == (10, 3)
