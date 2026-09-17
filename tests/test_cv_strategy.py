"""Tests for temporal cross-validation splitting."""

import numpy as np
import pytest

from football_predictor.training.cv_strategy import TemporalCrossValidator


@pytest.fixture
def data():
    n = 1420
    return np.zeros((n, 3)), np.zeros(n)


class TestTemporalCrossValidator:
    def test_training_blocks_are_substantial(self, data):
        """Regression: folds used to train on ~100 rows and test on 355."""
        X, y = data
        folds = list(TemporalCrossValidator(n_splits=4).split(X, y))

        assert len(folds) == 4
        for train_idx, test_idx in folds:
            assert len(train_idx) >= 0.4 * len(X)
            assert len(train_idx) > len(test_idx)

    def test_no_look_ahead(self, data):
        X, y = data
        for train_idx, test_idx in TemporalCrossValidator(n_splits=4).split(X, y):
            assert train_idx.max() < test_idx.min()

    def test_folds_are_chronological_and_expanding(self, data):
        X, y = data
        folds = list(TemporalCrossValidator(n_splits=4).split(X, y))

        train_sizes = [len(train) for train, _ in folds]
        test_starts = [test[0] for _, test in folds]

        assert train_sizes == sorted(train_sizes)
        assert test_starts == sorted(test_starts)

    def test_test_blocks_do_not_overlap(self, data):
        X, y = data
        seen: set[int] = set()
        for _, test_idx in TemporalCrossValidator(n_splits=4).split(X, y):
            assert not seen & set(test_idx.tolist())
            seen.update(test_idx.tolist())

    def test_absolute_min_train_size_is_honoured(self, data):
        X, y = data
        folds = list(TemporalCrossValidator(n_splits=3, min_train_size=900).split(X, y))

        assert folds
        assert all(len(train) >= 900 for train, _ in folds)

    def test_gap_is_applied(self, data):
        X, y = data
        for train_idx, test_idx in TemporalCrossValidator(n_splits=3, gap=50).split(X, y):
            assert test_idx.min() - train_idx.max() > 50

    def test_small_dataset_still_yields_folds(self):
        X, y = np.zeros((60, 2)), np.zeros(60)
        folds = list(TemporalCrossValidator(n_splits=3).split(X, y))

        assert folds
        assert all(len(test) > 0 for _, test in folds)
