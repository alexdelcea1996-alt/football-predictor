"""Tests for ensemble predictor."""

import pytest
import numpy as np

from football_predictor.models.ensemble import EnsemblePredictor


class TestEnsemble:
    """Tests for EnsemblePredictor."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        return X, y, feature_names
    
    def test_fit_and_predict(self, sample_data):
        """Test basic fit and predict."""
        X, y, names = sample_data
        
        ensemble = EnsemblePredictor()
        ensemble.fit(X[:150], y[:150], names)
        
        probs = ensemble.predict_proba(X[150:])
        
        assert probs.shape == (50, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all() and (probs <= 1).all()
    
    def test_predict_classes(self, sample_data):
        """Test class prediction."""
        X, y, names = sample_data
        
        ensemble = EnsemblePredictor()
        ensemble.fit(X[:150], y[:150], names)
        
        preds = ensemble.predict(X[150:])
        
        assert len(preds) == 50
        assert set(preds).issubset({0, 1, 2})
    
    def test_weighted_averaging(self, sample_data):
        """Test weighted ensemble averaging."""
        X, y, names = sample_data
        
        # Equal weights
        ensemble1 = EnsemblePredictor(weights=[1.0, 1.0, 1.0])
        ensemble1.fit(X[:150], y[:150], names)
        
        # CatBoost heavy
        ensemble2 = EnsemblePredictor(weights=[0.8, 0.1, 0.1])
        ensemble2.fit(X[:150], y[:150], names)
        
        probs1 = ensemble1.predict_proba(X[150:160])
        probs2 = ensemble2.predict_proba(X[150:160])
        
        # Different weights should give different predictions
        assert not np.allclose(probs1, probs2)
    
    def test_predict_with_confidence(self, sample_data):
        """Test prediction with confidence output."""
        X, y, names = sample_data
        
        ensemble = EnsemblePredictor()
        ensemble.fit(X[:150], y[:150], names)
        
        preds = ensemble.predict_with_confidence(X[150:155])
        
        assert len(preds) == 5
        assert all("predicted_outcome" in p for p in preds)
        assert all("confidence" in p for p in preds)
        assert all(p["confidence"] >= 1/3 for p in preds)
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        X, y, names = sample_data
        
        ensemble = EnsemblePredictor()
        ensemble.fit(X[:150], y[:150], names)
        
        importance = ensemble.get_feature_importance()
        
        assert len(importance) == 20
        assert all(v >= 0 for v in importance.values())
    
    def test_not_trained_error(self):
        """Test error on prediction before training."""
        ensemble = EnsemblePredictor()
        
        with pytest.raises(RuntimeError, match="not trained"):
            ensemble.predict_proba(np.random.randn(10, 5))
