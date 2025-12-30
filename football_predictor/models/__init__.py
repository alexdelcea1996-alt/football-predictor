"""Models package for football predictor."""

# Lazy imports to handle missing dependencies
try:
    from football_predictor.models.catboost_model import CatBoostModel
except ImportError:
    CatBoostModel = None  # type: ignore

try:
    from football_predictor.models.xgboost_model import XGBoostModel
except ImportError:
    XGBoostModel = None  # type: ignore

try:
    from football_predictor.models.logistic_model import LogisticModel
except ImportError:
    LogisticModel = None  # type: ignore

from football_predictor.models.ensemble import EnsemblePredictor

__all__ = [
    "CatBoostModel",
    "XGBoostModel",
    "LogisticModel",
    "EnsemblePredictor",
]
