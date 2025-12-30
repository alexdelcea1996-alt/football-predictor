"""Features package for football predictor."""

from football_predictor.features.elo import EloRatingSystem
from football_predictor.features.rolling_stats import RollingStatsCalculator
from football_predictor.features.head_to_head import HeadToHeadCalculator
from football_predictor.features.contextual import ContextualFeatures
from football_predictor.features.aggregator import FeatureAggregator

__all__ = [
    "EloRatingSystem",
    "RollingStatsCalculator",
    "HeadToHeadCalculator",
    "ContextualFeatures",
    "FeatureAggregator",
]
