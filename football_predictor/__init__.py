"""
Football Match Prediction System

A production-ready machine learning system for predicting football match outcomes
(Home Win, Draw, Away Win) with calibrated probability estimates.

Target: 52-56% accuracy with RPS < 0.22
"""

__version__ = "1.0.0"
__author__ = "Football Predictor Team"

from football_predictor.config import Settings

__all__ = ["Settings", "__version__"]
