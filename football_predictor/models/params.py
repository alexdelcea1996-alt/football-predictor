"""
Tuned hyperparameter loading.

Hyperparameter search (scripts/optimize_hyperparams.py) writes its results to
``models/best_params.json``. Without this module those results were only read
by one prediction script, so the training pipeline kept using the static
defaults from ``config.py`` and threw the tuning away.

Resolution order for every model parameter:
    explicit constructor argument > tuned params file > config.py default
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from football_predictor.config import get_settings

PARAMS_FILENAME = "best_params.json"

# Keys that are not model parameters
_NON_PARAM_KEYS = {"results", "metadata", "created_at", "data"}

# Accept both naming conventions used across the codebase
_ALIASES = {
    "logistic": ("logistic", "logreg", "logistic_regression"),
    "catboost": ("catboost",),
    "xgboost": ("xgboost", "xgb"),
    "poisson": ("poisson", "dixon_coles"),
}


def tuned_params_path(path: str | Path | None = None) -> Path:
    """Resolve the path of the tuned-parameters file."""
    if path is not None:
        return Path(path)
    return Path(get_settings().models_dir) / PARAMS_FILENAME


@lru_cache(maxsize=8)
def _load_file(path_str: str, mtime: float) -> dict[str, Any]:
    """Read and cache the params file (cache keyed on path + mtime)."""
    with open(path_str) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k not in _NON_PARAM_KEYS}


def load_tuned_params(
    model_name: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load tuned hyperparameters for a model, or {} when unavailable.

    Args:
        model_name: One of "catboost", "xgboost", "logistic", "poisson"
        path: Optional explicit path to a params JSON file

    Returns:
        Dict of hyperparameters (empty when the file or model key is missing)
    """
    resolved = tuned_params_path(path)
    if not resolved.exists():
        return {}

    try:
        data = _load_file(str(resolved), resolved.stat().st_mtime)
    except (json.JSONDecodeError, OSError):
        return {}

    for key in _ALIASES.get(model_name, (model_name,)):
        if key in data and isinstance(data[key], dict):
            return dict(data[key])
    return {}


def resolve_param(
    explicit: Any,
    tuned: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    """Apply the resolution order: explicit > tuned > default."""
    if explicit is not None:
        return explicit
    if key in tuned and tuned[key] is not None:
        return tuned[key]
    return default
