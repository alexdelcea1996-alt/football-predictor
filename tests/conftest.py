"""Shared test configuration.

Keeps tests fast and independent of whatever is in the repository's
models/best_params.json by pointing the models directory at a temp folder and
shrinking the boosted models.
"""

import os
import tempfile

_TMP_MODELS = os.path.join(tempfile.gettempdir(), "fp_test_models")
os.makedirs(_TMP_MODELS, exist_ok=True)

os.environ.setdefault("MODELS_DIR", _TMP_MODELS)
os.environ.setdefault("FP_MODEL_CATBOOST_ITERATIONS", "100")
os.environ.setdefault("FP_MODEL_XGBOOST_N_ESTIMATORS", "100")
os.environ.setdefault("FP_MODEL_EARLY_STOPPING_ROUNDS", "10")
