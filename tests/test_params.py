"""Tests for tuned hyperparameter loading."""

import json

from football_predictor.models.params import load_tuned_params, resolve_param


class TestLoadTunedParams:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_tuned_params("xgboost", tmp_path / "nope.json") == {}

    def test_reads_model_section(self, tmp_path):
        path = tmp_path / "best_params.json"
        path.write_text(json.dumps({
            "xgboost": {"n_estimators": 570, "max_depth": 3},
            "results": {"xgboost": {"rps": 0.15}},
        }))

        assert load_tuned_params("xgboost", path) == {"n_estimators": 570, "max_depth": 3}

    def test_logreg_alias_resolves_to_logistic(self, tmp_path):
        path = tmp_path / "best_params.json"
        path.write_text(json.dumps({"logreg": {"C": 0.001, "solver": "saga"}}))

        assert load_tuned_params("logistic", path)["solver"] == "saga"

    def test_results_section_is_not_a_model(self, tmp_path):
        path = tmp_path / "best_params.json"
        path.write_text(json.dumps({"results": {"rps": 0.13}}))

        assert load_tuned_params("results", path) == {}

    def test_corrupt_file_returns_empty(self, tmp_path):
        path = tmp_path / "best_params.json"
        path.write_text("{not json")

        assert load_tuned_params("xgboost", path) == {}


class TestResolveParam:
    def test_precedence_order(self):
        tuned = {"depth": 3}

        assert resolve_param(7, tuned, "depth", 6) == 7      # explicit wins
        assert resolve_param(None, tuned, "depth", 6) == 3   # tuned next
        assert resolve_param(None, {}, "depth", 6) == 6      # default last

    def test_null_tuned_value_falls_through(self):
        assert resolve_param(None, {"depth": None}, "depth", 6) == 6
