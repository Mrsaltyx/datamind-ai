"""Tests du tracking MLflow optionnel : degrade proprement sans mlflow."""

import pytest

from datamind.ml.tracker import log_training, tracking_enabled


def _fake_result():
    return {
        "success": True,
        "task_type": "binary_classification",
        "model_name": "Logistic Regression",
        "target_column": "y",
        "n_samples": 100,
        "n_features": 3,
        "n_folds": 5,
        "metrics": [{"name": "F1", "mean": 0.5, "std": 0.1}],
        "warnings": [],
    }


def test_failed_training_not_tracked():
    tracking = log_training({"success": False, "error": "boom"})
    assert tracking == {"tracked": False, "reason": "training_failed"}


def test_tracking_degrades_gracefully_without_mlflow():
    """Sans mlflow installe : pas de crash, tracked=False."""
    tracking = log_training(_fake_result(), report_text="rapport")
    if tracking_enabled():
        pytest.skip("mlflow installe dans cet env : voir test_tracking_with_mlflow")
    assert tracking["tracked"] is False


@pytest.mark.skipif(not tracking_enabled(), reason="mlflow non installe")
def test_tracking_with_mlflow(tmp_path):
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    import os

    os.environ["DATAMIND_MLFLOW_URI"] = uri
    try:
        tracking = log_training(_fake_result(), report_text="rapport test")
        assert tracking["tracked"] is True
        assert tracking["run_id"]

        runs = mlflow.search_runs(experiment_names=["datamind-ai"])
        assert len(runs) == 1
        row = runs.iloc[0]
        assert row["metrics.F1_mean"] == 0.5
        assert row["params.task_type"] == "binary_classification"
    finally:
        os.environ.pop("DATAMIND_MLFLOW_URI", None)
