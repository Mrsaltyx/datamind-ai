"""Phase 4: ML Advisor — model suggestions, evaluation strategy, full reports."""


from utils.ml_advisor import (
    generate_ml_report,
    suggest_evaluation_strategy,
    suggest_models,
)

# --- Classification models ---


def test_classification_models(sample_df):
    models = suggest_models(sample_df, "Heart Disease", "binary_classification")
    assert len(models) >= 3
    top = models[0]
    for key in ("name", "score", "hyperparams", "strengths", "weaknesses", "reasons"):
        assert key in top


# --- Regression models ---


def test_regression_models(large_regression_df):
    models = suggest_models(large_regression_df, "target_price", "regression")
    assert len(models) > 0
    top_names = [m["name"] for m in models[:3]]
    assert any("regression" in n.lower() or "regressor" in n.lower() for n in top_names)


# --- Evaluation strategy classification ---


def test_eval_strategy_classification(sample_df):
    eval_strat = suggest_evaluation_strategy(
        "binary_classification", len(sample_df), sample_df["Heart Disease"]
    )
    assert len(eval_strat["metrics"]) > 0
    assert "method" in eval_strat["validation"]
    assert "n_folds" in eval_strat["validation"]
    assert any(m["name"] == "ROC-AUC" for m in eval_strat["metrics"])


# --- Evaluation strategy regression ---


def test_eval_strategy_regression():
    eval_reg = suggest_evaluation_strategy("regression", 5000)
    assert len(eval_reg["metrics"]) > 0
    assert any(m["name"] == "RMSE" for m in eval_reg["metrics"])
    assert any(m["name"] == "R2" for m in eval_reg["metrics"])


# --- Full ML report ---


def test_full_ml_report(sample_df):
    report = generate_ml_report(sample_df)
    assert report["success"]
    for key in (
        "target",
        "task_type",
        "preprocessing",
        "recommended_models",
        "evaluation_strategy",
        "summary",
        "feature_engineering",
    ):
        assert key in report
    assert report["target"]["target_column"] == "Heart Disease"
    assert "classification" in report["task_type"]["task_type"]
    assert len(report["recommended_models"]) >= 1


def test_forced_target_report(sample_df):
    report = generate_ml_report(sample_df, "Heart Disease")
    assert report["success"]
    assert report["target"]["target_column"] == "Heart Disease"


# --- Tiny dataset ---


def test_tiny_dataset_report(tiny_df):
    report = generate_ml_report(tiny_df)
    assert report["success"]
    assert len(report["recommended_models"]) >= 1
    eval_tiny = report["evaluation_strategy"]
    assert (
        "Leave-One-Out" in eval_tiny["validation"]["method"]
        or len(eval_tiny.get("warnings", [])) > 0
    )


# --- Imbalanced dataset ---


def test_imbalanced_detection(imbalanced_df):
    eval_imb = suggest_evaluation_strategy("binary_classification", 1000, imbalanced_df["target"])
    assert len(eval_imb.get("warnings", [])) > 0
    assert any("desequilibre" in w.lower() for w in eval_imb["warnings"])
