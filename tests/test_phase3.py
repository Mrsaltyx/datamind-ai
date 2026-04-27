"""Phase 3: Preprocessing module — target detection, task type, feature engineering."""


from utils.preprocessing import (
    analyze_preprocessing_needs,
    detect_target_column,
    detect_task_type,
    suggest_feature_engineering,
)

# --- Target detection ---


def test_target_detected(sample_df):
    target_info = detect_target_column(sample_df)
    assert target_info["target_column"] == "Heart Disease"
    assert target_info["confidence"] > 0
    assert target_info["method"] in ("keyword", "last_column")


# --- Task type detection ---


def test_classification_binary(sample_df):
    task_info = detect_task_type(sample_df, "Heart Disease")
    assert "classification" in task_info["task_type"]
    assert task_info["nunique"] == 2
    assert task_info["subtype"] in (
        "categorical_binary",
        "numeric_binary",
        "categorical_multi",
        "low_cardinality_numeric",
        "continuous_numeric",
    )


# --- Preprocessing needs ---


def test_preprocessing_needs_keys(sample_df):
    needs = analyze_preprocessing_needs(sample_df, "Heart Disease")
    for key in ("missing_values", "encoding", "scaling", "id_columns", "warnings"):
        assert key in needs


def test_id_column_detected(sample_df):
    needs = analyze_preprocessing_needs(sample_df, "Heart Disease")
    id_cols = [item["column"] for item in needs["id_columns"]]
    assert "id" in id_cols


# --- Feature engineering ---


def test_feature_engineering_returns_list(sample_df):
    task_info = detect_task_type(sample_df, "Heart Disease")
    suggestions = suggest_feature_engineering(sample_df, "Heart Disease", task_info["task_type"])
    assert isinstance(suggestions, list)
    if suggestions:
        assert "type" in suggestions[0]
        assert "source" in suggestions[0]


# --- Regression dataset ---


def test_regression_target_detected(regression_df):
    target_info = detect_target_column(regression_df)
    assert "price" in target_info["target_column"].lower()


def test_regression_task_type(regression_df):
    target_info = detect_target_column(regression_df)
    task_info = detect_task_type(regression_df, target_info["target_column"])
    assert task_info["task_type"] == "regression"


def test_regression_encoding_suggested(regression_df):
    target_info = detect_target_column(regression_df)
    needs = analyze_preprocessing_needs(regression_df, target_info["target_column"])
    assert any(item["column"] == "feature3" for item in needs["encoding"])


# --- Dataset with missing values ---


def test_missing_values_detected(nan_df):
    needs = analyze_preprocessing_needs(nan_df, "Heart Disease")
    missing_cols = [item["column"] for item in needs["missing_values"]]
    assert "Age" in missing_cols
    assert "BP" in missing_cols


# --- Minimal dataset ---


def test_minimal_dataset_no_crash(minimal_df):
    target = detect_target_column(minimal_df)
    assert target["target_column"] == "y"
    task = detect_task_type(minimal_df, "y")
    assert "classification" in task["task_type"]
    needs = analyze_preprocessing_needs(minimal_df, "y")
    assert isinstance(needs, dict)
