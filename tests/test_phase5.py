"""Phase 5: Integration tests — full pipeline EDA + ML, agent initialization."""

import pytest

from agent.agent import DataMindAgent
from agent.tools import TOOLS_SCHEMA, execute_tool

# --- Tools schema ---


def test_tools_schema_contains_new_tools():
    tool_names = [t["function"]["name"] for t in TOOLS_SCHEMA]
    assert "detect_target_and_task" in tool_names
    assert "suggest_ml_pipeline" in tool_names
    assert len(tool_names) == 10


# --- detect_target_and_task ---


def test_detect_target_default(sample_df):
    res = execute_tool("detect_target_and_task", {}, sample_df)
    assert res["success"]
    assert len(res["text"]) > 0
    assert "Heart Disease" in res["text"]
    assert "classification" in res["text"]


def test_detect_target_forced(sample_df):
    res = execute_tool("detect_target_and_task", {"target_column": "Heart Disease"}, sample_df)
    assert res["success"]
    assert "user_specified" in res["text"]


def test_detect_target_invalid(sample_df):
    res = execute_tool("detect_target_and_task", {"target_column": "FAKE_COL"}, sample_df)
    assert not res["success"]
    assert "introuvable" in res["text"]


# --- suggest_ml_pipeline ---


def test_suggest_ml_default(sample_df):
    res = execute_tool("suggest_ml_pipeline", {}, sample_df)
    assert res["success"]
    assert (
        "Logistic Regression" in res["text"]
        or "Random Forest" in res["text"]
        or "Gradient Boosting" in res["text"]
    )
    assert "n_estimators" in res["text"] or "max_depth" in res["text"] or "C" in res["text"]
    assert "Fold" in res["text"] or "split" in res["text"]
    assert "ROC-AUC" in res["text"] or "RMSE" in res["text"]


def test_suggest_ml_forced(sample_df):
    res = execute_tool("suggest_ml_pipeline", {"target_column": "Heart Disease"}, sample_df)
    assert res["success"]


# --- Full EDA pipeline ---


def test_full_eda_pipeline(sample_df):
    assert execute_tool("describe_data", {}, sample_df)["success"]
    assert execute_tool("show_correlation", {}, sample_df)["success"]
    assert execute_tool("detect_outliers", {"column": "Age"}, sample_df)["success"]
    assert execute_tool("detect_target_and_task", {}, sample_df)["success"]
    assert execute_tool("suggest_ml_pipeline", {}, sample_df)["success"]


# --- Regression pipeline ---


def test_regression_pipeline(housing_regression_df):
    res_target = execute_tool("detect_target_and_task", {}, housing_regression_df)
    assert res_target["success"]
    assert "regression" in res_target["text"].lower()

    res_ml = execute_tool("suggest_ml_pipeline", {}, housing_regression_df)
    assert res_ml["success"]
    assert "RMSE" in res_ml["text"] or "R2" in res_ml["text"]


# --- Agent initialization ---


def test_agent_initialization(sample_df):
    agent = DataMindAgent()
    agent.set_data(sample_df)
    assert agent.df is not None
    assert len(agent.messages) == 2
    assert agent.messages[0]["role"] == "system"


# --- All EDA tools on sample ---


@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("describe_data", {}),
        ("describe_data", {"column": "Age"}),
        ("show_distribution", {"column": "Age"}),
        ("show_correlation", {}),
        ("detect_outliers", {"column": "Cholesterol"}),
        ("show_categorical", {"column": "Heart Disease"}),
        ("show_scatter", {"x_column": "Age", "y_column": "BP"}),
        ("compare_groups", {"categorical_column": "Heart Disease", "numeric_column": "Age"}),
    ],
)
def test_all_eda_tools(tool_name, args, sample_df):
    res = execute_tool(tool_name, args, sample_df)
    assert res["success"], f"Failed: {tool_name}({args}): {res['text'][:100]}"


# --- Unknown tool ---


def test_unknown_tool(sample_df):
    res = execute_tool("fake_tool", {}, sample_df)
    assert not res["success"]
    assert "inconnu" in res["text"].lower()
