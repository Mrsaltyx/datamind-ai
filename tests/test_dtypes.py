"""Verrouillage du fix dtype str/pandas 3 : les colonnes string ne doivent
plus etre invisibles pour la detection de tache et le preprocessing."""

import numpy as np
import pandas as pd

from datamind.analysis.data_loader import get_data_summary
from datamind.analysis.dtypes import categorical_columns, is_categorical_dtype
from datamind.analysis.preprocessing import (
    analyze_preprocessing_needs,
    detect_task_type,
)


def test_str_dtype_is_categorical():
    s = pd.Series(["a", "b", "a"])
    # pandas 3 : dtype str (numpy-backed), plus object
    assert is_categorical_dtype(s.dtype)
    assert "str" in str(s.dtype) or is_categorical_dtype(s.dtype)


def test_categorical_columns_with_strings():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    assert categorical_columns(df) == ["y"]


def test_string_target_is_classification():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "a", "b"]})
    task = detect_task_type(df, "y")
    assert "classification" in task["task_type"]


def test_string_feature_flagged_for_encoding():
    df = pd.DataFrame(
        {
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "f3": np.random.choice(["A", "B"], 100),
            "target_price": np.random.randn(100),
        }
    )
    needs = analyze_preprocessing_needs(df, "target_price")
    encoded = [item["column"] for item in needs["encoding"]]
    assert "f3" in encoded


def test_data_summary_reports_string_columns():
    df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    summary = get_data_summary(df)
    assert summary["categorical_cols"] == ["y"]


def test_tiny_string_dataset_full_report():
    """Regression : le rapport ML complet doit reussir sur un tiny dataset."""
    from datamind.analysis.ml_advisor import generate_ml_report

    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "a", "b", "a"]})
    report = generate_ml_report(df)
    assert report["success"]
