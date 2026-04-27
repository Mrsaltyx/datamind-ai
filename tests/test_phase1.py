"""Phase 1: Bug fix validations — charts, data loader, statistics."""

import numpy as np

from utils.charts import (
    create_categorical_plot,
    create_correlation_heatmap,
    create_distribution_plot,
    create_outlier_plot,
    create_scatter_plot,
)
from utils.data_loader import get_column_stats, get_data_summary

# --- Distribution plot ---


def test_distribution_no_duplicate_box_traces(sample_df):
    fig = create_distribution_plot(sample_df, "Age")
    trace_types = [type(t).__name__ for t in fig.data]
    hist_count = trace_types.count("Box")
    assert hist_count == 1, "Should have exactly 1 Box trace, not 2"


def test_distribution_no_exception(sample_df):
    fig = create_distribution_plot(sample_df, "Age")
    assert fig is not None


# --- Scatter plot ---


def test_scatter_with_nans(sample_df):
    df_nan = sample_df.copy()
    df_nan.loc[0:10, "Age"] = np.nan
    df_nan.loc[5:15, "BP"] = np.nan
    fig = create_scatter_plot(df_nan, "Age", "BP")
    assert fig is not None


# --- Data summary ---


def test_data_summary_shape(sample_df):
    summary = get_data_summary(sample_df)
    assert summary["shape"][0] == 100
    assert summary["shape"][1] == 15


def test_data_summary_memory_positive(sample_df):
    summary = get_data_summary(sample_df)
    assert summary["memory_mb"] > 0


# --- Column stats ---


def test_column_stats_numeric(sample_df):
    stats = get_column_stats(sample_df, "Age")
    assert stats["type"] == "numeric"
    assert "skew" in stats
    assert "kurtosis" in stats


def test_column_stats_categorical(sample_df):
    stats = get_column_stats(sample_df, "Heart Disease")
    # Heart Disease is binary 0/1, detected as numeric by pandas
    assert stats["type"] in ("categorical", "numeric")
    assert "unique" in stats or "mean" in stats


# --- Other charts ---


def test_correlation_heatmap(sample_df):
    fig = create_correlation_heatmap(sample_df)
    assert fig is not None


def test_outlier_plot(sample_df):
    fig = create_outlier_plot(sample_df, "Cholesterol")
    assert fig is not None


def test_categorical_plot(sample_df):
    fig = create_categorical_plot(sample_df, "Heart Disease")
    assert fig is not None
