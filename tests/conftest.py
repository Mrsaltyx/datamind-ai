"""Shared pytest fixtures for DataMind AI tests."""

import os

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_CSV_PATH = os.path.join(FIXTURES_DIR, "sample_heart.csv")


@pytest.fixture
def sample_df():
    """Load the sample heart disease dataset (100 rows, 15 cols)."""
    return pd.read_csv(SAMPLE_CSV_PATH)


@pytest.fixture
def sample_csv_bytes(sample_df):
    """Return sample_df as raw CSV bytes (for load_csv tests)."""
    return sample_df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def sample_csv_file(sample_csv_bytes):
    """Return a file-like object with getvalue() for load_csv."""

    class _FileWrapper:
        def __init__(self, content: bytes):
            self._content = content

        def getvalue(self) -> bytes:
            return self._content

    return _FileWrapper(sample_csv_bytes)


@pytest.fixture
def nan_df(sample_df):
    """Sample df with some NaN values injected."""
    df = sample_df.copy()
    df.loc[0:30, "Age"] = np.nan
    df.loc[0:20, "BP"] = np.nan
    return df


@pytest.fixture
def const_df():
    """DataFrame with a constant numeric column (IQR=0 edge case)."""
    return pd.DataFrame({"const": [42.0] * 100})


@pytest.fixture
def regression_df():
    """Synthetic regression dataset (500 rows)."""
    return pd.DataFrame(
        {
            "feature1": np.random.randn(500),
            "feature2": np.random.randn(500) * 100,
            "feature3": np.random.choice(["A", "B", "C", "D"], 500),
            "target_price": np.random.randn(500) * 50 + 200,
        }
    )


@pytest.fixture
def large_regression_df():
    """Larger regression dataset (5000 rows)."""
    return pd.DataFrame(
        {
            "f1": np.random.randn(5000),
            "f2": np.random.randn(5000) * 100,
            "f3": np.random.randn(5000) * 0.01,
            "target_price": np.random.randn(5000) * 50 + 200,
        }
    )


@pytest.fixture
def minimal_df():
    """Minimal 3-row, 2-col DataFrame."""
    return pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})


@pytest.fixture
def high_cardinality_df(sample_df):
    """Sample df with a high-cardinality string column."""
    df = sample_df.head(50).copy()
    df["rand_str"] = [f"val_{i}_{np.random.rand()}" for i in range(50)]
    return df


@pytest.fixture
def one_numeric_df():
    """DataFrame with only 1 numeric column."""
    return pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})


@pytest.fixture
def tiny_df():
    """5-row DataFrame for edge case testing."""
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": ["a", "b", "a", "b", "a"],
        }
    )


@pytest.fixture
def imbalanced_df():
    """Imbalanced classification dataset (900/100 split)."""
    return pd.DataFrame(
        {
            "f1": np.random.randn(1000),
            "f2": np.random.randn(1000),
            "target": ["A"] * 900 + ["B"] * 100,
        }
    )


@pytest.fixture
def housing_regression_df():
    """Housing-like regression dataset (2000 rows)."""
    return pd.DataFrame(
        {
            "taille_m2": np.random.randn(2000) * 30 + 100,
            "nb_chambres": np.random.randint(1, 6, 2000),
            "age_batiment": np.random.randint(0, 50, 2000),
            "quartier": np.random.choice(["A", "B", "C", "D", "E"], 2000),
            "prix": np.random.randn(2000) * 50000 + 300000,
        }
    )
