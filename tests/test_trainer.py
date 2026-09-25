"""Tests du trainer : baselines reelles entrainees sur datasets synthetiques."""

import numpy as np
import pandas as pd
import pytest

from datamind.ml.trainer import format_training_result, train_baseline

rng = np.random.default_rng(42)


def _binary_df(n=300):
    x1 = rng.normal(size=n)
    x2 = rng.choice(["A", "B", "C"], n)
    logit = 2 * x1 + (x2 == "B") * 1.0
    y = (logit + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": y})


def _multiclass_df(n=400):
    return pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "classe": rng.choice(["chat", "chien", "oiseau"], n),
        }
    )


def _regression_df(n=300):
    x1 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "surface": x1 * 30 + 100,
            "quartier": rng.choice(["A", "B"], n),
            "prix": x1 * 50000 + 300000 + rng.normal(size=n) * 10000,
        }
    )


# --- Cas nominaux ---


def test_binary_classification_trains():
    result = train_baseline(_binary_df())
    assert result["success"]
    assert result["task_type"] == "binary_classification"
    assert result["n_samples"] == 300
    names = [m["name"] for m in result["metrics"]]
    assert "F1" in names and "ROC-AUC" in names
    for m in result["metrics"]:
        assert 0.0 <= m["std"] < 1.0


def test_multiclass_trains():
    result = train_baseline(_multiclass_df())
    assert result["success"]
    names = [m["name"] for m in result["metrics"]]
    assert "F1 macro" in names and "Accuracy" in names


def test_regression_trains():
    result = train_baseline(_regression_df())
    assert result["success"]
    assert result["task_type"] == "regression"
    names = [m["name"] for m in result["metrics"]]
    assert "RMSE" in names and "R2" in names
    rmse = next(m for m in result["metrics"] if m["name"] == "RMSE")
    assert rmse["mean"] > 0


def test_target_column_override():
    df = _binary_df().rename(columns={"target": "survived"})
    result = train_baseline(df, target_column="survived")
    assert result["success"]
    assert result["target_column"] == "survived"


def test_string_target_supported():
    df = _binary_df()
    df["target"] = df["target"].map({0: "non", 1: "oui"})
    result = train_baseline(df)
    assert result["success"]


# --- Cas limites ---


def test_too_small_fails():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    result = train_baseline(df)
    assert not result["success"]
    assert "trop petit" in result["error"]


def test_single_class_fails():
    df = pd.DataFrame({"x": rng.normal(size=50), "target": [1] * 50})
    result = train_baseline(df)
    assert not result["success"]


def test_unknown_target_column_fails():
    result = train_baseline(_binary_df(), target_column="n existe pas")
    assert not result["success"]


def test_id_column_excluded_with_warning():
    n = 200
    df = _binary_df(n)
    df["customer_id"] = [f"id_{i}" for i in range(n)]
    result = train_baseline(df, target_column="target")
    assert result["success"]
    assert result["n_features"] == 2  # x1, x2 ; customer_id exclue
    assert any("customer_id" in w for w in result["warnings"])


def test_imbalanced_warning():
    n = 300
    df = pd.DataFrame(
        {
            "f": rng.normal(size=n),
            "target": ["A"] * 270 + ["B"] * 30,
        }
    )
    result = train_baseline(df)
    assert result["success"]
    assert any("desequilibre" in w for w in result["warnings"])


def test_missing_target_rows_dropped():
    df = _binary_df(100)
    df.loc[:9, "target"] = np.nan
    result = train_baseline(df)
    assert result["success"]
    assert result["n_samples"] == 90


# --- Formatage ---


def test_format_success():
    text = format_training_result(train_baseline(_binary_df()))
    assert "Baseline entrainee" in text
    assert "ROC-AUC" in text


def test_format_failure():
    text = format_training_result({"success": False, "error": "boom"})
    assert "boom" in text


@pytest.mark.parametrize("n", [10, 25, 60])
def test_minimum_sizes_still_train(n):
    """Propriete : des que le minimum est atteint, l'entrainement reussit."""
    df = _binary_df(max(n * 3, 30)).head(n)
    # garantit 2 classes et >= MIN_SAMPLES lignes
    df = pd.concat([df, df.head(max(0, 10 - n))], ignore_index=True) if n < 10 else df
    result = train_baseline(df)
    if len(df.dropna(subset=["target"])) >= 10:
        assert result["success"]
