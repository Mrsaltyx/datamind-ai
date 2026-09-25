"""Verrouillage de la detection de cible : correspondance par token exact.

Regression issue du test grandeur nature (consolidated.csv, 1M lignes) :
le mot-cle "y" matchait la sous-chaine "type" -> Soil_Type elu cible.
"""

import numpy as np
import pandas as pd

from datamind.analysis.preprocessing import (
    _name_tokens,
    analyze_preprocessing_needs,
    detect_target_column,
)
from datamind.ml.trainer import train_baseline


def test_name_tokens_split():
    assert _name_tokens("Soil_Type") == ["soil", "type"]
    assert _name_tokens("target_price") == ["target", "price"]
    assert _name_tokens("Heart Disease") == ["heart", "disease"]


def test_keyword_y_matches_only_column_named_y():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "a"]})
    assert detect_target_column(df)["target_column"] == "y"


def test_no_false_positive_on_type_column():
    """Le 'y' de 'type' ne doit plus matcher : retour au fallback derniere colonne."""
    df = pd.DataFrame(
        {
            "Region": ["W", "E", "W", "E"],
            "Soil_Type": ["Sandy", "Clay", "Sandy", "Clay"],
            "Crop": ["Cotton", "Rice", "Cotton", "Rice"],
            "Yield": [1.0, 2.0, 1.0, 2.0],
        }
    )
    target = detect_target_column(df)
    assert target["target_column"] == "Yield"  # fallback : derniere colonne
    assert target["method"] == "last_column"


def test_agriculture_schema_not_soil_type():
    """Regression grandeur nature : schema type consolidated.csv."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "Region": rng.choice(["West", "East"], n),
            "Soil_Type": rng.choice(["Sandy", "Clay"], n),
            "Crop": rng.choice(["Cotton", "Rice"], n),
            "Days_to_Harvest": rng.integers(80, 200, n),
            "Yield_tons_per_hectare": rng.normal(5, 1, n),
        }
    )
    target = detect_target_column(df)
    assert target["target_column"] != "Soil_Type"


def test_humidity_not_flagged_as_id():
    """Regression : 'id' dans 'humidity' ne doit pas marquer la colonne comme ID."""
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame(
        {
            "humidity": np.linspace(0, 100, n),  # cardinalite elevee, contient 'id'
            "customer_id": [f"cust_{i}" for i in range(n)],
            "target": rng.integers(0, 2, n),
        }
    )
    needs = analyze_preprocessing_needs(df, "target")
    id_cols = [item["column"] for item in needs["id_columns"]]
    assert "customer_id" in id_cols
    assert "humidity" not in id_cols


def test_trainer_excludes_id_by_token_not_substring():
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame(
        {
            "humidity_pct": rng.normal(60, 10, n),
            "record_id": [f"r_{i}" for i in range(n)],
            "target": rng.integers(0, 2, n),
        }
    )
    result = train_baseline(df)
    assert result["success"]
    assert "record_id" in " ".join(result["warnings"]) or result["n_features"] == 1
