import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.csv"
)


def load_fresh_df():
    return pd.read_csv(DATASET_PATH)


class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name):
        self.passed.append(name)
        print(f"  [PASS] {name}")

    def fail(self, name, err):
        self.failed.append((name, err))
        print(f"  [FAIL] {name}: {err}")

    def assert_true(self, name, condition, detail=""):
        if condition:
            self.ok(name)
        else:
            self.fail(name, detail or "Assertion fausse")

    def assert_eq(self, name, actual, expected):
        if actual == expected:
            self.ok(name)
        else:
            self.fail(name, f"attendu={expected}, obtenu={actual}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'=' * 60}")
        print(f"RESULTAT: {len(self.passed)}/{total} tests reussis")
        if self.failed:
            print(f"\nTests echoues:")
            for name, err in self.failed:
                print(f"  - {name}: {err}")
        print(f"{'=' * 60}")
        return len(self.failed) == 0


def test_phase3(result: TestResult):
    print("\n=== PHASE 3: Module Preprocessing ===\n")
    df = load_fresh_df()

    from utils.preprocessing import (
        detect_target_column,
        detect_task_type,
        analyze_preprocessing_needs,
        suggest_feature_engineering,
    )

    print("--- 3.1 Detection de la cible ---")
    target_info = detect_target_column(df)
    result.assert_eq("target detectee", target_info["target_column"], "Heart Disease")
    result.assert_true("confidence > 0", target_info["confidence"] > 0)
    result.assert_true(
        "method defini", target_info["method"] in ("keyword", "last_column")
    )
    print(
        f"    -> Colonne cible: {target_info['target_column']} (confiance: {target_info['confidence']}, methode: {target_info['method']})"
    )

    print("\n--- 3.2 Detection du type de tache ---")
    task_info = detect_task_type(df, "Heart Disease")
    result.assert_true(
        "classification binaire detectee", "classification" in task_info["task_type"]
    )
    result.assert_eq("nunique = 2", task_info["nunique"], 2)
    result.assert_true(
        "subtype defini",
        task_info["subtype"]
        in (
            "categorical_binary",
            "numeric_binary",
            "categorical_multi",
            "low_cardinality_numeric",
            "continuous_numeric",
        ),
    )
    print(f"    -> Type de tache: {task_info['task_type']} ({task_info['subtype']})")

    print("\n--- 3.3 Analyse des besoins de preprocessing ---")
    needs = analyze_preprocessing_needs(df, "Heart Disease")
    result.assert_true("cle 'missing_values' presente", "missing_values" in needs)
    result.assert_true("cle 'encoding' presente", "encoding" in needs)
    result.assert_true("cle 'scaling' presente", "scaling" in needs)
    result.assert_true("cle 'id_columns' presente", "id_columns" in needs)
    result.assert_true("cle 'warnings' presente", "warnings" in needs)

    id_cols = [item["column"] for item in needs["id_columns"]]
    result.assert_true("colonne 'id' detectee comme ID", "id" in id_cols)
    print(f"    -> Colonnes ID: {id_cols}")
    print(f"    -> Colonnes avec valeurs manquantes: {len(needs['missing_values'])}")
    print(f"    -> Colonnes a encoder: {len(needs['encoding'])}")
    print(f"    -> Transformations suggerees: {len(needs['transformation'])}")

    print("\n--- 3.4 Feature engineering ---")
    suggestions = suggest_feature_engineering(
        df, "Heart Disease", task_info["task_type"]
    )
    result.assert_true("suggestions retournees", isinstance(suggestions, list))
    if suggestions:
        for s in suggestions:
            result.assert_true("type defini", "type" in s)
            result.assert_true("source definie", "source" in s)
            break
    print(f"    -> {len(suggestions)} suggestions de feature engineering")

    print("\n--- 3.5 Test avec dataset regression synthetique ---")
    df_reg = pd.DataFrame(
        {
            "feature1": np.random.randn(500),
            "feature2": np.random.randn(500) * 100,
            "feature3": np.random.choice(["A", "B", "C", "D"], 500),
            "target_price": np.random.randn(500) * 50 + 200,
        }
    )
    target_reg = detect_target_column(df_reg)
    result.assert_true(
        "target regression detectee", "price" in target_reg["target_column"].lower()
    )
    task_reg = detect_task_type(df_reg, target_reg["target_column"])
    result.assert_eq("regression detectee", task_reg["task_type"], "regression")
    needs_reg = analyze_preprocessing_needs(df_reg, target_reg["target_column"])
    result.assert_true(
        "encoding suggere pour feature3",
        any(item["column"] == "feature3" for item in needs_reg["encoding"]),
    )
    result.assert_true("scaling suggere", len(needs_reg["scaling"]) > 0)
    print(
        f"    -> Target: {target_reg['target_column']}, Task: {task_reg['task_type']}"
    )

    print("\n--- 3.6 Test avec dataset avec valeurs manquantes ---")
    df_nan = df.head(200).copy()
    df_nan.loc[0:30, "Age"] = np.nan
    df_nan.loc[0:20, "BP"] = np.nan
    df_nan["new_col"] = np.nan
    needs_nan = analyze_preprocessing_needs(df_nan, "Heart Disease")
    missing_cols = [item["column"] for item in needs_nan["missing_values"]]
    result.assert_true("Age dans missing_values", "Age" in missing_cols)
    result.assert_true("BP dans missing_values", "BP" in missing_cols)
    result.assert_true("new_col dans missing_values", "new_col" in missing_cols)
    new_col_action = [
        item for item in needs_nan["missing_values"] if item["column"] == "new_col"
    ][0]
    result.assert_true("new_col > 50% manquants", new_col_action["missing_pct"] > 50)
    print(f"    -> Colonnes avec NaN: {missing_cols}")

    print("\n--- 3.7 Edge case: dataset minimal ---")
    df_min = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    target_min = detect_target_column(df_min)
    result.assert_eq(
        "target minimal = derniere colonne", target_min["target_column"], "y"
    )
    task_min = detect_task_type(df_min, "y")
    result.assert_true(
        "classification detectee", "classification" in task_min["task_type"]
    )
    needs_min = analyze_preprocessing_needs(df_min, "y")
    result.assert_true("needs ne crash pas", isinstance(needs_min, dict))


if __name__ == "__main__":
    t0 = time.time()
    r = TestResult()
    test_phase3(r)
    elapsed = time.time() - t0
    print(f"\nDuree: {elapsed:.2f}s")
    success = r.summary()
    sys.exit(0 if success else 1)
