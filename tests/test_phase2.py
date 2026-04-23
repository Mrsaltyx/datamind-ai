import sys
import os
import io
import traceback
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

    def assert_no_exception(self, name, func):
        try:
            func()
            self.ok(name)
        except Exception as e:
            self.fail(name, f"{type(e).__name__}: {e}")

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


def test_phase2(result: TestResult):
    print("\n=== PHASE 2: Robustesse du pipeline ===\n")
    df = load_fresh_df()

    from agent.tools import execute_tool

    print("--- 2.1 Validation des colonnes ---")
    res = execute_tool("show_distribution", {"column": "COLONNE_INEXISTANTE"}, df)
    result.assert_true("colonne inexistante: succes=False", not res["success"])
    result.assert_true(
        "colonne inexistante: message d'erreur", "introuvable" in res["text"]
    )

    res = execute_tool("show_distribution", {"column": "Heart Disease"}, df)
    result.assert_true("colonne non-numerique: succes=False", not res["success"])
    result.assert_true(
        "colonne non-numerique: message d'erreur", "numerique" in res["text"].lower()
    )

    res = execute_tool("detect_outliers", {"column": "Age"}, df)
    result.assert_true("detect_outliers colonne valide: succes=True", res["success"])

    res = execute_tool("describe_data", {"column": "FAKE_COL"}, df)
    result.assert_true(
        "describe_data colonne invalide: succes=False", not res["success"]
    )

    res = execute_tool("show_scatter", {}, df)
    result.assert_true("show_scatter sans params: succes=False", not res["success"])

    res = execute_tool("show_scatter", {"x_column": "Age", "y_column": "FAKE"}, df)
    result.assert_true(
        "show_scatter colonne y invalide: succes=False", not res["success"]
    )

    print("\n--- 2.2 Null safety ---")
    df_nan = df.head(100).copy()
    df_nan["Age"] = np.nan
    res = execute_tool("show_distribution", {"column": "Age"}, df_nan)
    result.assert_true(
        "distribution colonne toute NaN: succes=False", not res["success"]
    )
    result.assert_true(
        "distribution colonne toute NaN: message vide", "vide" in res["text"].lower()
    )

    res = execute_tool("detect_outliers", {"column": "Age"}, df_nan)
    result.assert_true("outliers colonne toute NaN: succes=False", not res["success"])

    print("\n--- 2.3 IQR zero edge case ---")
    df_const = pd.DataFrame({"const": [42.0] * 100})
    res = execute_tool("detect_outliers", {"column": "const"}, df_const)
    result.assert_true("IQR=0: succes=True (pas crash)", res["success"])
    result.assert_true("IQR=0: message informatif", "iqr" in res["text"].lower())

    print("\n--- 2.4 Context trimming ---")
    from agent.agent import DataMindAgent

    agent = DataMindAgent()
    agent.set_data(df)
    initial_count = len(agent.messages)
    for i in range(50):
        agent.messages.append({"role": "user", "content": f"msg {i}"})
        agent.messages.append({"role": "assistant", "content": f"rep {i}"})
    result.assert_true("context depasse max", len(agent.messages) > 30)
    agent._trim_context()
    result.assert_true("context tronque a max", len(agent.messages) <= 30)
    system_count = sum(1 for m in agent.messages if m.get("role") == "system")
    result.assert_eq("system messages preserves", system_count, 2)

    print("\n--- 2.5 Chargement CSV robustesse ---")
    from utils.data_loader import load_csv as loader_load

    csv_utf8 = "col1,col2\n1,2\n3,4\n".encode("utf-8")
    mock_file = io.BytesIO(csv_utf8)
    mock_file.getvalue = lambda: csv_utf8
    df_loaded = loader_load(mock_file)
    result.assert_eq("csv utf-8 charge", df_loaded.shape[0], 2)

    csv_semicolon = "a;b\n10;20\n30;40\n".encode("utf-8")
    mock_file2 = io.BytesIO(csv_semicolon)
    mock_file2.getvalue = lambda: csv_semicolon
    df_semi = loader_load(mock_file2)
    result.assert_eq("csv point-virgule charge", df_semi.shape[0], 2)

    csv_empty = "col1\n".encode("utf-8")
    mock_empty = io.BytesIO(csv_empty)
    mock_empty.getvalue = lambda: csv_empty

    try:
        loader_load(mock_empty)
        result.fail("csv vide: devrait lever une erreur", "aucune exception levee")
    except ValueError:
        result.ok("csv vide: leve ValueError")

    csv_latin1 = "nom,age\n".encode("latin-1")
    csv_latin1 += "R\xe9my,30\nPaul,25\n".encode("latin-1")
    mock_latin = io.BytesIO(csv_latin1)
    mock_latin.getvalue = lambda: csv_latin1
    df_lat = loader_load(mock_latin)
    result.assert_eq("csv latin-1 charge", df_lat.shape[0], 2)

    print("\n--- 2.6 High cardinality protection ---")
    df_high = df.head(200).copy()
    df_high["rand_str"] = [f"val_{i}_{np.random.rand()}" for i in range(200)]
    res = execute_tool("show_categorical", {"column": "rand_str"}, df_high)
    result.assert_true(
        "high cardinality: refuse", not res["success"] or "100" in res["text"]
    )

    print("\n--- 2.7 Correlation avec peu de colonnes numeriques ---")
    df_one_num = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "c"]})
    res = execute_tool("show_correlation", {}, df_one_num)
    result.assert_true(
        "correlation 1 col numerique: message informatif", not res["success"]
    )


if __name__ == "__main__":
    t0 = time.time()
    r = TestResult()
    test_phase2(r)
    elapsed = time.time() - t0
    print(f"\nDuree: {elapsed:.2f}s")
    success = r.summary()
    sys.exit(0 if success else 1)
