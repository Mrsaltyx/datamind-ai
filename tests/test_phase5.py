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


def test_phase5(result: TestResult):
    print("\n=== PHASE 5: Test d'integration complet ===\n")
    df = load_fresh_df()

    from agent.tools import execute_tool, TOOLS_SCHEMA

    print("--- 5.1 Schema des outils contient les nouveaux outils ---")
    tool_names = [t["function"]["name"] for t in TOOLS_SCHEMA]
    result.assert_true(
        "detect_target_and_task dans schema", "detect_target_and_task" in tool_names
    )
    result.assert_true(
        "suggest_ml_pipeline dans schema", "suggest_ml_pipeline" in tool_names
    )
    result.assert_eq("total outils = 10", len(tool_names), 10)
    print(f"    -> Outils disponibles: {tool_names}")

    print("\n--- 5.2 detect_target_and_task via execute_tool ---")
    res = execute_tool("detect_target_and_task", {}, df)
    result.assert_true("detect_target succes", res["success"])
    result.assert_true("detect_target a du texte", len(res["text"]) > 0)
    result.assert_true(
        "detect_target contient Heart Disease", "Heart Disease" in res["text"]
    )
    result.assert_true(
        "detect_target contient classification", "classification" in res["text"]
    )
    print(f"    -> Longueur resultat: {len(res['text'])} chars")

    print("\n--- 5.3 detect_target_and_task avec cible forcee ---")
    res_forced = execute_tool(
        "detect_target_and_task", {"target_column": "Heart Disease"}, df
    )
    result.assert_true("detect_target forced succes", res_forced["success"])
    result.assert_true(
        "detect_target forced contient user_specified",
        "user_specified" in res_forced["text"],
    )

    print("\n--- 5.4 detect_target_and_task avec cible invalide ---")
    res_invalid = execute_tool(
        "detect_target_and_task", {"target_column": "FAKE_COL"}, df
    )
    result.assert_true(
        "detect_target invalide: succes=False", not res_invalid["success"]
    )
    result.assert_true(
        "detect_target invalide: message erreur", "introuvable" in res_invalid["text"]
    )

    print("\n--- 5.5 suggest_ml_pipeline via execute_tool ---")
    res_ml = execute_tool("suggest_ml_pipeline", {}, df)
    result.assert_true("suggest_ml succes", res_ml["success"])
    result.assert_true(
        "suggest_ml contient modeles",
        "Gradient Boosting" in res_ml["text"] or "Random Forest" in res_ml["text"],
    )
    result.assert_true(
        "suggest_ml contient hyperparams",
        "n_estimators" in res_ml["text"] or "max_depth" in res_ml["text"],
    )
    result.assert_true(
        "suggest_ml contient validation",
        "Fold" in res_ml["text"] or "split" in res_ml["text"],
    )
    result.assert_true(
        "suggest_ml contient metriques",
        "ROC-AUC" in res_ml["text"] or "RMSE" in res_ml["text"],
    )
    print(f"    -> Longueur rapport ML: {len(res_ml['text'])} chars")
    first_line = res_ml["text"].split("\n")[0]
    print(f"    -> Premiere ligne: {first_line}")

    print("\n--- 5.6 suggest_ml_pipeline avec cible forcee ---")
    res_ml_forced = execute_tool(
        "suggest_ml_pipeline", {"target_column": "Heart Disease"}, df
    )
    result.assert_true("suggest_ml forced succes", res_ml_forced["success"])

    print("\n--- 5.7 Pipeline complet EDA + ML sur train.csv ---")
    desc_res = execute_tool("describe_data", {}, df)
    result.assert_true("describe_data dans pipeline complet", desc_res["success"])

    corr_res = execute_tool("show_correlation", {}, df)
    result.assert_true("correlation dans pipeline complet", corr_res["success"])

    outlier_res = execute_tool("detect_outliers", {"column": "Age"}, df)
    result.assert_true("outliers dans pipeline complet", outlier_res["success"])

    target_res = execute_tool("detect_target_and_task", {}, df)
    result.assert_true("target detection dans pipeline complet", target_res["success"])

    ml_res = execute_tool("suggest_ml_pipeline", {}, df)
    result.assert_true("ML pipeline dans pipeline complet", ml_res["success"])

    print("\n--- 5.8 Pipeline sur dataset regression synthetique ---")
    df_reg = pd.DataFrame(
        {
            "taille_m2": np.random.randn(2000) * 30 + 100,
            "nb_chambres": np.random.randint(1, 6, 2000),
            "age_batiment": np.random.randint(0, 50, 2000),
            "quartier": np.random.choice(["A", "B", "C", "D", "E"], 2000),
            "prix": np.random.randn(2000) * 50000 + 300000,
        }
    )
    res_target_reg = execute_tool("detect_target_and_task", {}, df_reg)
    result.assert_true("regression target detectee", res_target_reg["success"])
    result.assert_true(
        "regression type detecte", "regression" in res_target_reg["text"].lower()
    )

    res_ml_reg = execute_tool("suggest_ml_pipeline", {}, df_reg)
    result.assert_true("regression ML succes", res_ml_reg["success"])
    result.assert_true(
        "regression ML metriques",
        "RMSE" in res_ml_reg["text"] or "R2" in res_ml_reg["text"],
    )

    print("\n--- 5.9 Verification de l'agent (sans appel API) ---")
    from agent.agent import DataMindAgent

    agent = DataMindAgent()
    agent.set_data(df)
    result.assert_true("agent initialise", agent.df is not None)
    result.assert_eq("agent messages count", len(agent.messages), 2)
    result.assert_true(
        "agent premier message system", agent.messages[0]["role"] == "system"
    )

    print("\n--- 5.10 Tous les outils EDA fonctionnent sur train.csv ---")
    tools_to_test = [
        ("describe_data", {}),
        ("describe_data", {"column": "Age"}),
        ("show_distribution", {"column": "Age"}),
        ("show_correlation", {}),
        ("detect_outliers", {"column": "Cholesterol"}),
        ("show_categorical", {"column": "Heart Disease"}),
        ("show_scatter", {"x_column": "Age", "y_column": "BP"}),
        (
            "compare_groups",
            {"categorical_column": "Heart Disease", "numeric_column": "Age"},
        ),
    ]
    for tool_name, args in tools_to_test:
        res = execute_tool(tool_name, args, df)
        result.assert_true(
            f"{tool_name}({args}) -> success",
            res["success"],
            f"erreur: {res['text'][:100]}",
        )

    print("\n--- 5.11 Outil inconnu ---")
    res_unknown = execute_tool("fake_tool", {}, df)
    result.assert_true("outil inconnu: succes=False", not res_unknown["success"])
    result.assert_true(
        "outil inconnu: message", "inconnu" in res_unknown["text"].lower()
    )


if __name__ == "__main__":
    t0 = time.time()
    r = TestResult()
    test_phase5(r)
    elapsed = time.time() - t0
    print(f"\nDuree: {elapsed:.2f}s")
    success = r.summary()
    sys.exit(0 if success else 1)
