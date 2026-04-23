import sys
import os
import time
import json

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


def test_phase4(result: TestResult):
    print("\n=== PHASE 4: Module ML Advisor ===\n")
    df = load_fresh_df()

    from utils.ml_advisor import (
        suggest_models,
        suggest_evaluation_strategy,
        generate_ml_report,
    )

    print("--- 4.1 Suggestion de modeles (classification) ---")
    models = suggest_models(df, "Heart Disease", "binary_classification")
    result.assert_true("modeles retournes", len(models) > 0)
    result.assert_true("top 3 modeles", len(models) >= 3)

    top = models[0]
    result.assert_true("modele a un nom", "name" in top)
    result.assert_true("modele a un score", "score" in top)
    result.assert_true("modele a des hyperparams", "hyperparams" in top)
    result.assert_true("modele a des forces", "strengths" in top)
    result.assert_true("modele a des faiblesses", "weaknesses" in top)
    result.assert_true("modele a des raisons", "reasons" in top)

    print(f"    -> Top 3 modeles:")
    for m in models[:3]:
        print(
            f"       {m['name']} (score: {m['score']}, complexite: {m['complexity']})"
        )

    print("\n--- 4.2 Suggestion de modeles (regression) ---")
    df_reg = pd.DataFrame(
        {
            "f1": np.random.randn(5000),
            "f2": np.random.randn(5000) * 100,
            "f3": np.random.randn(5000) * 0.01,
            "target_price": np.random.randn(5000) * 50 + 200,
        }
    )
    models_reg = suggest_models(df_reg, "target_price", "regression")
    result.assert_true("modeles regression retournes", len(models_reg) > 0)
    top_reg_names = [m["name"] for m in models_reg[:3]]
    result.assert_true(
        "modeles regression pertinents",
        any(
            "regression" in n.lower() or "regressor" in n.lower() for n in top_reg_names
        ),
    )
    for m in models_reg[:3]:
        print(f"       {m['name']} (score: {m['score']})")

    print("\n--- 4.3 Strategie d'evaluation ---")
    eval_strat = suggest_evaluation_strategy(
        "binary_classification", len(df), df["Heart Disease"]
    )
    result.assert_true("metriques definies", len(eval_strat["metrics"]) > 0)
    result.assert_true("validation definie", "method" in eval_strat["validation"])
    result.assert_true("validation n_folds", "n_folds" in eval_strat["validation"])
    result.assert_true(
        "ROC-AUC dans metriques",
        any(m["name"] == "ROC-AUC" for m in eval_strat["metrics"]),
    )
    print(f"    -> Methode: {eval_strat['validation']['method']}")
    print(f"    -> Metriques: {[m['name'] for m in eval_strat['metrics']]}")

    print("\n--- 4.4 Strategie d'evaluation (regression) ---")
    eval_reg = suggest_evaluation_strategy("regression", 5000)
    result.assert_true("metriques regression", len(eval_reg["metrics"]) > 0)
    result.assert_true(
        "RMSE dans metriques", any(m["name"] == "RMSE" for m in eval_reg["metrics"])
    )
    result.assert_true(
        "R2 dans metriques", any(m["name"] == "R2" for m in eval_reg["metrics"])
    )
    print(f"    -> Metriques: {[m['name'] for m in eval_reg['metrics']]}")

    print("\n--- 4.5 Rapport ML complet (train.csv) ---")
    report = generate_ml_report(df)
    result.assert_true("rapport succes", report["success"])
    result.assert_true("rapport a target", "target" in report)
    result.assert_true("rapport a task_type", "task_type" in report)
    result.assert_true("rapport a preprocessing", "preprocessing" in report)
    result.assert_true("rapport a modeles", "recommended_models" in report)
    result.assert_true("rapport a evaluation", "evaluation_strategy" in report)
    result.assert_true("rapport a summary", "summary" in report)
    result.assert_true("rapport a feature_engineering", "feature_engineering" in report)
    result.assert_eq(
        "rapport target correct", report["target"]["target_column"], "Heart Disease"
    )
    result.assert_true(
        "rapport classification", "classification" in report["task_type"]["task_type"]
    )
    result.assert_true(
        "rapport top modeles >= 1", len(report["recommended_models"]) >= 1
    )

    print(f"\n{report['summary']}")

    print("\n--- 4.6 Rapport avec cible forcee ---")
    report_forced = generate_ml_report(df, "Heart Disease")
    result.assert_true("rapport force succes", report_forced["success"])
    result.assert_eq(
        "rapport force target",
        report_forced["target"]["target_column"],
        "Heart Disease",
    )

    print("\n--- 4.7 Dataset petit (edge case) ---")
    df_tiny = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": ["a", "b", "a", "b", "a"],
        }
    )
    report_tiny = generate_ml_report(df_tiny)
    result.assert_true("rapport tiny succes", report_tiny["success"])
    result.assert_true(
        "rapport tiny modeles", len(report_tiny["recommended_models"]) >= 1
    )
    eval_tiny = report_tiny["evaluation_strategy"]
    result.assert_true(
        "tiny: Leave-One-Out ou warning",
        "Leave-One-Out" in eval_tiny["validation"]["method"]
        or len(eval_tiny.get("warnings", [])) > 0,
    )

    print("\n--- 4.8 Dataset desequilibre ---")
    df_imb = pd.DataFrame(
        {
            "f1": np.random.randn(1000),
            "f2": np.random.randn(1000),
            "target": ["A"] * 900 + ["B"] * 100,
        }
    )
    eval_imb = suggest_evaluation_strategy(
        "binary_classification", 1000, pd.Series(df_imb["target"])
    )
    result.assert_true("desequilibre detecte", len(eval_imb.get("warnings", [])) > 0)
    result.assert_true(
        "warning desequilibre",
        any("desequilibre" in w.lower() for w in eval_imb["warnings"]),
    )


if __name__ == "__main__":
    t0 = time.time()
    r = TestResult()
    test_phase4(r)
    elapsed = time.time() - t0
    print(f"\nDuree: {elapsed:.2f}s")
    success = r.summary()
    sys.exit(0 if success else 1)
