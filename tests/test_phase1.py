import sys
import os
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

    def assert_type(self, name, obj, expected_type):
        if isinstance(obj, expected_type):
            self.ok(name)
        else:
            self.fail(
                name,
                f"type attendu={expected_type.__name__}, obtenu={type(obj).__name__}",
            )

    def assert_no_exception(self, name, func):
        try:
            func()
            self.ok(name)
        except Exception as e:
            self.fail(name, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

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


def test_phase1(result: TestResult):
    print("\n=== PHASE 1: Corrections de bugs ===\n")
    df = load_fresh_df()

    from utils.charts import create_distribution_plot

    def t_distribution_no_duplicate_traces():
        fig = create_distribution_plot(df, "Age")
        trace_types = [type(t).__name__ for t in fig.data]
        hist_count = trace_types.count("Box")
        result.assert_eq(
            "distribution_plot: 1 seul Box trace (pas 2)",
            hist_count,
            1,
        )

    result.assert_no_exception(
        "distribution_plot: pas d'exception", t_distribution_no_duplicate_traces
    )

    from utils.charts import create_scatter_plot

    def t_scatter_with_nans():
        df_nan = df.copy()
        df_nan.loc[0:100, "Age"] = np.nan
        df_nan.loc[50:150, "BP"] = np.nan
        fig = create_scatter_plot(df_nan, "Age", "BP")
        result.assert_type("scatter_plot avec NaN: retourne Figure", fig, object)

    result.assert_no_exception("scatter_plot avec NaN convergents", t_scatter_with_nans)

    from utils.data_loader import get_data_summary, get_column_stats

    def t_summary_on_large_df():
        summary = get_data_summary(df)
        result.assert_eq("summary shape rows", summary["shape"][0], 630000)
        result.assert_eq("summary shape cols", summary["shape"][1], 15)
        result.assert_true(
            "summary memory_mb > 0", summary["memory_mb"] > 0, "memoire <= 0"
        )

    result.assert_no_exception("data_summary sur dataset large", t_summary_on_large_df)

    def t_column_stats_numeric():
        stats = get_column_stats(df, "Age")
        result.assert_eq("column_stats type numeric", stats["type"], "numeric")
        result.assert_true("column_stats skew defined", "skew" in stats, "skew absent")
        result.assert_true(
            "column_stats kurtosis defined", "kurtosis" in stats, "kurtosis absent"
        )

    result.assert_no_exception("column_stats numerique", t_column_stats_numeric)

    def t_column_stats_categorical():
        stats = get_column_stats(df, "Heart Disease")
        result.assert_eq("column_stats type categorical", stats["type"], "categorical")
        result.assert_true(
            "column_stats unique defined", "unique" in stats, "unique absent"
        )

    result.assert_no_exception("column_stats categoriel", t_column_stats_categorical)

    from utils.charts import (
        create_correlation_heatmap,
        create_outlier_plot,
        create_categorical_plot,
    )

    result.assert_no_exception(
        "correlation_heatmap",
        lambda: create_correlation_heatmap(df),
    )

    def t_outlier_plot():
        fig = create_outlier_plot(df, "Cholesterol")
        result.assert_type("outlier_plot retourne Figure", fig, object)

    result.assert_no_exception("outlier_plot", t_outlier_plot)

    result.assert_no_exception(
        "categorical_plot",
        lambda: create_categorical_plot(df, "Heart Disease"),
    )


if __name__ == "__main__":
    t0 = time.time()
    r = TestResult()
    test_phase1(r)
    elapsed = time.time() - t0
    print(f"\nDuree: {elapsed:.2f}s")
    success = r.summary()
    sys.exit(0 if success else 1)
