"""Entrainement de baselines scikit-learn, mesurees par validation croisee.

Version bornee volontairement : UNE baseline par type de tache, pas de
grid search, pas de persistance. Le but est un chiffre honnete et
reproductible, pas un champion Kaggle.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datamind.analysis.dtypes import is_categorical_dtype
from datamind.analysis.preprocessing import (
    ID_KEYWORDS,
    detect_target_column,
    detect_task_type,
)

RANDOM_STATE = 42
MIN_SAMPLES = 10
IMBALANCE_RATIO = 3.0

_MODEL_NAMES = {
    "binary_classification": "Logistic Regression (class_weight selon equilibre)",
    "multiclass_classification": "Logistic Regression (multinomial)",
    "regression": "Ridge (alpha=1.0)",
}


def train_baseline(df: pd.DataFrame, target_column: str | None = None) -> dict:
    """Entraine une baseline et retourne des metriques mesurees en CV.

    Returns:
        dict avec success=True + metriques, ou success=False + error.
    """
    if df is None or len(df) == 0:
        return {"success": False, "error": "Aucune donnee chargee."}

    if target_column is None:
        target_column = detect_target_column(df)["target_column"]
    if target_column not in df.columns:
        return {
            "success": False,
            "error": f"Colonne cible '{target_column}' introuvable.",
        }

    task_info = detect_task_type(df, target_column)
    task_type = task_info["task_type"]
    if task_type == "unknown":
        return {
            "success": False,
            "error": task_info.get("error", "Type de tache indeterminable."),
        }

    # Lignes sans cible : inutilisables pour l'entrainement
    work = df.dropna(subset=[target_column]).reset_index(drop=True)
    if len(work) < MIN_SAMPLES:
        return {
            "success": False,
            "error": (
                f"Dataset trop petit apres suppression des cibles manquantes "
                f"({len(work)} lignes < {MIN_SAMPLES})."
            ),
        }

    y_raw = work[target_column]
    warnings: list[str] = []

    is_classification = "classification" in task_type
    if is_classification:
        classes = y_raw.unique()
        if len(classes) < 2:
            return {
                "success": False,
                "error": f"La cible '{target_column}' n'a qu'une seule classe.",
            }
        y = pd.factorize(y_raw)[0]
        class_counts = y_raw.value_counts()
        imbalance_ratio = class_counts.max() / max(class_counts.min(), 1)
        is_imbalanced = imbalance_ratio > IMBALANCE_RATIO
        if is_imbalanced:
            warnings.append(
                f"Classes desequilibrees (ratio {imbalance_ratio:.1f}:1). "
                "class_weight='balanced' applique ; considerez F1 plutot que l'accuracy."
            )
    else:
        y = y_raw.astype(float)
        imbalance_ratio = None
        is_imbalanced = False

    # Colonnes ID non informatives (memes heuristiques que l'advisor)
    total_rows = len(work)
    id_cols = [
        c
        for c in work.columns
        if c != target_column
        and any(kw in c.lower().strip() for kw in ID_KEYWORDS)
        and work[c].nunique() / total_rows > 0.8
    ]
    if id_cols:
        warnings.append(f"Colonnes identifiants exclues : {', '.join(id_cols)}")

    features = work.drop(columns=[target_column] + id_cols)
    if features.shape[1] == 0:
        return {"success": False, "error": "Aucune feature utilisable apres nettoyage."}

    cat_cols = [c for c in features.columns if is_categorical_dtype(features[c].dtype)]
    num_cols = [c for c in features.columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if is_classification:
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced" if is_imbalanced else None,
        )
    else:
        model = Ridge(alpha=1.0)

    pipeline = Pipeline([("prep", preprocessor), ("model", model)])

    # Validation croisee : folds bornes par la classe minoritaire
    if is_classification:
        min_class_count = int(pd.Series(y).value_counts().min())
        n_splits = max(2, min(5, min_class_count))
        cv: KFold | StratifiedKFold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
        )
        scoring = (
            ["f1", "roc_auc"] if task_type == "binary_classification" else ["f1_macro", "accuracy"]
        )
    else:
        n_splits = 5
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

    if len(work) < 100:
        warnings.append(
            f"Dataset tres petit ({len(work)} lignes) : les metriques sont tres instables."
        )

    scores = cross_validate(pipeline, features, y, cv=cv, scoring=scoring)

    metric_names = {
        "f1": "F1",
        "roc_auc": "ROC-AUC",
        "f1_macro": "F1 macro",
        "accuracy": "Accuracy",
        "neg_root_mean_squared_error": "RMSE",
        "neg_mean_absolute_error": "MAE",
        "r2": "R2",
    }
    metrics = []
    for s in scoring:
        values = scores[f"test_{s}"]
        mean = float(values.mean())
        if s.startswith("neg_"):
            mean = -mean
        metrics.append(
            {
                "name": metric_names[s],
                "mean": round(mean, 4),
                "std": round(float(values.std()), 4),
            }
        )

    # Refit sur tout le dataset : c'est cet artefact qui est logge/exporte
    fitted_pipeline = pipeline.fit(features, y)

    return {
        "success": True,
        "pipeline": fitted_pipeline,
        "task_type": task_type,
        "model_name": _MODEL_NAMES.get(task_type, "baseline"),
        "target_column": target_column,
        "n_samples": len(work),
        "n_features": int(features.shape[1]),
        "n_folds": n_splits,
        "metrics": metrics,
        "warnings": warnings,
    }


def format_training_result(result: dict) -> str:
    """Resume lisible du resultat d'entrainement (pour l'agent et l'API)."""
    if not result.get("success"):
        return f"Echec de l'entrainement : {result.get('error', 'erreur inconnue')}"

    lines = [
        f"Baseline entrainee : {result['model_name']}",
        f"Cible : {result['target_column']} ({result['task_type']})",
        f"Donnees : {result['n_samples']} echantillons, {result['n_features']} features, "
        f"{result['n_folds']}-fold CV",
        "",
        "Metriques (moyenne ± ecart-type) :",
    ]
    for m in result["metrics"]:
        lines.append(f"  - {m['name']} : {m['mean']:.4f} ± {m['std']:.4f}")
    for w in result.get("warnings", []):
        lines.append(f"Attention : {w}")
    return "\n".join(lines)
