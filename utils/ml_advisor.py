import numpy as np
import pandas as pd

from utils.preprocessing import (
    analyze_preprocessing_needs,
    detect_target_column,
    detect_task_type,
    suggest_feature_engineering,
)

CLASSIFICATION_MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression",
        "suitable_for": ["binary_classification", "multiclass_classification"],
        "complexity": "faible",
        "strengths": ["interpretable", "rapide", "baseline solide"],
        "weaknesses": ["relations non-lineaires", "features corrigeles"],
        "hyperparams": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "lbfgs"],
            "max_iter": [1000],
        },
        "min_samples": 5,
    },
    "decision_tree": {
        "name": "Decision Tree",
        "suitable_for": ["binary_classification", "multiclass_classification"],
        "complexity": "faible",
        "strengths": ["interpretable", "non-lineaire", "pas de scaling"],
        "weaknesses": ["surapprentissage", "instabilite"],
        "hyperparams": {
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
        "min_samples": 5,
    },
    "random_forest": {
        "name": "Random Forest",
        "suitable_for": ["binary_classification", "multiclass_classification"],
        "complexity": "moyen",
        "strengths": [
            "robuste",
            "gestion des outliers",
            "feature importance",
            "non-lineaire",
        ],
        "weaknesses": ["lent sur tres grands datasets", "peu interpretable"],
        "hyperparams": {
            "n_estimators": [100, 200, 500],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        },
        "min_samples": 200,
    },
    "gradient_boosting": {
        "name": "Gradient Boosting (XGBoost/LightGBM)",
        "suitable_for": ["binary_classification", "multiclass_classification"],
        "complexity": "eleve",
        "strengths": [
            "performances elevees",
            "feature importance",
            "gestion des types mixtes",
        ],
        "weaknesses": ["surapprentissage si mal configure", "lent a entrainer"],
        "hyperparams": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
        },
        "min_samples": 500,
    },
    "svc": {
        "name": "SVM (Support Vector Classifier)",
        "suitable_for": ["binary_classification", "multiclass_classification"],
        "complexity": "moyen",
        "strengths": ["efficace en haute dimension", "marge maximale"],
        "weaknesses": ["tres lent sur grands datasets", "scaling obligatoire"],
        "hyperparams": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
        "min_samples": 50,
        "max_samples": 50000,
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "suitable_for": ["binary_classification", "multiclass_classification"],
        "complexity": "faible",
        "strengths": ["simple", "non-parametrique"],
        "weaknesses": [
            "lent en prediction",
            "sensible au scaling",
            "sensible aux dimensions",
        ],
        "hyperparams": {
            "n_neighbors": [3, 5, 7, 11, 21],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "min_samples": 50,
        "max_samples": 20000,
    },
}

REGRESSION_MODELS = {
    "linear_regression": {
        "name": "Linear Regression",
        "suitable_for": ["regression"],
        "complexity": "faible",
        "strengths": ["interpretable", "rapide", "baseline solide"],
        "weaknesses": ["relations non-lineaires", "outliers"],
        "hyperparams": {
            "fit_intercept": [True, False],
        },
        "min_samples": 5,
    },
    "ridge": {
        "name": "Ridge Regression",
        "suitable_for": ["regression"],
        "complexity": "faible",
        "strengths": ["regularisation L2", "multicollinearite", "interpretable"],
        "weaknesses": ["relations non-lineaires"],
        "hyperparams": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "solver": ["auto", "lsqr", "sag"],
        },
        "min_samples": 5,
    },
    "lasso": {
        "name": "Lasso Regression",
        "suitable_for": ["regression"],
        "complexity": "faible",
        "strengths": ["selection de features (L1)", "interpretable", "sparse"],
        "weaknesses": ["instable si features correlees"],
        "hyperparams": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "selection": ["cyclic", "random"],
        },
        "min_samples": 5,
    },
    "decision_tree_reg": {
        "name": "Decision Tree Regressor",
        "suitable_for": ["regression"],
        "complexity": "faible",
        "strengths": ["non-lineaire", "interpretable"],
        "weaknesses": ["surapprentissage", "instabilite"],
        "hyperparams": {
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
        },
        "min_samples": 5,
    },
    "random_forest_reg": {
        "name": "Random Forest Regressor",
        "suitable_for": ["regression"],
        "complexity": "moyen",
        "strengths": ["robuste", "non-lineaire", "feature importance"],
        "weaknesses": ["lent sur grands datasets"],
        "hyperparams": {
            "n_estimators": [100, 200, 500],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5],
        },
        "min_samples": 200,
    },
    "gradient_boosting_reg": {
        "name": "Gradient Boosting Regressor",
        "suitable_for": ["regression"],
        "complexity": "eleve",
        "strengths": ["performances elevees", "flexible"],
        "weaknesses": ["surapprentissage", "lent"],
        "hyperparams": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 0.9, 1.0],
        },
        "min_samples": 500,
    },
    "svr": {
        "name": "SVR (Support Vector Regression)",
        "suitable_for": ["regression"],
        "complexity": "moyen",
        "strengths": ["efficace en haute dimension"],
        "weaknesses": ["tres lent sur grands datasets", "scaling obligatoire"],
        "hyperparams": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
            "epsilon": [0.01, 0.1, 0.5],
        },
        "min_samples": 50,
        "max_samples": 20000,
    },
}


def suggest_models(df: pd.DataFrame, target_col: str, task_type: str) -> list:
    n_samples = len(df)
    n_features = len([c for c in df.columns if c != target_col])
    numeric_features = len(df.select_dtypes(include=[np.number]).columns.tolist())
    cat_features = len(df.select_dtypes(include=["object", "category"]).columns.tolist())
    has_high_cardinality = any(
        df[c].nunique() > 30
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c != target_col
    )

    model_pool = CLASSIFICATION_MODELS if "classification" in task_type else REGRESSION_MODELS

    scored = []
    for key, model in model_pool.items():
        score = 50
        reasons = []

        if n_samples < model.get("min_samples", 0):
            continue

        max_samp = model.get("max_samples")
        if max_samp and n_samples > max_samp:
            score -= 20
            reasons.append(f"Dataset trop large pour {model['name']} (>{max_samp:,} lignes)")

        if n_samples < 1000:
            if model["complexity"] == "faible":
                score += 20
                reasons.append("Petit dataset -> modele simple privilegie")
            elif model["complexity"] == "eleve":
                score -= 10
                reasons.append("Modele complexe risque de surapprentissage")

        if n_samples >= 10000:
            if model["complexity"] == "eleve":
                score += 15
                reasons.append("Grand dataset -> modele complexe justifie")
            if model["name"] in (
                "SVM (Support Vector Classifier)",
                "SVR (Support Vector Regression)",
                "K-Nearest Neighbors",
            ):
                score -= 25
                reasons.append("Tres lent sur grands datasets")

        if n_features > n_samples:
            if key in ("ridge", "lasso", "logistic_regression"):
                score += 15
                reasons.append("Regularisation utile (features > samples)")
            if key in ("gradient_boosting", "gradient_boosting_reg"):
                score += 10
                reasons.append("Feature importance naturelle")

        if cat_features > 0:
            if has_high_cardinality:
                if "gradient_boosting" in key:
                    score += 15
                    reasons.append("Gradient Boosting gere bien la haute cardinalite")
                if "logistic" in key or "linear" in key or "ridge" in key or "lasso" in key:
                    score -= 5
                    reasons.append("Necessite encoding adequat pour variables categorielles")

        if numeric_features > n_features * 0.8:
            if key in ("svc", "svr"):
                score += 5
                reasons.append("Donnees numeriques -> SVM efficace")

        if "feature importance" in " ".join(model["strengths"]):
            score += 5
            reasons.append("Fournit l'importance des features")

        scored.append(
            {
                "model_key": key,
                "name": model["name"],
                "score": score,
                "complexity": model["complexity"],
                "strengths": model["strengths"],
                "weaknesses": model["weaknesses"],
                "reasons": reasons,
                "hyperparams": model["hyperparams"],
            }
        )

    scored.sort(key=lambda x: -x["score"])
    return scored


def suggest_evaluation_strategy(
    task_type: str, n_samples: int, target_series: pd.Series = None
) -> dict:
    strategy = {
        "task_type": task_type,
        "metrics": [],
        "validation": {},
        "warnings": [],
    }

    is_imbalanced = False
    if target_series is not None and "classification" in task_type:
        vc = target_series.value_counts()
        if len(vc) == 2:
            ratio = vc.max() / vc.min()
            if ratio > 3:
                is_imbalanced = True
                strategy["warnings"].append(
                    f"Classes desequilibrees (ratio {ratio:.1f}:1). "
                    f"Utiliser class_weight='balanced' ou SMOTE."
                )

    if "classification" in task_type:
        if task_type == "binary_classification":
            strategy["metrics"] = [
                {
                    "name": "ROC-AUC",
                    "description": "Capacite de discrimination globale",
                },
                {"name": "F1-Score", "description": "Equilibre precision/recall"},
                {
                    "name": "Precision",
                    "description": "Taux de vrais positifs parmi les predictions positives",
                },
                {"name": "Recall", "description": "Taux de positifs detectes"},
            ]
            if is_imbalanced:
                strategy["metrics"].insert(
                    0,
                    {
                        "name": "F1-Score (classe minoritaire)",
                        "description": "Metrique principale en cas de desequilibre",
                    },
                )
        else:
            strategy["metrics"] = [
                {
                    "name": "Accuracy",
                    "description": "Taux de bonnes predictions global",
                },
                {"name": "Macro F1-Score", "description": "Moyenne des F1 par classe"},
                {
                    "name": "Confusion Matrix",
                    "description": "Repartition des predictions par classe",
                },
            ]
    elif task_type == "regression":
        strategy["metrics"] = [
            {
                "name": "RMSE",
                "description": "Racine de l'erreur quadratique moyenne (sensible aux outliers)",
            },
            {
                "name": "MAE",
                "description": "Erreur absolue moyenne (robuste aux outliers)",
            },
            {"name": "R2", "description": "Proportion de variance expliquee"},
        ]

    if n_samples < 1000:
        n_folds = 5
    elif n_samples < 10000:
        n_folds = 5
    else:
        n_folds = 5

    strategy["validation"] = {
        "method": "Stratified K-Fold" if "classification" in task_type else "K-Fold",
        "n_folds": n_folds,
        "shuffle": True,
        "random_state": 42,
    }

    if n_samples < 100:
        strategy["validation"]["method"] = "Leave-One-Out"
        strategy["warnings"].append("Dataset tres petit : Leave-One-Out recommande.")
    elif n_samples > 100000:
        strategy["validation"]["method"] = "Train/Validation/Test split (80/10/10)"
        strategy["warnings"].append("Grand dataset : un simple split peut suffire.")

    return strategy


def generate_ml_report(df: pd.DataFrame, target_col: str = None) -> dict:
    target_info = detect_target_column(df)
    if target_col is None:
        target_col = target_info["target_column"]

    task_info = detect_task_type(df, target_col)
    task_type = task_info["task_type"]

    if task_type == "unknown":
        return {
            "success": False,
            "error": f"Impossible de determiner le type de tache pour la cible '{target_col}'",
        }

    needs = analyze_preprocessing_needs(df, target_col)
    models = suggest_models(df, target_col, task_type)
    eval_strategy = suggest_evaluation_strategy(task_type, len(df), df[target_col].dropna())
    feat_suggestions = suggest_feature_engineering(df, target_col, task_type)

    top_models = models[:3] if models else []

    report = {
        "success": True,
        "target": target_info,
        "task_type": task_info,
        "preprocessing": needs,
        "recommended_models": top_models,
        "evaluation_strategy": eval_strategy,
        "feature_engineering": feat_suggestions,
        "summary": _build_summary(df, target_col, task_info, top_models, needs, eval_strategy),
    }

    return report


def _build_summary(df, target_col, task_info, top_models, needs, eval_strategy) -> str:
    lines = []
    lines.append(f"## Rapport ML - Dataset ({len(df):,} lignes x {len(df.columns)} colonnes)")
    lines.append("")
    lines.append(f"**Cible detectee** : `{target_col}` ({task_info['dtype']})")
    lines.append(f"**Type de tache** : {task_info['task_type']} ({task_info['subtype']})")
    lines.append(f"**Nombre de classes** : {task_info['nunique']}")
    lines.append("")

    if top_models:
        lines.append("### Modeles recommandes")
        for i, m in enumerate(top_models, 1):
            lines.append(
                f"  {i}. **{m['name']}** (score: {m['score']}/100, complexite: {m['complexity']})"
            )
            for r in m.get("reasons", []):
                lines.append(f"     - {r}")
        lines.append("")

    pp_steps = []
    if needs["id_columns"]:
        pp_steps.append(f"Supprimer {len(needs['id_columns'])} colonne(s) ID")
    if needs["missing_values"]:
        pp_steps.append(
            f"Imputer {len(needs['missing_values'])} colonne(s) avec valeurs manquantes"
        )
    if needs["encoding"]:
        pp_steps.append(f"Encoder {len(needs['encoding'])} colonne(s) categorielle(s)")
    if needs["scaling"]:
        pp_steps.append("Standardiser les features numeriques")
    if needs["transformation"]:
        pp_steps.append(f"Transformer {len(needs['transformation'])} colonne(s) asymetriques")
    if needs["multicollinearity"]:
        pp_steps.append(f"Traiter {len(needs['multicollinearity'])} paire(s) de multicollinearite")

    if pp_steps:
        lines.append("### Preprocessing necessaire")
        for step in pp_steps:
            lines.append(f"  - {step}")
        lines.append("")

    lines.append("### Strategie d'evaluation")
    lines.append(f"  - **Methode** : {eval_strategy['validation']['method']}")
    lines.append(f"  - **Metriques** : {', '.join(m['name'] for m in eval_strategy['metrics'])}")
    if eval_strategy.get("warnings"):
        for w in eval_strategy["warnings"]:
            lines.append(f"  - **Attention** : {w}")

    return "\n".join(lines)
