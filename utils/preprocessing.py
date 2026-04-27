import numpy as np
import pandas as pd

from utils.data_loader import get_data_summary

TARGET_KEYWORDS = [
    ("target", 1.0),
    ("label", 1.0),
    ("survived", 0.95),
    ("churn", 0.95),
    ("disease", 0.95),
    ("fraud", 0.95),
    ("diagnosis", 0.95),
    ("outcome", 0.9),
    ("class", 0.85),
    ("price", 0.85),
    ("default", 0.85),
    ("response", 0.8),
    ("result", 0.7),
    ("category", 0.7),
    ("output", 0.65),
    ("prediction", 0.65),
    ("y", 0.5),
]

ID_KEYWORDS = ["id", "index", "uid", "uuid", "name", "identifier"]


def detect_target_column(df: pd.DataFrame) -> dict:
    candidates = []
    for col in df.columns:
        col_lower = col.lower().strip()
        best_score = 0
        for kw, priority in TARGET_KEYWORDS:
            if kw in col_lower:
                match_score = priority * (len(kw) / max(len(col_lower), 1))
                if match_score > best_score:
                    best_score = match_score
        if best_score > 0:
            candidates.append((col, round(best_score, 3), "keyword"))

    if not candidates:
        last_col = df.columns[-1]
        candidates.append((last_col, 0.3, "last_column"))

    candidates.sort(key=lambda x: -x[1])
    target_col = candidates[0][0]

    return {
        "target_column": target_col,
        "confidence": round(candidates[0][1], 2),
        "method": candidates[0][2],
        "all_candidates": [
            {"column": c, "confidence": round(s, 2), "method": m} for c, s, m in candidates[:5]
        ],
    }


def detect_task_type(df: pd.DataFrame, target_col: str) -> dict:
    if target_col not in df.columns:
        return {"task_type": "unknown", "error": f"Colonne '{target_col}' introuvable."}

    series = df[target_col].dropna()
    if series.empty:
        return {"task_type": "unknown", "error": "Colonne cible vide."}

    nunique = series.nunique()
    dtype = str(series.dtype)

    if pd.api.types.is_numeric_dtype(series):
        if nunique <= 2:
            task = "binary_classification"
            subtype = "numeric_binary"
        elif nunique <= 20 and nunique / len(series) < 0.05:
            task = "multiclass_classification"
            subtype = "low_cardinality_numeric"
        else:
            task = "regression"
            subtype = "continuous_numeric"
    elif pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        if nunique == 2:
            task = "binary_classification"
            subtype = "categorical_binary"
        elif nunique <= 20:
            task = "multiclass_classification"
            subtype = "categorical_multi"
        else:
            task = "multiclass_classification"
            subtype = "high_cardinality_categorical"
    else:
        task = "unknown"
        subtype = str(dtype)

    return {
        "task_type": task,
        "subtype": subtype,
        "nunique": int(nunique),
        "dtype": dtype,
        "is_balanced": None,
    }


def analyze_preprocessing_needs(df: pd.DataFrame, target_col: str = None) -> dict:
    needs = {
        "missing_values": [],
        "encoding": [],
        "scaling": [],
        "transformation": [],
        "low_variance": [],
        "multicollinearity": [],
        "id_columns": [],
        "warnings": [],
    }

    summary = get_data_summary(df)
    total_rows = len(df)

    for col in df.columns:
        col_lower = col.lower().strip()
        missing_pct = summary["missing_pct"].get(col, 0)

        if any(kw in col_lower for kw in ID_KEYWORDS) and col != target_col:
            if df[col].nunique() / total_rows > 0.8:
                needs["id_columns"].append(
                    {
                        "column": col,
                        "action": "supprimer (identifiant non informatif)",
                    }
                )
                continue

        if missing_pct > 0:
            action = "imputer"
            strategy = "median" if pd.api.types.is_numeric_dtype(df[col]) else "mode"
            if missing_pct > 50:
                action = "supprimer ou imputer avec flag"
                strategy = "flag + constante"
            needs["missing_values"].append(
                {
                    "column": col,
                    "missing_pct": round(missing_pct, 2),
                    "action": action,
                    "strategy": strategy,
                }
            )

        if col == target_col:
            continue

        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            nunique = df[col].nunique()
            if nunique <= 10:
                enc_type = "one-hot encoding"
            elif nunique <= 30:
                enc_type = "one-hot encoding (avec drop_first=True) ou label encoding"
            else:
                enc_type = "target encoding ou frequency encoding (haute cardinalite)"
            needs["encoding"].append(
                {
                    "column": col,
                    "nunique": int(nunique),
                    "method": enc_type,
                }
            )
        elif pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if len(series) > 0:
                variance = series.var()
                if variance < 1e-6:
                    needs["low_variance"].append(
                        {
                            "column": col,
                            "variance": float(variance),
                            "action": "supprimer (variance quasi-nulle)",
                        }
                    )

                if len(series) > 10:
                    skew = float(series.skew())
                    if abs(skew) > 1.5:
                        method = "log1p" if series.min() >= 0 else "Yeo-Johnson"
                        needs["transformation"].append(
                            {
                                "column": col,
                                "skew": round(skew, 2),
                                "method": method,
                                "reason": f"asymetrie elevee (skew={skew:.2f})",
                            }
                        )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]
    if len(feature_cols) >= 2:
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        for col in feature_cols:
            for other_col in feature_cols:
                if col < other_col and upper.loc[col, other_col] > 0.95:
                    high_corr_pairs.append(
                        {
                            "col1": col,
                            "col2": other_col,
                            "correlation": round(float(upper.loc[col, other_col]), 3),
                        }
                    )
        if high_corr_pairs:
            needs["multicollinearity"] = high_corr_pairs

    if len(feature_cols) >= 2:
        ranges = df[feature_cols].agg(["min", "max"])
        range_spread = ranges.loc["max"] - ranges.loc["min"]
        max_spread = range_spread.max()
        min_spread = range_spread[range_spread > 0].min()
        if max_spread / min_spread > 100:
            needs["scaling"].append(
                {
                    "action": "standardiser (StandardScaler ou RobustScaler)",
                    "reason": f"ecart d'echelle significatif entre features (ratio: {max_spread / min_spread:.0f}x)",
                }
            )

    if total_rows < 100:
        needs["warnings"].append(
            "Dataset tres petit (<100 lignes) : risque de surapprentissage eleve."
        )
    if total_rows > 100000:
        needs["warnings"].append(
            "Dataset volumineux (>100k lignes) : certains modeles peuvent etre lents a entrainer."
        )

    return needs


def suggest_feature_engineering(df: pd.DataFrame, target_col: str, task_type: str) -> list:
    suggestions = []

    datetime_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        try:
            parsed = pd.to_datetime(df[col], format="mixed")
            if parsed.notna().sum() / len(df) > 0.8:
                datetime_cols.append(col)
        except (ValueError, TypeError):
            pass

    for col in datetime_cols:
        suggestions.append(
            {
                "type": "temporal",
                "source": col,
                "features": ["year", "month", "day_of_week", "hour"],
                "reason": "Extraction de composantes temporelles",
            }
        )

    if task_type == "binary_classification" and target_col in df.columns:
        target_series = df[target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            pass

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    if len(numeric_cols) >= 2 and len(numeric_cols) <= 20:
        suggestions.append(
            {
                "type": "interactions",
                "source": f"top {min(5, len(numeric_cols))} features",
                "features": ["produits croises des features les plus correlnees"],
                "reason": "Interactions entre features potentiellement predictives",
            }
        )

    for col in numeric_cols[:5]:
        series = df[col].dropna()
        if len(series) > 10:
            q75 = series.quantile(0.75)
            q25 = series.quantile(0.25)
            if q75 > q25:
                suggestions.append(
                    {
                        "type": "binning",
                        "source": col,
                        "features": [f"{col}_binned (quartiles ou deciles)"],
                        "reason": "Binning pour capturer des relations non-lineaires",
                    }
                )
                break

    return suggestions
