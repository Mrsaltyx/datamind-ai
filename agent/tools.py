import json

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from utils.charts import (
    create_categorical_plot,
    create_correlation_heatmap,
    create_distribution_plot,
    create_group_comparison,
    create_outlier_plot,
    create_scatter_plot,
    create_trend_plot,
)
from utils.data_loader import get_column_stats, get_data_summary, get_sample_data

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "describe_data",
            "description": "Obtenir une description statistique complete du jeu de donnees ou d'une colonne specifique. Inclut dimensions, types, valeurs manquantes, statistiques descriptives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Optional specific column name to describe. If omitted, describes the whole dataset.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_distribution",
            "description": "Generer un graphique de distribution (histogramme + boite a moustaches) pour une colonne numerique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The numeric column to plot.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_correlation",
            "description": "Generer une carte de correlations pour toutes les colonnes numeriques du jeu de donnees.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_outliers",
            "description": "Detecter et visualiser les valeurs aberrantes dans une colonne numerique avec la methode IQR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The numeric column to analyze for outliers.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_trends",
            "description": "Tracer une tendance de serie temporelle pour une colonne numerique sur une colonne de date/heure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {
                        "type": "string",
                        "description": "The date/time column.",
                    },
                    "value_column": {
                        "type": "string",
                        "description": "The numeric column to plot over time.",
                    },
                },
                "required": ["date_column", "value_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_groups",
            "description": "Comparer une variable numerique entre differentes categories. Affiche un diagramme en barres des moyennes et des boites a moustaches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categorical_column": {
                        "type": "string",
                        "description": "The categorical column to group by.",
                    },
                    "numeric_column": {
                        "type": "string",
                        "description": "The numeric column to compare.",
                    },
                },
                "required": ["categorical_column", "numeric_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_categorical",
            "description": "Afficher la distribution d'une colonne categorielle (diagramme en barres + camembert).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The categorical column to visualize.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_scatter",
            "description": "Creer un nuage de points entre deux colonnes numeriques avec une droite de tendance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_column": {
                        "type": "string",
                        "description": "Column for X axis.",
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column for Y axis.",
                    },
                    "color_column": {
                        "type": "string",
                        "description": "Optional column to color points by.",
                    },
                },
                "required": ["x_column", "y_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_target_and_task",
            "description": "Detecter automatiquement la colonne cible du dataset et determiner le type de tache ML (classification ou regression). Analyse aussi les besoins de preprocessing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "Optional explicit target column. If omitted, auto-detects the target.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_ml_pipeline",
            "description": "Generer un rapport ML complet : detection de la cible, type de tache, preprocessing, modeles recommandes avec hyperparametres, et strategie d'evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "Optional explicit target column. If omitted, auto-detects.",
                    }
                },
                "required": [],
            },
        },
    },
]


def _validate_column(df: pd.DataFrame, col: str, expected_type: str = None) -> str | None:
    if col not in df.columns:
        available = df.columns.tolist()
        close = [c for c in available if col.lower() in c.lower()]
        suggestion = (
            f" Colonnes similaires : {close}"
            if close
            else f" Colonnes disponibles : {available[:10]}"
        )
        return f"Colonne '{col}' introuvable.{suggestion}"
    if expected_type == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
        return f"Colonne '{col}' n'est pas numerique (type: {df[col].dtype})."
    if expected_type == "categorical" and pd.api.types.is_numeric_dtype(df[col]):
        if df[col].nunique() > 20:
            return f"Colonne '{col}' semble numerique avec {df[col].nunique()} valeurs uniques."
    return None


def _safe_describe(series: pd.Series) -> dict:
    if series.empty:
        return {"error": "Serie vide"}
    desc = series.describe()
    result = {}
    for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        if k in desc.index:
            result[k] = float(desc[k])
    return result


def execute_tool(tool_name: str, arguments: dict, df: pd.DataFrame) -> dict:
    result = {"success": False, "text": "", "figure": None}

    try:
        if tool_name == "describe_data":
            col = arguments.get("column")
            if col:
                err = _validate_column(df, col)
                if err:
                    result["text"] = err
                    return result
                stats = get_column_stats(df, col)
                result["text"] = json.dumps(stats, indent=2, default=str)
            else:
                summary = get_data_summary(df)
                result["text"] = json.dumps(summary, indent=2, default=str)
                result["text"] += f"\n\n--- Sample Data ---\n{get_sample_data(df)}"
            result["success"] = True

        elif tool_name == "show_distribution":
            col = arguments.get("column")
            if not col:
                result["text"] = "Parametre 'column' manquant."
                return result
            err = _validate_column(df, col, "numeric")
            if err:
                result["text"] = err
                return result
            series = df[col].dropna()
            if series.empty:
                result["text"] = f"Colonne '{col}' est entierement vide."
                return result
            result["figure"] = create_distribution_plot(df, col)
            desc = _safe_describe(series)
            skew_val = float(series.skew()) if len(series) > 2 else 0.0
            result["text"] = (
                f"Distribution de '{col}':\n"
                f"  Moyenne: {desc.get('mean', 0):.2f} | Mediane: {desc.get('50%', 0):.2f}\n"
                f"  Std: {desc.get('std', 0):.2f} | Skew: {skew_val:.2f}\n"
                f"  Min: {desc.get('min', 0):.2f} | Max: {desc.get('max', 0):.2f}"
            )
            result["success"] = True

        elif tool_name == "show_correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                result["text"] = "Pas assez de colonnes numeriques pour une matrice de correlation."
                return result
            fig = create_correlation_heatmap(df)
            if fig:
                corr = numeric_df.corr()
                top_corr = corr.unstack().sort_values(ascending=False)
                top_corr = top_corr[top_corr < 1.0].head(5)
                insights = "\n".join(
                    [f"  {idx[0]} <-> {idx[1]}: {val:.3f}" for idx, val in top_corr.items()]
                )
                result["text"] = f"Top correlations:\n{insights}"
                result["figure"] = fig
                result["success"] = True
            else:
                result["text"] = "Pas assez de colonnes numeriques pour une matrice de correlation."

        elif tool_name == "detect_outliers":
            col = arguments.get("column")
            if not col:
                result["text"] = "Parametre 'column' manquant."
                return result
            err = _validate_column(df, col, "numeric")
            if err:
                result["text"] = err
                return result
            series = df[col].dropna()
            if series.empty:
                result["text"] = f"Colonne '{col}' est entierement vide."
                return result
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                result["text"] = (
                    f"Colonne '{col}' a un IQR de 0 (variance quasi-nulle). Pas d'outliers detectables."
                )
                result["success"] = True
                return result
            n_outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
            pct = n_outliers / len(series) * 100
            result["figure"] = create_outlier_plot(df, col)
            result["text"] = (
                f"Outliers dans '{col}': {n_outliers} detectes ({pct:.1f}%)\n"
                f"  Bornes IQR: [{q1 - 1.5 * iqr:.2f}, {q3 + 1.5 * iqr:.2f}]"
            )
            result["success"] = True

        elif tool_name == "show_trends":
            date_col = arguments.get("date_column")
            val_col = arguments.get("value_column")
            if not date_col or not val_col:
                result["text"] = "Parametres 'date_column' et 'value_column' requis."
                return result
            err_date = _validate_column(df, date_col)
            err_val = _validate_column(df, val_col, "numeric")
            if err_date:
                result["text"] = err_date
                return result
            if err_val:
                result["text"] = err_val
                return result
            trend = df[[date_col, val_col]].dropna().sort_values(date_col)
            if trend.empty:
                result["text"] = f"Pas de donnees valides pour {date_col} / {val_col}."
                return result
            result["figure"] = create_trend_plot(df, date_col, val_col)
            result["text"] = (
                f"Tendance de '{val_col}' dans le temps:\n"
                f"  Periode: {trend[date_col].iloc[0]} -> {trend[date_col].iloc[-1]}\n"
                f"  Points: {len(trend)}"
            )
            result["success"] = True

        elif tool_name == "compare_groups":
            cat_col = arguments.get("categorical_column")
            num_col = arguments.get("numeric_column")
            if not cat_col or not num_col:
                result["text"] = "Parametres 'categorical_column' et 'numeric_column' requis."
                return result
            err_cat = _validate_column(df, cat_col)
            err_num = _validate_column(df, num_col, "numeric")
            if err_cat:
                result["text"] = err_cat
                return result
            if err_num:
                result["text"] = err_num
                return result
            if df[cat_col].nunique() > 50:
                result["text"] = (
                    f"Colonne '{cat_col}' a {df[cat_col].nunique()} valeurs uniques (>50). Trop de groupes pour une comparaison pertinente."
                )
                return result
            result["figure"] = create_group_comparison(df, cat_col, num_col)
            grp = (
                df.groupby(cat_col)[num_col]
                .agg(["mean", "median", "count"])
                .sort_values("mean", ascending=False)
            )
            result["text"] = (
                f"Comparaison de '{num_col}' par '{cat_col}':\n{grp.head(10).to_string()}"
            )
            result["success"] = True

        elif tool_name == "show_categorical":
            col = arguments.get("column")
            if not col:
                result["text"] = "Parametre 'column' manquant."
                return result
            err = _validate_column(df, col)
            if err:
                result["text"] = err
                return result
            if df[col].nunique() > 100:
                result["text"] = (
                    f"Colonne '{col}' a {df[col].nunique()} valeurs uniques (>100). Visualisation non pertinente."
                )
                return result
            result["figure"] = create_categorical_plot(df, col)
            vc = df[col].value_counts().head(10)
            result["text"] = f"Top valeurs de '{col}':\n{vc.to_string()}"
            result["success"] = True

        elif tool_name == "show_scatter":
            x_col = arguments.get("x_column")
            y_col = arguments.get("y_column")
            if not x_col or not y_col:
                result["text"] = "Parametres 'x_column' et 'y_column' requis."
                return result
            err_x = _validate_column(df, x_col, "numeric")
            err_y = _validate_column(df, y_col, "numeric")
            if err_x:
                result["text"] = err_x
                return result
            if err_y:
                result["text"] = err_y
                return result
            color_col = arguments.get("color_column")
            if color_col:
                err_c = _validate_column(df, color_col)
                if err_c:
                    color_col = None
            result["figure"] = create_scatter_plot(df, x_col, y_col, color_col)
            valid = df[[x_col, y_col]].dropna()
            if len(valid) < 2:
                result["text"] = f"Pas assez de donnees valides pour scatter {x_col} vs {y_col}."
                result["success"] = True
                return result
            r, p = sp_stats.pearsonr(valid[x_col], valid[y_col])
            result["text"] = (
                f"Scatter: {x_col} vs {y_col}\n  Correlation Pearson: r={r:.3f} (p={p:.4f})"
            )
            result["success"] = True

        elif tool_name == "detect_target_and_task":
            from utils.preprocessing import (
                analyze_preprocessing_needs,
                detect_target_column,
                detect_task_type,
            )

            target_col = arguments.get("target_column")
            if not target_col:
                target_info = detect_target_column(df)
                target_col = target_info["target_column"]
            else:
                err = _validate_column(df, target_col)
                if err:
                    result["text"] = err
                    return result
                target_info = {
                    "target_column": target_col,
                    "confidence": 1.0,
                    "method": "user_specified",
                }

            task_info = detect_task_type(df, target_col)
            needs = analyze_preprocessing_needs(df, target_col)

            report = {
                "target": target_info,
                "task_type": task_info,
                "preprocessing_needs": {k: v for k, v in needs.items() if v and k != "warnings"},
            }
            if needs.get("warnings"):
                report["warnings"] = needs["warnings"]

            result["text"] = json.dumps(report, indent=2, default=str)
            result["success"] = True

        elif tool_name == "suggest_ml_pipeline":
            from utils.ml_advisor import generate_ml_report

            target_col = arguments.get("target_column")
            report = generate_ml_report(df, target_col)

            if not report.get("success"):
                result["text"] = report.get("error", "Erreur lors de la generation du rapport ML.")
                return result

            summary = report.get("summary", "")
            top_models_info = []
            for m in report.get("recommended_models", [])[:3]:
                top_models_info.append(
                    f"  {m['name']} (score: {m['score']}/100, complexite: {m['complexity']})"
                )
            models_text = "\n".join(top_models_info)

            eval_strat = report.get("evaluation_strategy", {})
            metrics_text = ", ".join(m["name"] for m in eval_strat.get("metrics", []))
            val_method = eval_strat.get("validation", {}).get("method", "N/A")

            result["text"] = (
                f"{summary}\n\n"
                f"### Detail des modeles\n{models_text}\n\n"
                f"### Hyperparametres recommandes (modele #1)\n"
                f"{json.dumps(report['recommended_models'][0]['hyperparams'], indent=2)}\n\n"
                f"### Validation\n"
                f"Methode: {val_method}\n"
                f"Metriques: {metrics_text}"
            )
            result["success"] = True

        else:
            result["text"] = f"Outil inconnu: {tool_name}"

    except Exception as e:
        result["text"] = f"Erreur lors de l'execution de {tool_name}: {str(e)}"

    return result
