"""Tracking MLflow optionnel.

Datamind ne reimplémente pas de tracker : si MLflow est installé
(extra `datamind-ai[mlflow]`), chaque entrainement logge un run
(params, metriques CV, pipeline, rapport). Sinon, le tracking est
silencieusement desactive — le reste de l'application est inchange.

Kill-switch : DATAMIND_TRACKING=off
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

TRACKING_ENV_VAR = "DATAMIND_TRACKING"
DEFAULT_EXPERIMENT = "datamind-ai"


def tracking_enabled() -> bool:
    """MLflow installe ET pas desactive par environnement."""
    if os.getenv(TRACKING_ENV_VAR, "").lower() in ("off", "0", "false"):
        return False
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


def log_training(result: dict, pipeline=None, report_text: str = "") -> dict:
    """Logge un resultat d'entrainement dans MLflow. Jamais d'exception
    remontee : le tracking est un bonus, pas un point de panne."""
    if not result.get("success"):
        return {"tracked": False, "reason": "training_failed"}
    if not tracking_enabled():
        return {"tracked": False, "reason": "mlflow_unavailable"}

    try:
        import mlflow

        # Backend SQLite : le filestore ./mlruns est en maintenance mode depuis
        # MLflow 3.x (leve une exception sans opt-in explicite).
        tracking_uri = os.getenv("DATAMIND_MLFLOW_URI", "sqlite:///./data/mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("DATAMIND_MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT))

        run_name = f"{result['task_type']}__{result['model_name'][:40]}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "task_type": result["task_type"],
                    "model": result["model_name"],
                    "target_column": result["target_column"],
                    "n_samples": result["n_samples"],
                    "n_features": result["n_features"],
                    "n_folds": result["n_folds"],
                }
            )
            for m in result.get("metrics", []):
                mlflow.log_metric(f"{m['name']}_mean", m["mean"])
                mlflow.log_metric(f"{m['name']}_std", m["std"])

            if report_text:
                mlflow.log_text(report_text, "report.txt")
            if pipeline is not None:
                # numpy.dtype : metadonnees benignes des dtypes pandas du
                # ColumnTransformer (serialiseur skops de MLflow 3.x).
                mlflow.sklearn.log_model(
                    pipeline, name="model", skops_trusted_types=["numpy.dtype"]
                )

            run_id = mlflow.active_run().info.run_id

        logger.info("Run MLflow logge : %s", run_id)
        return {"tracked": True, "run_id": run_id, "tracking_uri": tracking_uri}
    except Exception:
        logger.exception("Tracking MLflow en echec (non bloquant)")
        return {"tracked": False, "reason": "tracking_error"}
