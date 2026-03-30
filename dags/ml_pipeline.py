from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Repo root: in Docker use ML_PROJECT_ROOT=/opt/airflow/project; locally — parent of dags/
_PROJECT_DEFAULT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT", str(_PROJECT_DEFAULT))

# Threshold for model promotion (adjust for your baseline; lab example uses 0.5)
ACCURACY_THRESHOLD = float(os.environ.get("ML_ACCURACY_THRESHOLD", "0.5"))

METRICS_PATH = os.path.join(PROJECT_ROOT, "data", "models", "metrics.json")


def _evaluate_metrics(**context) -> dict:
    """Read metrics written by train.py; push dict to XCom for branching."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {
        "accuracy": float(data["accuracy"]),
        "f1": float(data["f1"]),
    }
    return out


def _branch_on_accuracy(**context) -> str:
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="evaluate")
    if metrics is None:
        return "stop_pipeline"
    acc = float(metrics["accuracy"])
    if acc > ACCURACY_THRESHOLD:
        return "register_model"
    return "stop_pipeline"


def _register_model_staging(**context) -> None:
    """Register latest MLflow model run to Registry (Staging) when backend allows; else log."""
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = pathlib.Path(PROJECT_ROOT) / "mlruns"
    mlflow.set_tracking_uri(tracking_uri.as_uri())
    client = MlflowClient()
    exp = client.get_experiment_by_name("Telco_Churn_Experiment")
    if exp is None:
        print("No experiment Telco_Churn_Experiment; skip registry.")
        return
    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
    if not runs:
        print("No runs; skip registry.")
        return
    runs.sort(key=lambda r: r.info.start_time, reverse=True)
    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/random_forest_model"
    try:
        result = mlflow.register_model(model_uri, "TelcoChurnModel")
        client.transition_model_version_stage(
            name="TelcoChurnModel",
            version=str(result.version),
            stage="Staging",
            archive_existing_versions=False,
        )
        print(f"Registered version {result.version} to Staging.")
    except Exception as exc:
        # File store often has no Model Registry; document without failing the DAG in dev
        print(f"Registry optional in this setup: {exc}")


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="DVC prepare/train, evaluate metrics.json, branch on accuracy",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["lab5"],
) as dag:
    # DVC_NO_ANALYTICS: avoid daemon + ValueError after successful repro in containers
    env_project = {
        "ML_PROJECT_ROOT": PROJECT_ROOT,
        "DVC_NO_ANALYTICS": "true",
        "CI": "true",
    }

    check_data = BashOperator(
        task_id="check_data",
        bash_command='set -euo pipefail && test -f "$ML_PROJECT_ROOT/data/raw/telco.csv"',
        env=env_project,
    )

    # Use `python -m dvc`: the `dvc` console script is often not on PATH in Airflow workers.
    prepare = BashOperator(
        task_id="prepare",
        bash_command=(
            'set -euo pipefail && cd "$ML_PROJECT_ROOT" && '
            'python -m dvc repro -s prepare'
        ),
        env=env_project,
    )

    train = BashOperator(
        task_id="train",
        bash_command=(
            'set -euo pipefail && cd "$ML_PROJECT_ROOT" && '
            'python -m dvc repro -s train'
        ),
        env=env_project,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate_metrics,
    )

    branch_on_accuracy = BranchPythonOperator(
        task_id="branch_on_accuracy",
        python_callable=_branch_on_accuracy,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model_staging,
    )

    stop_pipeline = BashOperator(
        task_id="stop_pipeline",
        bash_command='echo "Accuracy below threshold or missing metrics; pipeline stopped."',
    )

    check_data >> prepare >> train >> evaluate >> branch_on_accuracy
    branch_on_accuracy >> [register_model, stop_pipeline]
