"""DAG integrity: Airflow DagBag must load without import_errors (Lab 5 CI)."""

import os


def test_dag_bag_no_import_errors():
    os.environ.setdefault("AIRFLOW_HOME", "/tmp/airflow_ci_test")
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder="dags", include_examples=False)
    assert len(dag_bag.import_errors) == 0, dag_bag.import_errors
