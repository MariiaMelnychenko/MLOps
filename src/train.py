import argparse
import json
import os
import pathlib

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def load_prepared_data(prepared_dir):
    train_path = os.path.join(prepared_dir, "train.csv")
    test_path = os.path.join(prepared_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred, output_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(model, X, output_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(args):
    # Use portable absolute URI so mlflow works on both Windows and Linux
    tracking_uri = pathlib.Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Telco_Churn_Experiment")

    X_train, X_test, y_train, y_test = load_prepared_data(args.prepared_dir)
    os.makedirs(args.models_dir, exist_ok=True)

    with mlflow.start_run():
        mlflow.set_tag("author", "MariiaMelnychenko")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset_version", "v1")

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)

        # Save plots and core artifacts to models dir first
        cm_path = os.path.join(args.models_dir, "confusion_matrix.png")
        fi_path = os.path.join(args.models_dir, "feature_importance.png")
        metrics_path = os.path.join(args.models_dir, "metrics.json")
        model_path = os.path.join(args.models_dir, "model.pkl")

        plot_confusion_matrix(y_test, y_test_pred, cm_path)
        plot_feature_importance(model, X_train, fi_path)

        metrics = {
            "accuracy": float(test_acc),
            "f1": float(test_f1),
            "train_accuracy": float(train_acc),
            "train_f1": float(train_f1),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        joblib.dump(model, model_path)
        print(f"Saved: {model_path}, {metrics_path}, {cm_path}, {fi_path}")
        print(f"Metrics: accuracy={test_acc:.4f}, f1={test_f1:.4f}")

        # Log artifacts to MLflow (after files are saved)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(metrics_path)
        mlflow.sklearn.log_model(model, "random_forest_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prepared_dir",
        type=str,
        help="Path to prepared data directory (e.g. data/prepared)",
    )
    parser.add_argument(
        "models_dir",
        type=str,
        help="Path to output models directory (e.g. data/models)",
    )
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)

    args = parser.parse_args()
    main(args)
