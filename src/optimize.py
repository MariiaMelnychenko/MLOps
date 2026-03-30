from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def git_revision(cwd: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            timeout=5,
        ).strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return "unknown"


def load_prepared_data(
    prepared_dir: str, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    root = Path(get_original_cwd())
    prep = root / prepared_dir
    train_df = pd.read_csv(prep / "train.csv")
    test_df = pd.read_csv(prep / "test.csv")
    if target_column not in train_df.columns:
        raise ValueError(f"Колонка '{target_column}' відсутня в train.csv")
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    return X_train, X_test, y_train, y_test


def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_type == "logistic_regression":
        clf = LogisticRegression(random_state=seed, max_iter=2000, **params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Невідомий model.type='{model_type}'")


def evaluate_metric(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    if metric == "f1":
        return float(
            f1_score(
                y_val,
                y_pred,
                average="binary" if len(np.unique(y_val)) == 2 else "weighted",
            )
        )
    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)
            if proba.shape[1] == 2:
                y_score = proba[:, 1]
            else:
                return float(
                    roc_auc_score(y_val, proba, multi_class="ovr", average="weighted")
                )
        else:
            y_score = model.decision_function(X_val)
        return float(roc_auc_score(y_val, y_score))
    raise ValueError("metric має бути 'f1' або 'roc_auc'")


def evaluate_cv(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str,
    seed: int,
    n_splits: int,
) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        m = clone(model)
        scores.append(evaluate_metric(m, X_tr, y_tr, X_va, y_va, metric))
    return float(np.mean(scores))


def make_sampler(
    sampler_name: str, seed: int, grid_space: Dict[str, list] | None
) -> optuna.samplers.BaseSampler:
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "grid":
        if not grid_space:
            raise ValueError("Для sampler=grid потрібен grid_space")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError("sampler: tpe | random | grid")


def suggest_params(
    trial: optuna.Trial, model_type: str, cfg: DictConfig
) -> Dict[str, Any]:
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                int(space.n_estimators.low),
                int(space.n_estimators.high),
            ),
            "max_depth": trial.suggest_int(
                "max_depth", int(space.max_depth.low), int(space.max_depth.high)
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                int(space.min_samples_split.low),
                int(space.min_samples_split.high),
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                int(space.min_samples_leaf.low),
                int(space.min_samples_leaf.high),
            ),
        }
    if model_type == "logistic_regression":
        space = cfg.hpo.logistic_regression
        return {
            "C": trial.suggest_float(
                "C", float(space.C.low), float(space.C.high), log=True
            ),
            "solver": trial.suggest_categorical("solver", list(space.solver)),
            "penalty": trial.suggest_categorical("penalty", list(space.penalty)),
        }
    raise ValueError(f"Невідомий model.type='{model_type}'")


def objective_factory(
    cfg: DictConfig,
    X_train_full: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_full: pd.Series,
    y_test: pd.Series,
):
    val_size = float(cfg.hpo.val_size)
    metric_name = str(cfg.hpo.metric)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", str(trial.number))
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", str(cfg.hpo.sampler))
            mlflow.set_tag("seed", str(cfg.seed))
            mlflow.log_params(
                {k: str(v) if v is not None else "null" for k, v in params.items()}
            )
            model = build_model(cfg.model.type, params=params, seed=int(cfg.seed))
            if cfg.hpo.use_cv:
                score = evaluate_cv(
                    model,
                    X_train_full,
                    y_train_full,
                    metric=metric_name,
                    seed=int(cfg.seed),
                    n_splits=int(cfg.hpo.cv_folds),
                )
            else:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size=val_size,
                    random_state=int(cfg.seed),
                    stratify=y_train_full,
                )
                score = evaluate_metric(model, X_tr, y_tr, X_val, y_val, metric_name)

            mlflow.log_metric(metric_name, score)
            return score

    return objective


def register_model_if_enabled(model_uri: str, model_name: str, stage: str) -> None:
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
        archive_existing_versions=False,
    )
    client.set_model_version_tag(model_name, mv.version, "registered_by", "lab3")
    client.set_model_version_tag(model_name, mv.version, "stage", stage)


def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    root = Path(get_original_cwd())
    os.chdir(root)

    uri = cfg.mlflow.tracking_uri
    if uri and str(uri).strip():
        mlflow.set_tracking_uri(str(uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))

    rev = git_revision(root)
    X_train_full, X_test, y_train_full, y_test = load_prepared_data(
        str(cfg.data.prepared_dir), str(cfg.data.target_column)
    )

    grid_space: Dict[str, list] | None = None
    if str(cfg.hpo.sampler).lower() == "grid":
        if cfg.model.type == "random_forest":
            g = cfg.hpo.grid.random_forest
            grid_space = {
                "n_estimators": list(g.n_estimators),
                "max_depth": list(g.max_depth),
                "min_samples_split": list(g.min_samples_split),
                "min_samples_leaf": list(g.min_samples_leaf),
            }
        elif cfg.model.type == "logistic_regression":
            g = cfg.hpo.grid.logistic_regression
            grid_space = {
                "C": list(g.C),
                "solver": list(g.solver),
                "penalty": list(g.penalty),
            }
        else:
            raise ValueError("Grid sampler не налаштований для цієї моделі")

    sampler = make_sampler(str(cfg.hpo.sampler), int(cfg.seed), grid_space)

    parent_name = f"hpo_parent_{cfg.hpo.sampler}_{cfg.model.type}"
    with mlflow.start_run(run_name=parent_name) as parent_run:
        mlflow.set_tag("model_type", str(cfg.model.type))
        mlflow.set_tag("sampler", str(cfg.hpo.sampler))
        mlflow.set_tag("seed", str(cfg.seed))
        mlflow.set_tag("git_commit", rev)
        mlflow.log_param("n_trials", int(cfg.hpo.n_trials))
        mlflow.log_param("use_cv", bool(cfg.hpo.use_cv))
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(cfg_dict, "config_resolved.json")

        study = optuna.create_study(
            direction=str(cfg.hpo.direction),
            sampler=sampler,
        )
        objective = objective_factory(cfg, X_train_full, X_test, y_train_full, y_test)
        study.optimize(
            objective, n_trials=int(cfg.hpo.n_trials), show_progress_bar=False
        )

        best_trial = study.best_trial
        metric_name = str(cfg.hpo.metric)
        mlflow.log_metric(f"best_{metric_name}", float(best_trial.value))
        mlflow.log_dict(dict(best_trial.params), "best_params.json")

        best_model = build_model(
            cfg.model.type, params=best_trial.params, seed=int(cfg.seed)
        )
        best_model.fit(X_train_full, y_train_full)
        y_test_pred = best_model.predict(X_test)
        final_f1 = float(
            f1_score(
                y_test,
                y_test_pred,
                average="binary" if len(np.unique(y_test)) == 2 else "weighted",
            )
        )
        mlflow.log_metric(f"final_{metric_name}_test", final_f1)
        mlflow.log_metric("final_accuracy_test", float((y_test_pred == y_test).mean()))

        models_dir = root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        best_path = models_dir / "best_model.pkl"
        joblib.dump(best_model, best_path)
        mlflow.log_artifact(str(best_path))

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, name="model")

        if bool(cfg.mlflow.register_model):
            try:
                model_uri = f"runs:/{parent_run.info.run_id}/model"
                register_model_if_enabled(
                    model_uri,
                    str(cfg.mlflow.model_name),
                    str(cfg.mlflow.stage),
                )
            except Exception as e:
                print(f"Model Registry недоступний (локальний store?): {e}")

    print(f"Найкращі параметри: {best_trial.params}")
    print(f"Найкраща метрика ({metric_name}) на val/CV: {best_trial.value:.4f}")
    print(f"Фінальна F1 на test: {final_f1:.4f}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
