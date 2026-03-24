"""Post-train тести: перевірка артефактів та Quality Gate."""

import json
import os

import pytest

MODELS_DIR = os.getenv("MODELS_DIR", "data/models")
F1_THRESHOLD = float(os.getenv("F1_THRESHOLD", "0.50"))


def test_model_pkl_exists():
    path = os.path.join(MODELS_DIR, "model.pkl")
    assert os.path.exists(path), f"model.pkl not found: {path}"
    assert os.path.getsize(path) > 0, "model.pkl is empty"


def test_metrics_json_exists():
    path = os.path.join(MODELS_DIR, "metrics.json")
    assert os.path.exists(path), f"metrics.json not found: {path}"


def test_confusion_matrix_png_exists():
    path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    assert os.path.exists(path), f"confusion_matrix.png not found: {path}"


def test_feature_importance_png_exists():
    path = os.path.join(MODELS_DIR, "feature_importance.png")
    assert os.path.exists(path), f"feature_importance.png not found: {path}"


def test_metrics_json_has_required_keys():
    path = os.path.join(MODELS_DIR, "metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    for key in ("accuracy", "f1"):
        assert key in metrics, f"Key '{key}' missing in metrics.json"
        assert isinstance(metrics[key], (int, float)), f"{key} is not a number"


def test_quality_gate_f1():
    path = os.path.join(MODELS_DIR, "metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    f1 = float(metrics["f1"])
    assert (
        f1 >= F1_THRESHOLD
    ), f"Quality Gate FAILED: f1={f1:.4f} < threshold={F1_THRESHOLD:.2f}"


def test_quality_gate_accuracy():
    path = os.path.join(MODELS_DIR, "metrics.json")
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    acc = float(metrics["accuracy"])
    assert acc >= 0.50, f"Quality Gate FAILED: accuracy={acc:.4f} < 0.50"
