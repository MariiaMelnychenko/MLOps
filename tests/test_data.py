"""Pre-train тести: перевірка якості та структури вхідних даних."""

import os

import pandas as pd
import pytest

DATA_DIR = os.getenv("DATA_DIR", "data/prepared")


@pytest.fixture
def train_df():
    path = os.path.join(DATA_DIR, "train.csv")
    assert os.path.exists(path), f"train.csv not found: {path}"
    return pd.read_csv(path)


@pytest.fixture
def test_df():
    path = os.path.join(DATA_DIR, "test.csv")
    assert os.path.exists(path), f"test.csv not found: {path}"
    return pd.read_csv(path)


def test_train_csv_exists():
    assert os.path.exists(os.path.join(DATA_DIR, "train.csv"))


def test_test_csv_exists():
    assert os.path.exists(os.path.join(DATA_DIR, "test.csv"))


def test_target_column_present(train_df, test_df):
    assert "Churn" in train_df.columns, "Churn column missing in train"
    assert "Churn" in test_df.columns, "Churn column missing in test"


def test_no_nulls_in_target(train_df, test_df):
    assert train_df["Churn"].notna().all(), "NaN in train Churn"
    assert test_df["Churn"].notna().all(), "NaN in test Churn"


def test_minimum_rows(train_df, test_df):
    assert train_df.shape[0] >= 50, f"Too few train rows: {train_df.shape[0]}"
    assert test_df.shape[0] >= 10, f"Too few test rows: {test_df.shape[0]}"


def test_no_duplicate_columns(train_df):
    assert len(train_df.columns) == len(set(train_df.columns)), "Duplicate columns"


def test_target_is_binary(train_df):
    unique = train_df["Churn"].nunique()
    assert unique == 2, f"Churn should have 2 classes, got {unique}"
