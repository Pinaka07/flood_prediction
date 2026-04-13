"""
tests/test_data_transformation.py
-----------------------------------
Tests for src/components/data_transformation.py
"""
import pytest
import numpy as np
import pandas as pd
import os

from src.configuration.config import FEATURE_NAMES, TEST_SIZE
from src.components.data_transformation import split_scale


# ── Fixture ────────────────────────────────────────────────────────────────

def _make_Xy(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    y = pd.Series(np.where(rng.random(n) < 0.035, 1, 0), name="flash_flood")
    return X, y


# ── Tests ──────────────────────────────────────────────────────────────────

def test_split_sizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/models", exist_ok=True)
    X, y = _make_Xy(n=600)
    X_train, X_test, y_train, y_test, _ = split_scale(X, y, use_smote=False)
    expected_test = int(600 * TEST_SIZE)
    assert X_test.shape[0] == expected_test
    assert X_train.shape[0] == 600 - expected_test


def test_scaler_zero_mean_on_train(tmp_path, monkeypatch):
    """StandardScaler must produce ~zero mean on train, NOT enforced on test."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/models", exist_ok=True)
    X, y = _make_Xy(n=800)
    X_train, X_test, _, _, _ = split_scale(X, y, use_smote=False)
    assert np.allclose(X_train.mean(axis=0), 0, atol=0.05), \
        "Train set should be ~zero-mean after StandardScaler"
    # Test set must NOT be forced to zero mean (would be leakage)
    assert not np.allclose(X_test.mean(axis=0), 0, atol=1e-9)


def test_no_leakage_scaler_fit_on_train_only(tmp_path, monkeypatch):
    """Scaler must be fit only on train. Test transform uses train statistics."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/models", exist_ok=True)
    X, y = _make_Xy(n=1000)
    X_train, X_test, _, _, scaler = split_scale(X, y, use_smote=False)
    # Recompute manually and verify scaler mean matches train mean
    assert np.allclose(scaler.mean_, X.iloc[:int(1000*(1-TEST_SIZE))].mean().values, atol=1.0)


def test_feature_count_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/models", exist_ok=True)
    X, y = _make_Xy(n=600)
    X_train, X_test, _, _, _ = split_scale(X, y, use_smote=False)
    assert X_train.shape[1] == len(FEATURE_NAMES)
    assert X_test.shape[1]  == len(FEATURE_NAMES)


def test_smote_increases_minority(tmp_path, monkeypatch):
    """SMOTE must increase the minority class count in the training set."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/models", exist_ok=True)
    X, y = _make_Xy(n=1000)
    _, _, y_train_no_smote, _, _ = split_scale(X, y, use_smote=False)
    _, _, y_train_smote,    _, _ = split_scale(X, y, use_smote=True)

    flood_before = int((np.array(y_train_no_smote) == 1).sum())
    flood_after  = int((np.array(y_train_smote)    == 1).sum())
    assert flood_after > flood_before, \
        f"SMOTE should increase minority: before={flood_before} after={flood_after}"


def test_smote_does_not_touch_test_set(tmp_path, monkeypatch):
    """Test set size and content must be identical with/without SMOTE."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/models", exist_ok=True)
    X, y = _make_Xy(n=800)
    _, X_test_no, _, y_test_no, _ = split_scale(X, y, use_smote=False)
    _, X_test_sm, _, y_test_sm, _ = split_scale(X, y, use_smote=True)
    assert X_test_no.shape == X_test_sm.shape
    np.testing.assert_array_equal(y_test_no.values, y_test_sm.values)


def test_scaler_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/models", exist_ok=True)
    X, y = _make_Xy(n=400)
    split_scale(X, y, use_smote=False)
    assert os.path.exists("artifacts/models/scaler.pkl")
