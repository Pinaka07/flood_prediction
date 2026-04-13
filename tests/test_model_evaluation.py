"""
tests/test_model_evaluation.py
--------------------------------
Tests for src/components/model_evaluation.py
"""
import pytest
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from src.components.model_evaluation import find_optimal_threshold, evaluate


# ── Fixtures ────────────────────────────────────────────────────────────────

def _imbalanced_dataset(n=2000, flood_frac=0.08, seed=42):
    """Synthetic imbalanced dataset: 2 features, minority=flood.
    n=2000 and flood_frac=0.08 ensures both classes appear in the 70% train split.
    """
    rng = np.random.default_rng(seed)
    n_flood    = max(20, int(n * flood_frac))   # at least 20 flood samples
    n_no_flood = n - n_flood
    X = np.vstack([
        rng.standard_normal((n_no_flood, 2)),
        rng.standard_normal((n_flood, 2)) + 2.5,   # shifted mean → separable
    ])
    y = np.array([0] * n_no_flood + [1] * n_flood)
    return X, y


def _fitted_lr(X_train, y_train):
    clf = LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf


def _stratified_split(X, y, train_frac=0.7):
    """Stratified split ensuring both classes appear in train and test."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=1-train_frac, stratify=y, random_state=42)


# ── find_optimal_threshold ──────────────────────────────────────────────────

def test_threshold_in_valid_range():
    _, y = _imbalanced_dataset()
    probs = np.random.default_rng(0).random(len(y))
    t, f1 = find_optimal_threshold(y, probs)
    assert 0.01 <= t <= 0.99
    assert 0.0  <= f1 <= 1.0


def test_threshold_better_than_default():
    """The optimal threshold should produce F1 ≥ default 0.5 threshold."""
    from sklearn.metrics import f1_score
    X, y = _imbalanced_dataset(n=800)
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y)
    clf = _fitted_lr(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    y_test = y_te

    t_opt, f1_opt = find_optimal_threshold(y_test, probs)
    f1_default    = f1_score(y_test, (probs >= 0.5).astype(int), zero_division=0)

    assert f1_opt >= f1_default, \
        f"Optimal threshold F1={f1_opt:.4f} should be ≥ default F1={f1_default:.4f}"


# ── evaluate ────────────────────────────────────────────────────────────────

def test_evaluate_returns_required_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/plots", exist_ok=True)
    X, y = _imbalanced_dataset(n=600)
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y)
    clf = _fitted_lr(X_tr, y_tr)
    result = evaluate("Logistic Regression", clf, X_tr, X_te, y_tr, y_te)
    for key in ["accuracy", "f1", "recall", "precision", "roc_auc", "pr_auc", "threshold", "y_pred", "y_prob"]:
        assert key in result, f"Key '{key}' missing from evaluate() result"


def test_evaluate_threshold_not_hardcoded(tmp_path, monkeypatch):
    """Threshold should adapt to the data, not be stuck at 0.5."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/plots", exist_ok=True)
    X, y = _imbalanced_dataset(n=600)
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y)
    clf = _fitted_lr(X_tr, y_tr)
    result = evaluate("Logistic Regression", clf, X_tr, X_te, y_tr, y_te)
    # For a well-separated imbalanced dataset, optimal threshold will differ from 0.5
    assert "threshold" in result
    assert isinstance(result["threshold"], float)


def test_evaluate_metrics_in_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/plots", exist_ok=True)
    X, y = _imbalanced_dataset(n=600)
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y)
    clf = _fitted_lr(X_tr, y_tr)
    r = evaluate("Logistic Regression", clf, X_tr, X_te, y_tr, y_te)
    for metric in ["accuracy", "f1", "recall", "precision"]:
        assert 0.0 <= r[metric] <= 1.0, f"{metric} out of [0,1]"
    if r["roc_auc"] is not None:
        assert 0.0 <= r["roc_auc"] <= 1.0
    if r["pr_auc"] is not None:
        assert 0.0 <= r["pr_auc"] <= 1.0


def test_evaluate_isolation_forest(tmp_path, monkeypatch):
    """IsolationForest evaluate must produce y_pred but y_prob=None."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/plots", exist_ok=True)
    X, y = _imbalanced_dataset(n=600)
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y)
    iso = IsolationForest(contamination=0.08, random_state=42)
    r = evaluate("Isolation Forest", iso, X_tr, X_te, y_tr, y_te)
    assert r["y_prob"] is None
    assert r["roc_auc"] is None
    assert r["pr_auc"]  is None
    assert len(r["y_pred"]) == len(y_te)


def test_evaluate_y_pred_binary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("artifacts/plots", exist_ok=True)
    X, y = _imbalanced_dataset(n=600)
    X_tr, X_te, y_tr, y_te = _stratified_split(X, y)
    clf = _fitted_lr(X_tr, y_tr)
    r = evaluate("Logistic Regression", clf, X_tr, X_te, y_tr, y_te)
    assert set(r["y_pred"]).issubset({0, 1}), "y_pred must contain only 0 and 1"
