"""
tests/test_model_trainer.py
----------------------------
Tests for src/components/model_trainer.py
"""
import pytest
import numpy as np
from src.components.model_trainer import get_models, PARAM_GRIDS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest


EXPECTED_MODELS = {
    "Logistic Regression", "Decision Tree",
    "Random Forest", "Gradient Boost", "Isolation Forest",
}


def _y_train(n=700, flood_frac=0.035, seed=0):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < flood_frac, 1, 0)


# ── Registry ───────────────────────────────────────────────────────────────

def test_registry_contains_all_models():
    models = get_models(_y_train())
    assert set(models.keys()) == EXPECTED_MODELS


def test_registry_correct_types():
    models = get_models(_y_train())
    assert isinstance(models["Logistic Regression"], LogisticRegression)
    assert isinstance(models["Decision Tree"],       DecisionTreeClassifier)
    assert isinstance(models["Random Forest"],       RandomForestClassifier)
    assert isinstance(models["Gradient Boost"],      GradientBoostingClassifier)
    assert isinstance(models["Isolation Forest"],    IsolationForest)


# ── Imbalance handling ─────────────────────────────────────────────────────

def test_logistic_regression_class_weight_balanced():
    m = get_models(_y_train())["Logistic Regression"]
    assert m.class_weight == "balanced"


def test_decision_tree_depth_constrained():
    m = get_models(_y_train())["Decision Tree"]
    assert m.max_depth is not None and m.max_depth <= 10, \
        "Decision Tree max_depth must be constrained to prevent memorisation"
    assert m.min_samples_leaf >= 10, \
        "min_samples_leaf must be ≥ 10 to prevent overfitting minority class"
    assert m.class_weight == "balanced"


def test_random_forest_balanced_subsample():
    m = get_models(_y_train())["Random Forest"]
    assert m.class_weight == "balanced_subsample"
    assert m.max_depth is not None, "Random Forest max_depth should be constrained"
    assert m.min_samples_leaf >= 5


def test_gradient_boost_subsample_lt_one():
    m = get_models(_y_train())["Gradient Boost"]
    assert m.subsample < 1.0, \
        "subsample < 1.0 adds stochastic regularisation to Gradient Boost"
    assert m.learning_rate <= 0.1


def test_contamination_clipped():
    """Contamination must be in [0.01, 0.5] regardless of y_train distribution."""
    # All negative
    y_all_neg = np.zeros(500)
    models = get_models(y_all_neg)
    assert models["Isolation Forest"].contamination >= 0.01

    # All positive (edge case)
    y_all_pos = np.ones(500)
    models = get_models(y_all_pos)
    assert models["Isolation Forest"].contamination <= 0.5


# ── PARAM_GRIDS ────────────────────────────────────────────────────────────

def test_param_grids_no_isolation_forest():
    assert "Isolation Forest" not in PARAM_GRIDS, \
        "IsolationForest is unsupervised — must not have a GridSearchCV param grid"


def test_param_grids_all_supervised_present():
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boost"]:
        assert name in PARAM_GRIDS, f"'{name}' missing from PARAM_GRIDS"


def test_param_grids_include_class_weight():
    """Supervised grids should lock in class_weight to prevent accidentally removing it."""
    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        grid = PARAM_GRIDS[name]
        assert "class_weight" in grid, \
            f"'{name}' param_grid missing 'class_weight' — imbalance won't be handled"


def test_gradient_boost_grid_has_subsample():
    assert "subsample" in PARAM_GRIDS["Gradient Boost"], \
        "Gradient Boost grid should search subsample values for regularisation"
