"""
model_trainer.py
----------------
Model registry with imbalance-aware defaults and hyperparameter grids.

Key changes vs original
-----------------------
- class_weight='balanced'   on LR, DT, RF  →  penalises missing a flood 28×
- max_depth / min_samples_leaf constraints  →  prevents memorisation
- subsample + lower learning_rate on GB    →  stochastic regularisation
- Param grids tuned for F1 (not accuracy)  →  correct objective for imbalance
- Isolation Forest removed from GridSearch →  it is unsupervised, no CV needed
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from src.configuration.config import RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_models(y_train) -> dict:
    """
    Return a dict of name → estimator, each configured to handle the
    ~28:1 class imbalance present in the flash flood dataset.

    Parameters
    ----------
    y_train : array-like — training labels (used only for contamination rate)
    """
    contamination = float(np.clip(y_train.mean(), 0.01, 0.5))
    logger.info("Contamination rate (flood fraction): %.4f", contamination)

    models = {
        # L2 regularisation tightened (C=0.1 vs default 1.0)
        "Logistic Regression": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),

        # max_depth=6, min_samples_leaf=20 — stops the tree from
        # memorising the tiny minority class on training data
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        # balanced_subsample weights each bootstrap separately
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # subsample=0.8 adds stochastic noise → reduces overfitting
        # lower learning_rate forces more trees to correct errors gradually
        "Gradient Boost": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),

        # Unsupervised — fitted on normal-class samples only in model_evaluation
        "Isolation Forest": IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
        ),
    }

    logger.info("Model registry built: %s", list(models.keys()))
    return models


# ── Hyperparameter grids (used by training_pipeline GridSearchCV) ──────────
# Grids are optimised for F1 score, NOT accuracy.
# IsolationForest is excluded — unsupervised, GridSearchCV does not apply.

# ── How many GridSearchCV fits each model runs (combos × cv_folds) ──────────
# Logistic Regression:  3 combos × 3 folds =   9 fits  (~seconds)
# Decision Tree:        6 combos × 3 folds =  18 fits  (~seconds)
# Random Forest:        4 combos × 3 folds =  12 fits  (~2-5 min)
# Gradient Boost:       4 combos × 3 folds =  12 fits  (~5-10 min)
# Total: 51 fits — manageable on 120k rows.
#
# Rule: each param list has at most 2-3 values.
# cv_folds is set to 3 in training_pipeline for speed (was 5).

PARAM_GRIDS = {
    "Logistic Regression": {
        "C":            [0.1, 1.0, 10.0],   # 3 values → 9 fits
        "class_weight": ["balanced"],
    },
    "Decision Tree": {
        "max_depth":        [6, 8],          # 2 values
        "min_samples_leaf": [10, 20, 50],    # 3 values → 6 combos → 18 fits
        "class_weight":     ["balanced"],
    },
    "Random Forest": {
        "n_estimators":     [200],           # fixed — already good default
        "max_depth":        [10, None],      # 2 values
        "min_samples_leaf": [5, 10],         # 2 values → 4 combos → 12 fits
        "class_weight":     ["balanced_subsample"],
    },
    "Gradient Boost": {
        "n_estimators":  [100],              # fixed — faster, good enough
        "learning_rate": [0.05, 0.1],        # 2 values
        "max_depth":     [3, 4],             # 2 values → 4 combos → 12 fits
        "subsample":     [0.8],              # fixed
    },
}
