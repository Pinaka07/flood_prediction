"""
data_transformation.py
-----------------------
Stratified train/test split + StandardScaler + SMOTE oversampling.

SMOTE is applied to the TRAINING set ONLY — the test set is never touched.
This is critical: applying SMOTE before splitting (or to the full dataset)
causes data leakage and inflated metrics.

SMOTE sampling_strategy=0.20  →  Flood grows to 20 % of No-Flood count.
Forcing 50/50 typically hurts generalisation on real-world imbalanced data.
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.configuration.config import TEST_SIZE, RANDOM_STATE, MODEL_DIR, SMOTE_SAMPLING_STRATEGY
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Try SMOTE ──────────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False
    logger.warning(
        "imbalanced-learn not installed — SMOTE disabled. "
        "Run:  pip install imbalanced-learn"
    )


def split_scale(X, y, use_smote: bool = True):
    """
    1. Stratified 70/30 split.
    2. Fit StandardScaler on train only; transform both splits.
    3. Apply SMOTE to scaled training data (if available and requested).

    Parameters
    ----------
    X, y      : feature matrix and target series from data_ingestion
    use_smote : apply SMOTE oversampling (default True)

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    # ── Split ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,          # preserves 3.4 % minority in every fold
    )
    logger.info("Split — train: %s  test: %s", X_train.shape, X_test.shape)

    # ── Scale ───────────────────────────────────────────────────────────────
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit on train only
    X_test_sc  = scaler.transform(X_test)        # never fit on test

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    logger.info("Scaler saved.")

    # ── SMOTE ───────────────────────────────────────────────────────────────
    if use_smote and _SMOTE_OK:
        before = dict(zip(*np.unique(y_train, return_counts=True)))
        smote  = SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )
        X_train_sc, y_train = smote.fit_resample(X_train_sc, y_train)
        after = dict(zip(*np.unique(y_train, return_counts=True)))
        logger.info("SMOTE applied — before: %s  after: %s", before, after)
    elif use_smote and not _SMOTE_OK:
        logger.warning("SMOTE requested but imbalanced-learn not found. Skipping.")
    else:
        logger.info("SMOTE disabled — using class_weight='balanced' only.")

    return X_train_sc, X_test_sc, y_train, y_test, scaler
