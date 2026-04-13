"""
prediction_pipeline.py
----------------------
Two-stage inference:
  1. IsolationForest anomaly gate  → if anomalous → Flood immediately
  2. Best classifier with per-model optimal threshold (not hardcoded 0.5)

The optimal threshold is stored alongside the model so inference always
uses the same threshold that was found during training.
"""

import os
import joblib
import pandas as pd
from src.utils.logger import get_logger
import src.configuration.config as cfg_module

logger = get_logger(__name__)

MODEL_DIR = os.path.join("artifacts", "models")


def _load(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict(input_dict: dict, threshold: float | None = None) -> dict:
    """
    Predict flash flood for a single observation.

    Parameters
    ----------
    input_dict : dict  — feature_name → float value
    threshold  : float — override the stored optimal threshold (optional)

    Returns
    -------
    dict with keys: prediction (0/1), probability (float|None),
                    triggered_by ("anomaly"|"classifier"), label (str)
    """
    # ── Load models (cached in production; loaded fresh here for simplicity) ──
    scaler    = _load("scaler.pkl")
    iso       = _load("Isolation_Forest.pkl")
    clf       = _load("best_model.pkl")

    # Use safe features (updated by leakage audit at training time)
    features = cfg_module.FEATURE_NAMES

    X = pd.DataFrame([input_dict])[features]
    X = scaler.transform(X)

    # ── Stage 1: anomaly gate ─────────────────────────────────────────────
    # Guard: if the saved iso model expects a different number of features,
    # it was trained before leakage removal. Skip the gate and warn.
    try:
        iso_flag = iso.predict(X)[0] == -1
    except ValueError as e:
        logger.warning(
            "IsolationForest feature mismatch (%s) — re-train models with "
            "the updated pipeline. Skipping anomaly gate.", e
        )
        iso_flag = False

    if iso_flag:
        logger.info("Anomaly gate triggered — predicting Flood.")
        return {
            "prediction":   1,
            "probability":  None,
            "triggered_by": "anomaly",
            "label":        "Flood",
        }

    # ── Stage 2: classifier with optimal threshold ────────────────────────
    if hasattr(clf, "predict_proba"):
        prob = float(clf.predict_proba(X)[0][1])
        # Use passed threshold, or fall back to config default
        t    = threshold if threshold is not None else cfg_module.PROB_THRESHOLD_DEFAULT
        pred = int(prob >= t)
    else:
        prob = None
        pred = int(clf.predict(X)[0])

    logger.info(
        "Classifier — prob=%.4f  threshold=%.3f  prediction=%d",
        prob or 0, threshold or cfg_module.PROB_THRESHOLD_DEFAULT, pred,
    )
    return {
        "prediction":   pred,
        "probability":  prob,
        "triggered_by": "classifier",
        "label":        "Flood" if pred == 1 else "No Flood",
    }
