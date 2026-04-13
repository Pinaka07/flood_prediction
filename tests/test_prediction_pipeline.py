"""
tests/test_prediction_pipeline.py
-----------------------------------
Tests for src/pipeline/prediction_pipeline.py
"""
import pytest
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import src.configuration.config as cfg_module
from src.configuration.config import FEATURE_NAMES


# ── Setup: write minimal fake models so predict() can load them ────────────

@pytest.fixture(autouse=True)
def fake_models(tmp_path, monkeypatch):
    """Create toy sklearn models and patch MODEL_DIR to tmp_path."""
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "artifacts" / "models"
    model_dir.mkdir(parents=True)

    n, n_feats = 300, len(FEATURE_NAMES)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n_feats))
    y = np.where(rng.random(n) < 0.035, 1, 0)

    scaler = StandardScaler().fit(X)
    X_sc   = scaler.transform(X)

    clf = LogisticRegression(class_weight="balanced", max_iter=200)
    clf.fit(X_sc, y)

    iso = IsolationForest(contamination=0.035, random_state=0)
    iso.fit(X_sc[y == 0])

    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(clf,    model_dir / "best_model.pkl")
    joblib.dump(iso,    model_dir / "Isolation_Forest.pkl")

    # Patch the MODEL_DIR constant in prediction_pipeline
    import src.pipeline.prediction_pipeline as pp
    monkeypatch.setattr(pp, "MODEL_DIR", str(model_dir))


def _sample():
    rng = np.random.default_rng(7)
    return {f: float(rng.standard_normal()) for f in FEATURE_NAMES}


# ── Tests ──────────────────────────────────────────────────────────────────

def test_predict_returns_dict():
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample())
    assert isinstance(result, dict)


def test_predict_required_keys():
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample())
    for key in ["prediction", "probability", "triggered_by", "label"]:
        assert key in result, f"Key '{key}' missing from predict() output"


def test_predict_binary_prediction():
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample())
    assert result["prediction"] in (0, 1)


def test_predict_valid_label():
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample())
    assert result["label"] in ("Flood", "No Flood")


def test_predict_triggered_by_valid():
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample())
    assert result["triggered_by"] in ("anomaly", "classifier")


def test_predict_probability_range():
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample())
    if result["probability"] is not None:
        assert 0.0 <= result["probability"] <= 1.0


def test_predict_missing_feature_raises():
    from src.pipeline.prediction_pipeline import predict
    bad = {f: 1.0 for f in FEATURE_NAMES[:-1]}   # one feature missing
    with pytest.raises(Exception):
        predict(bad)


def test_predict_flood_label_matches_prediction():
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample())
    if result["prediction"] == 1:
        assert result["label"] == "Flood"
    else:
        assert result["label"] == "No Flood"


def test_threshold_override():
    """Passing threshold=0.0 should force every classifier output to Flood."""
    from src.pipeline.prediction_pipeline import predict
    result = predict(_sample(), threshold=0.0)
    # If not triggered by anomaly gate, classifier with t=0.0 → always flood
    if result["triggered_by"] == "classifier":
        assert result["prediction"] == 1

    result2 = predict(_sample(), threshold=1.0)
    if result2["triggered_by"] == "classifier":
        assert result2["prediction"] == 0
