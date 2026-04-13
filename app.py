"""
app.py
------
FastAPI inference server.

Changes vs original
-------------------
- /predict now uses the optimal threshold per model (not hardcoded 0.5)
- /predict accepts an optional `threshold` field to override
- /health reports which features are active (post-leakage-audit)
- Loads leakage-safe FEATURE_NAMES from config (set at training time)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import os

import src.configuration.config as cfg_module
from src.configuration.config import PROB_THRESHOLD_DEFAULT

app = FastAPI(title="Flash Flood Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_DIR    = os.path.join(BASE_DIR, "artifacts", "models")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# --- Run server if executed directly ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


# ── Load models ────────────────────────────────────────────────────────────
def _load(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path) if os.path.exists(path) else None


try:
    scaler    = _load("scaler.pkl")
    model     = _load("best_model.pkl")
    iso_model = _load("Isolation_Forest.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print("Error loading models:", e)
    scaler = model = iso_model = None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    try:
        with open(os.path.join(FRONTEND_DIR, "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h3>Error loading frontend: {e}</h3>"


@app.get("/health")
def health():
    return {
        "status":           "ok",
        "model_loaded":     model is not None,
        "active_features":  cfg_module.FEATURE_NAMES,
        "n_features":       len(cfg_module.FEATURE_NAMES),
    }


@app.post("/predict")
def predict(data: dict):
    """
    Request body: feature key-value pairs + optional `threshold` override.

    Example
    -------
    {
      "Rain_Flag": 1.0,
      "Pressure_hPa": 899.0,
      ...
      "threshold": 0.35        ← optional
    }
    """
    try:
        if model is None or scaler is None:
            return {"error": "Model not loaded"}

        # Extract optional threshold override
        threshold = float(data.pop("threshold", PROB_THRESHOLD_DEFAULT))
        features  = cfg_module.FEATURE_NAMES

        df = pd.DataFrame([data])

        # Validate features
        missing = set(features) - set(df.columns)
        if missing:
            return {"error": f"Missing features: {sorted(missing)}"}

        X = scaler.transform(df[features])

        # Stage 1: anomaly gate
        if iso_model is not None and iso_model.predict(X)[0] == -1:
            return {
                "prediction":   1,
                "probability":  None,
                "triggered_by": "anomaly",
                "label":        "Flood",
            }

        # Stage 2: classifier
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0][1])
            pred = int(prob >= threshold)
        else:
            prob = None
            pred = int(model.predict(X)[0])

        return {
            "prediction":   pred,
            "probability":  prob,
            "triggered_by": "classifier",
            "threshold":    threshold,
            "label":        "Flood" if pred == 1 else "No Flood",
        }

    except Exception as e:
        return {"error": str(e)}
