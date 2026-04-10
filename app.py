from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from src.configuration.config import FEATURE_NAMES

app = FastAPI()

# 🔥 CORS (IMPORTANT for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 Load models at startup
MODEL_DIR = "artifacts/models"

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))

    iso_path = os.path.join(MODEL_DIR, "Isolation_Forest.pkl")
    iso_model = joblib.load(iso_path) if os.path.exists(iso_path) else None

    print("Models loaded successfully")

except Exception as e:
    print("Error loading models:", e)
    scaler = None
    model = None
    iso_model = None


# ✅ Root endpoint
@app.get("/")
def home():
    return {"message": "Flood Prediction API is running"}


# ✅ Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# ✅ Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        if model is None or scaler is None:
            return {"error": "Model not loaded properly"}

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Ensure feature order
        X = df[FEATURE_NAMES]

        # Scale
        X = scaler.transform(X)

        # Optional anomaly detection
        if iso_model is not None:
            if iso_model.predict(X)[0] == -1:
                return {
                    "prediction": 1,
                    "note": "Anomaly detected"
                }

        # Predict
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
            pred = int(prob > 0.5)
        else:
            pred = int(model.predict(X)[0])
            prob = None

        return {
            "prediction": pred,
            "probability": float(prob) if prob is not None else None
        }

    except Exception as e:
        return {"error": str(e)}