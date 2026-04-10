from fastapi import FastAPI
from src.pipeline.prediction_pipeline import predict

app=FastAPI()

@app.get("/")
def home():
    return{"message":"Flood Prediction API"}

@app.get("/health")
def health():
    return{"status":"ok"}

@app.post("/predict")
def get_prediction(data:dict):
    return{"prediction":predict(data)}