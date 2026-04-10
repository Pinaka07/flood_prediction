from src.utils.logger import get_logger
logger=get_logger(__name__)
import joblib,pandas as pd
from src.configuration.config import FEATURE_NAMES

def predict(input_dict):
    sc=joblib.load("artifacts/models/scaler.pkl")
    iso=joblib.load("artifacts/models/Isolation_Forest.pkl")
    clf=joblib.load("artifacts/models/best_model.pkl")

    X=pd.DataFrame([input_dict])[FEATURE_NAMES]
    X=sc.transform(X)

    if iso.predict(X)[0]==-1:
        return 1

    if hasattr(clf,"predict_proba"):
     prob=clf.predict_proba(X)[0][1]
    else:
     prob=clf.predict(X)[0]

    return 1 if prob>0.5 else 0