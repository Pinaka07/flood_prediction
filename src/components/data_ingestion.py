import pandas as pd
from src.configuration.config import FEATURE_NAMES

def load_data(path):

    df=pd.read_csv(path)

    df["flash_flood"]=df["flash_flood"].map({False:0,True:1,0:0,1:1})

    if "masterTime" in df.columns:
        df.drop(columns=["masterTime"],inplace=True)

    X=df[FEATURE_NAMES]
    y=df["flash_flood"]

    return X,y