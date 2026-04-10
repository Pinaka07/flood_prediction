from src.components.data_ingestion import load_data
from src.components.data_transformation import split_scale
from src.components.model_trainer import get_models
from src.components.model_evaluation import evaluate,plot_all_roc,summarize_results
from src.configuration.config import DATA_PATH,MODEL_DIR
import os,joblib

import mlflow
import mlflow.sklearn

def run_training():

    mlflow.set_experiment("flood_prediction")

    X,y=load_data(DATA_PATH)
    X_train,X_test,y_train,y_test,sc=split_scale(X,y)

    models=get_models(y_train)
    results={}

    best_model=None
    best_f1=0

    os.makedirs(MODEL_DIR,exist_ok=True)

    with mlflow.start_run():

        for name,model in models.items():

            res=evaluate(name,model,X_train,X_test,y_train,y_test)
            results[name]=res

            print(f"{name}: F1={res['f1']:.4f}")

            safe_name=name.replace(" ","_")

            mlflow.log_param(f"model_{safe_name}","used")
            mlflow.log_metric(f"{safe_name}_f1",res["f1"])
            mlflow.log_metric(f"{safe_name}_accuracy",res["accuracy"])

            if res["roc_auc"] is not None:
                mlflow.log_metric(f"{safe_name}_roc_auc",res["roc_auc"])

            mlflow.sklearn.log_model(model,safe_name)

            if res["f1"]>best_f1:
                best_f1=res["f1"]
                best_model=model

            joblib.dump(model,os.path.join(MODEL_DIR,f"{safe_name}.pkl"))

        joblib.dump(best_model,os.path.join(MODEL_DIR,"best_model.pkl"))

        # 🔥 log comparison plots
        plot_all_roc(results,y_test)

    summarize_results(results)

    print("\nBest model saved successfully")