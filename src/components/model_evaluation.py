import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score,confusion_matrix,roc_curve
import os
import mlflow

PLOT_DIR="artifacts/plots"

def plot_individual_model(name,y_test,y_pred,y_prob):

    os.makedirs(PLOT_DIR,exist_ok=True)
    safe_name=name.replace(" ","_")

    # 🔥 Confusion Matrix
    cm=confusion_matrix(y_test,y_pred)
    plt.figure()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path=f"{PLOT_DIR}/{safe_name}_cm.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # 🔥 ROC Curve
    if y_prob is not None:
        fpr,tpr,_=roc_curve(y_test,y_prob)
        plt.figure()
        plt.plot(fpr,tpr,label=name)
        plt.plot([0,1],[0,1],'--')
        plt.title(f"{name} - ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.tight_layout()

        roc_path=f"{PLOT_DIR}/{safe_name}_roc.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)


def evaluate(name,model,X_train,X_test,y_train,y_test):

    if name=="Isolation Forest":
        model.fit(X_train[y_train.values==0])
        y_pred=np.where(model.predict(X_test)==-1,1,0)
        y_prob=None
    else:
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        y_prob=model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else None

    plot_individual_model(name,y_test,y_pred,y_prob)

    try:
        roc_auc=roc_auc_score(y_test,y_prob) if y_prob is not None else None
    except:
        roc_auc=None

    return{
        "accuracy":accuracy_score(y_test,y_pred),
        "f1":f1_score(y_test,y_pred,zero_division=0),
        "recall":recall_score(y_test,y_pred,zero_division=0),
        "precision":precision_score(y_test,y_pred,zero_division=0),
        "roc_auc":roc_auc,
        "y_pred":y_pred,
        "y_prob":y_prob
    }


def plot_all_roc(results,y_test):

    plt.figure()

    for name,res in results.items():
        if res["y_prob"] is None:
            continue

        fpr,tpr,_=roc_curve(y_test,res["y_prob"])
        plt.plot(fpr,tpr,label=name)

    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve Comparison")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()

    os.makedirs(PLOT_DIR,exist_ok=True)
    path=f"{PLOT_DIR}/roc_comparison.png"
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)


def summarize_results(results):

    rows=[]

    for name,res in results.items():
        rows.append({
            "Model":name,
            "Accuracy":res["accuracy"],
            "F1 Score":res["f1"],
            "Recall":res["recall"],
            "Precision":res["precision"],
            "ROC AUC":res["roc_auc"]
        })

    df=pd.DataFrame(rows)
    df=df.sort_values(by="F1 Score",ascending=False)

    print("\n===== MODEL PERFORMANCE =====\n")
    print(df.to_string(index=False))

    os.makedirs("artifacts",exist_ok=True)
    df.to_csv("artifacts/model_performance.csv",index=False)

    return df