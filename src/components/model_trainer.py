from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,IsolationForest

def get_models(y_train):

    contamination=min(max(float(y_train.mean()),0.01),0.5)

    models={

    "Logistic Regression":LogisticRegression(max_iter=200),

    "Decision Tree":DecisionTreeClassifier(),

    "Random Forest":RandomForestClassifier(n_estimators=200),

    "Gradient Boost":GradientBoostingClassifier(),

    "Isolation Forest":IsolationForest(contamination=contamination)
    }

    return models