"""
training_pipeline.py
--------------------
End-to-end training orchestration.

Pipeline steps
--------------
1.  Load data + leakage audit            (data_ingestion)
2.  Imbalance report + diagnostic plots  (model_evaluation)
3.  Stratified split + StandardScaler    (data_transformation)
4.  SMOTE oversampling on train only     (data_transformation)
5.  Build balanced model registry        (model_trainer)
6.  GridSearchCV (scoring=f1, cv=5)      (this file)
7.  Evaluate with optimal thresholds     (model_evaluation)
8.  Generate all plots                   (model_evaluation)
9.  Save best model + all models         (joblib)
10. MLflow logging                       (mlflow)
"""

import os
import warnings
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd

# Suppress pandas FutureWarning for replace() downcasting (display noise only)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

from src.components.data_ingestion import load_data
from src.components.data_transformation import split_scale
from src.components.data_validation import validate_data
from src.components.model_trainer import get_models, PARAM_GRIDS
from src.components.model_evaluation import (
    evaluate,
    plot_all_roc,
    plot_pr_curves,
    plot_threshold_analysis,
    plot_all_confusion_matrices,
    plot_metrics_comparison,
    plot_class_distribution,
    plot_leakage_audit,
    plot_feature_importance,
    summarize_results,
)
from src.configuration.config import (
    DATA_PATH, MODEL_DIR, RANDOM_STATE,
    ALL_FEATURE_NAMES,
)
import src.configuration.config as cfg_module
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training(use_smote: bool = True, skip_tuning: bool = False) -> None:
    """
    Run the full training pipeline.

    Parameters
    ----------
    use_smote    : Apply SMOTE to training data (default True)
    skip_tuning  : Skip GridSearchCV — useful for fast test runs (default False)
    """
    mlflow.set_experiment("flood_prediction_balanced")

    # ── 1. Validate schema ────────────────────────────────────────────────
    logger.info("══ Step 1 — Schema Validation ══")
    raw_df = pd.read_csv(DATA_PATH)
    validate_data(raw_df)

    # ── 2. Load + leakage audit ───────────────────────────────────────────
    logger.info("══ Step 2 — Data Loading + Leakage Audit ══")
    X, y = load_data(DATA_PATH)

    # After load_data(), cfg_module.FEATURE_NAMES is updated with safe features
    safe_features = cfg_module.FEATURE_NAMES
    logger.info("Features used for training: %d", len(safe_features))

    # ── 3. Diagnostic plots ────────────────────────────────────────────────
    logger.info("══ Step 3 — Diagnostic Plots ══")
    raw_df["flash_flood"] = raw_df["flash_flood"].map({False:0,True:1,0:0,1:1})
    plot_class_distribution(y)
    plot_leakage_audit(raw_df, "flash_flood", ALL_FEATURE_NAMES)

    # ── 4. Split + Scale + SMOTE ──────────────────────────────────────────
    logger.info("══ Step 4 — Split / Scale / SMOTE ══")
    X_train, X_test, y_train, y_test, scaler = split_scale(X, y, use_smote=use_smote)

    # ── 5. Model registry ─────────────────────────────────────────────────
    logger.info("══ Step 5 — Build Balanced Model Registry ══")
    models = get_models(y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    results  = {}
    best_model     = None
    best_f1        = 0.0
    best_model_name = ""
    trained_models  = {}

    # End any stale run from a previous interrupted execution
    mlflow.end_run()
    with mlflow.start_run():

        mlflow.log_param("use_smote",    use_smote)
        mlflow.log_param("skip_tuning",  skip_tuning)
        mlflow.log_param("n_features",   len(safe_features))
        mlflow.log_param("train_size",   len(y_train))
        mlflow.log_param("test_size",    len(y_test))
        mlflow.log_param("flood_train",  int(np.array(y_train).sum()))

        # ── 6. Hyperparameter tuning ───────────────────────────────────────
        logger.info("══ Step 6 — Hyperparameter Tuning ══")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # 3 folds: 3× faster than 5

        for name, model in models.items():

            if name == "Isolation Forest":
                # Unsupervised — GridSearchCV does not apply
                logger.info("  %-24s  → skipping tuning (unsupervised)", name)
                trained_models[name] = model
                continue

            if skip_tuning or name not in PARAM_GRIDS:
                logger.info("  %-24s  → using default params", name)
                trained_models[name] = model
            else:
                logger.info("  Tuning %-20s …", name)
                search = GridSearchCV(
                    estimator=model,
                    param_grid=PARAM_GRIDS[name],
                    cv=cv,
                    scoring="f1",       # ← F1, not accuracy
                    n_jobs=-1,
                    verbose=1,          # show progress per fit
                )
                search.fit(X_train, y_train)
                trained_models[name] = search.best_estimator_
                logger.info(
                    "    best CV F1=%.4f  params=%s",
                    search.best_score_, search.best_params_,
                )
                mlflow.log_params({
                    f"{name.replace(' ','_')}_best_{k}": v
                    for k, v in search.best_params_.items()
                })

        # ── 7. Evaluate with optimal threshold ────────────────────────────
        logger.info("══ Step 7 — Evaluation (threshold-optimised) ══")

        for name, model in trained_models.items():
            res = evaluate(name, model, X_train, X_test, y_train, y_test)
            results[name] = res

            safe = name.replace(" ", "_")
            mlflow.log_metric(f"{safe}_f1",        res["f1"])
            mlflow.log_metric(f"{safe}_recall",     res["recall"])
            mlflow.log_metric(f"{safe}_precision",  res["precision"])
            mlflow.log_metric(f"{safe}_threshold",  res["threshold"])
            if res["pr_auc"]:
                mlflow.log_metric(f"{safe}_pr_auc", res["pr_auc"])
            if res["roc_auc"]:
                mlflow.log_metric(f"{safe}_roc_auc",res["roc_auc"])

            mlflow.sklearn.log_model(model, safe)

            # Save individual model
            joblib.dump(model, os.path.join(MODEL_DIR, f"{safe}.pkl"))

            # Track best by F1
            if res["f1"] > best_f1:
                best_f1        = res["f1"]
                best_model     = model
                best_model_name = name

        joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_f1",   best_f1)
        logger.info("Best model: %s  (F1=%.4f)", best_model_name, best_f1)

        # ── 8. Aggregate plots ─────────────────────────────────────────────
        logger.info("══ Step 8 — Generating Plots ══")
        plot_all_roc(results, y_test)
        plot_pr_curves(results, y_test)
        plot_threshold_analysis(results, y_test)
        plot_all_confusion_matrices(results, y_test)
        plot_metrics_comparison(results)
        plot_feature_importance(trained_models, safe_features)

        # ── 9. Print summary + classification reports ──────────────────────
        summarize_results(results)

        print("\n===== CLASSIFICATION REPORTS =====")
        for name, res in results.items():
            print(f"\n── {name} ──")
            print(classification_report(
                y_test, res["y_pred"],
                target_names=["No Flood", "Flood"],
                zero_division=0,
            ))

    logger.info("Training complete. Outputs → artifacts/")
