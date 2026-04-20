"""
generate_ppt_graphs.py
======================
Generates every graph needed for the Flood Prediction PPT.

HOW TO RUN
----------
    # From the root of the flood_prediction project:
    pip install shap imbalanced-learn matplotlib seaborn scikit-learn pandas numpy joblib
    python generate_ppt_graphs.py

OUTPUT
------
All images are saved to:   ppt_graphs/<SLIDE_XX_description>/
One PNG per graph, named clearly so you can drag-and-drop into PowerPoint.

PROJECT-SPECIFIC SETTINGS (edit if needed)
-------------------------------------------
DATA_PATH   : path to your CSV file
TARGET_COL  : target column name
MODEL_DIR   : folder that contains the saved .pkl files
OUTPUT_DIR  : where generated graphs are saved
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    learning_curve, validation_curve, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve,
    classification_report,
    precision_score, recall_score, f1_score,
)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ─── !! EDIT THESE IF YOUR PATHS ARE DIFFERENT !! ────────────────────────────
DATA_PATH  = "final_data_derived_with_flash_flood_indicator.csv"
TARGET_COL = "flash_flood"
MODEL_DIR  = os.path.join("artifacts", "models")
OUTPUT_DIR = "ppt_graphs"

LEAKAGE_COLS = ["RR_3hr", "RR_6hr", "RR_12hr", "rain_intensity", "dRR"]
NON_NUMERIC_COLS = ["masterTime", "flood_risk_level"]
RANDOM_STATE = 42
TEST_SIZE    = 0.30
# ─────────────────────────────────────────────────────────────────────────────

# ─── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f9f9f9",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

PALETTE = {
    "primary":   "#1565C0",
    "danger":    "#C62828",
    "success":   "#2E7D32",
    "warning":   "#F9A825",
    "neutral":   "#546E7A",
    "models":    ["#1565C0", "#2E7D32", "#F9A825", "#C62828"],
}


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def save(fig, folder: str, filename: str):
    """Save figure and print confirmation."""
    path = Path(OUTPUT_DIR) / folder
    path.mkdir(parents=True, exist_ok=True)
    full = path / filename
    fig.savefig(full, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK]  {full}")


def load_models() -> dict:
    """Load all saved .pkl models from MODEL_DIR."""
    names = {
        "Logistic Regression": "Logistic_Regression.pkl",
        "Decision Tree":       "Decision_Tree.pkl",
        "Random Forest":       "Random_Forest.pkl",
        "Gradient Boost":      "Gradient_Boost.pkl",
    }
    loaded = {}
    for name, fname in names.items():
        fpath = os.path.join(MODEL_DIR, fname)
        if os.path.exists(fpath):
            loaded[name] = joblib.load(fpath)
        else:
            print(f"  [WARN]  {fpath} not found - skipping {name}")
    return loaded


def load_data():
    """Load CSV, drop leakage columns, return X, y."""
    df = pd.read_csv(DATA_PATH)
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    cols_to_drop += [c for c in NON_NUMERIC_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return df, X, y


def split_scale(X, y):
    """Stratified split + StandardScaler (fit on train only)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        sc = joblib.load(scaler_path)
        X_tr_sc = sc.transform(X_tr)
        X_te_sc = sc.transform(X_te)
    else:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc  = sc.transform(X_te)
    X_tr_sc = pd.DataFrame(X_tr_sc, columns=X.columns)
    X_te_sc  = pd.DataFrame(X_te_sc,  columns=X.columns)
    return X_tr_sc, X_te_sc, y_tr, y_te


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 3 — Dataset Overview
# ════════════════════════════════════════════════════════════════════════════

def slide3_dataset_overview(df, y):
    print("\n[SLIDE 3] Dataset Overview")

    # 3-a  Class Distribution
    counts = y.value_counts().sort_index()
    labels = ["No Flood", "Flood"]
    colors = [PALETTE["primary"], PALETTE["danger"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f"{val:,}\n({val/len(y)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Class Distribution — Flash Flood Dataset")
    ax.set_ylabel("Number of Records")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "SLIDE_03_dataset_overview", "class_distribution.png")

    # 3-b  Missing Value Heatmap
    missing = df.isnull().sum()
    if missing.sum() == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No missing values found in dataset",
                ha="center", va="center", fontsize=14, color=PALETTE["success"])
        ax.axis("off")
        ax.set_title("Missing Value Check")
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(df.isnull().T, cbar=False, cmap="Reds", ax=ax)
        ax.set_title("Missing Values Heatmap (red = missing)")
    save(fig, "SLIDE_03_dataset_overview", "missing_values.png")

    # 3-c  Dataset Summary Stats table
    stats = df.describe().T[["mean","std","min","max"]].round(2)
    fig, ax = plt.subplots(figsize=(12, max(4, len(stats)*0.35)))
    ax.axis("off")
    tbl = ax.table(
        cellText=stats.values,
        rowLabels=stats.index,
        colLabels=["Mean","Std Dev","Min","Max"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax.set_title("Dataset Summary Statistics", pad=20, fontsize=13)
    save(fig, "SLIDE_03_dataset_overview", "summary_stats_table.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 4 — EDA: Correlation & KDE
# ════════════════════════════════════════════════════════════════════════════

def slide4_eda_correlation(df, y):
    print("\n[SLIDE 4] EDA — Correlation & KDE")

    # 4-a  Correlation Heatmap
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
                center=0, linewidths=0.3, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save(fig, "SLIDE_04_eda_correlation", "correlation_heatmap.png")

    # 4-b  Target correlation bar chart (top 15 features)
    target_corr = df.corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    top15 = target_corr.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_bar = [PALETTE["danger"] if v > 0.3 else PALETTE["primary"] for v in top15.values]
    ax.barh(top15.index[::-1], top15.values[::-1], color=colors_bar[::-1])
    ax.set_title("Feature Correlation with flash_flood (Top 15)")
    ax.set_xlabel("|Pearson r|")
    ax.axvline(x=0.3, color=PALETTE["warning"], linestyle="--", label="0.3 threshold")
    ax.legend()
    save(fig, "SLIDE_04_eda_correlation", "target_correlation_bar.png")

    # 4-c  KDE plots — top 6 most correlated features
    top6 = target_corr.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(top6):
        sns.kdeplot(df[y == 0][col], ax=axes[i], label="No Flood",
                    fill=True, color=PALETTE["primary"], alpha=0.5)
        sns.kdeplot(df[y == 1][col], ax=axes[i], label="Flood",
                    fill=True, color=PALETTE["danger"], alpha=0.5)
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)
    plt.suptitle("Feature Distributions: Flood vs No-Flood (Top 6 Features)", y=1.02)
    plt.tight_layout()
    save(fig, "SLIDE_04_eda_correlation", "kde_by_class.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 5 — EDA: Box Plots & Pairplot
# ════════════════════════════════════════════════════════════════════════════

def slide5_eda_boxplots(df, y):
    print("\n[SLIDE 5] EDA - Box Plots")

    target_corr = df.corr()[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    top8 = target_corr.head(8).index.tolist()

    # 5-a  Box plots for top 8 features split by class (using matplotlib to avoid seaborn bug)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(top8):
        no_flood = df[y == 0][col].dropna()
        flood = df[y == 1][col].dropna()
        data = [no_flood, flood]
        bp = axes[i].boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True)
        for patch, color in zip(bp['boxes'], [PALETTE["primary"], PALETTE["danger"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[i].set_xticklabels(['No Flood', 'Flood'])
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("")
    plt.suptitle("Box Plots: Top 8 Features by Flood Class", y=1.01, fontsize=14)
    plt.tight_layout()
    save(fig, "SLIDE_05_eda_boxplots", "boxplots_by_class.png")

    # 5-b  Pairplot (top 4 only)
    top4 = top8[:4]
    pair_df = df[top4 + [TARGET_COL]].sample(n=min(3000, len(df)), random_state=42)
    pair_df[TARGET_COL] = pair_df[TARGET_COL].map({0: "No Flood", 1: "Flood"})
    g = sns.pairplot(pair_df, hue=TARGET_COL, diag_kind="kde",
                     palette=[PALETTE["primary"], PALETTE["danger"]],
                     plot_kws={"alpha": 0.4, "s": 20})
    g.fig.suptitle("Pairplot - Top 4 Features by Class", y=1.02, fontsize=13)
    save(g.fig, "SLIDE_05_eda_boxplots", "pairplot.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 7 — Data Transformation: Before/After Scaling
# ════════════════════════════════════════════════════════════════════════════

def slide7_transformation(X, y):
    print("\n[SLIDE 7] Data Transformation")

    target_corr = X.corrwith(y).abs().sort_values(ascending=False)
    top3 = target_corr.head(3).index.tolist()

    sc = StandardScaler()
    X_sc = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for i, col in enumerate(top3):
        axes[0, i].hist(X[col], bins=40, color=PALETTE["primary"], alpha=0.8, edgecolor="white")
        axes[0, i].set_title(f"{col}\n(Before Scaling)")
        axes[0, i].set_ylabel("Count" if i == 0 else "")

        axes[1, i].hist(X_sc[col], bins=40, color=PALETTE["success"], alpha=0.8, edgecolor="white")
        axes[1, i].set_title(f"{col}\n(After StandardScaler)")
        axes[1, i].set_ylabel("Count" if i == 0 else "")

    plt.suptitle("Before vs After StandardScaler (Top 3 Features)", fontsize=14)
    plt.tight_layout()
    save(fig, "SLIDE_07_transformation", "before_after_scaling.png")

    # SMOTE class imbalance visual
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=0.20, random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_sc.values, y)
        before_counts = dict(zip(*np.unique(y, return_counts=True)))
        after_counts  = dict(zip(*np.unique(y_res, return_counts=True)))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, counts, title in zip(
            axes,
            [before_counts, after_counts],
            ["Before SMOTE", "After SMOTE (strategy=0.20)"]
        ):
            labels = ["No Flood", "Flood"]
            vals   = [counts.get(0, 0), counts.get(1, 0)]
            ax.bar(labels, vals, color=[PALETTE["primary"], PALETTE["danger"]],
                   width=0.5, edgecolor="white")
            for j, v in enumerate(vals):
                ax.text(j, v + 200, f"{v:,}", ha="center", fontsize=10)
            ax.set_title(title)
            ax.set_ylabel("Count")
        plt.suptitle("Class Balance: Before vs After SMOTE Oversampling", fontsize=13)
        plt.tight_layout()
        save(fig, "SLIDE_07_transformation", "smote_before_after.png")
    except Exception as e:
        print(f"  [WARN]  Skipping SMOTE plot: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 8 — Anomaly Detection (Isolation Forest)
# ════════════════════════════════════════════════════════════════════════════

def slide8_anomaly_detection(X, y):
    print("\n[SLIDE 8] Anomaly Detection")

    iso_path = os.path.join(MODEL_DIR, "Isolation_Forest.pkl")
    if os.path.exists(iso_path):
        iso = joblib.load(iso_path)
    else:
        contamination = float(np.clip(y.mean(), 0.01, 0.5))
        iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
        iso.fit(X)

    sample = X.sample(n=min(5000, len(X)), random_state=42)
    preds  = iso.predict(sample)  # -1 = anomaly, 1 = normal

    # Use PCA to reduce to 2D for visualisation
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(sample)

    fig, ax = plt.subplots(figsize=(9, 6))
    normal_mask  = preds == 1
    anomaly_mask = preds == -1

    ax.scatter(coords[normal_mask,  0], coords[normal_mask,  1],
               c=PALETTE["primary"], alpha=0.3, s=15, label=f"Normal ({normal_mask.sum():,})")
    ax.scatter(coords[anomaly_mask, 0], coords[anomaly_mask, 1],
               c=PALETTE["danger"],  alpha=0.7, s=25, label=f"Anomaly ({anomaly_mask.sum():,})",
               marker="x", linewidths=1.5)

    ax.set_title("Isolation Forest — Anomaly Detection\n(2D via PCA)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.legend(fontsize=10)
    save(fig, "SLIDE_08_anomaly_detection", "isolation_forest_scatter.png")

    # Anomaly score distribution
    scores = iso.decision_function(sample)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[normal_mask],  bins=50, color=PALETTE["primary"],
            alpha=0.7, label="Normal",  edgecolor="white")
    ax.hist(scores[anomaly_mask], bins=50, color=PALETTE["danger"],
            alpha=0.7, label="Anomaly", edgecolor="white")
    ax.axvline(x=0, color="black", linestyle="--", label="Decision Boundary")
    ax.set_title("Isolation Forest — Anomaly Score Distribution")
    ax.set_xlabel("Anomaly Score (negative = more anomalous)")
    ax.set_ylabel("Count")
    ax.legend()
    save(fig, "SLIDE_08_anomaly_detection", "anomaly_score_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 9 — Model Training Time
# ════════════════════════════════════════════════════════════════════════════

def slide9_training_time(X_train, y_train):
    print("\n[SLIDE 9] Training Time Comparison")

    model_defs = {
        "Logistic\nRegression": LogisticRegression(C=0.1, class_weight="balanced",
                                                    max_iter=500, random_state=RANDOM_STATE),
        "Decision\nTree":       DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                        random_state=RANDOM_STATE),
        "Random\nForest":       RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                        random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient\nBoost":      GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }

    times = {}
    for name, model in model_defs.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        times[name] = round(time.time() - t0, 2)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = PALETTE["models"]
    bars = ax.barh(list(times.keys()), list(times.values()),
                   color=colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars, times.values()):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val}s", va="center", fontsize=10)
    ax.set_title("Model Training Time Comparison")
    ax.set_xlabel("Time (seconds)")
    ax.set_xlim(0, max(times.values()) * 1.25)
    save(fig, "SLIDE_09_training_time", "training_time_comparison.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 10 — Model Comparison (Accuracy, F1, CV Box Plot)
# ════════════════════════════════════════════════════════════════════════════

def slide10_model_comparison(models, X_train, X_test, y_train, y_test, X, y):
    print("\n[SLIDE 10] Model Comparison")

    if not models:
        print("  [WARN]  No saved models found — skipping slide 10")
        return

    # Re-fit loaded models on training data (needed for fresh predict)
    for name, m in models.items():
        m.fit(X_train, y_train)

    # 10-a  Bar chart: Accuracy & F1
    metrics = {}
    for name, m in models.items():
        y_pred = m.predict(X_test)
        metrics[name] = {
            "Accuracy": np.mean(y_pred == y_test),
            "F1 Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred),
        }

    metric_df = pd.DataFrame(metrics).T
    x = np.arange(len(metric_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, col in enumerate(metric_df.columns):
        ax.bar(x + i*width, metric_df[col], width, label=col,
               color=PALETTE["models"][i], edgecolor="white")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_df.index, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (Test Set)")
    ax.legend()
    for bar in ax.patches:
        if bar.get_height() > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    save(fig, "SLIDE_10_model_comparison", "model_metrics_bar.png")

    # 10-b  Cross-validation box plot (F1)
    cv_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sc  = StandardScaler()
    X_np = sc.fit_transform(X)

    for name, m in models.items():
        scores = cross_val_score(m, X_np, y, cv=skf, scoring="f1", n_jobs=-1)
        cv_results[name] = scores

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(cv_results.values(), labels=cv_results.keys(),
                    patch_artist=True, notch=False, widths=0.4)
    for patch, color in zip(bp["boxes"], PALETTE["models"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("5-Fold Cross-Validation F1 Score Distribution")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)
    save(fig, "SLIDE_10_model_comparison", "cv_boxplot.png")

    # 10-c  Radar / Spider chart
    categories = list(metric_df.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (name, row) in enumerate(metric_df.iterrows()):
        vals = row.tolist() + row.tolist()[:1]
        ax.plot(angles, vals, "o-", linewidth=2,
                label=name, color=PALETTE["models"][i])
        ax.fill(angles, vals, alpha=0.1, color=PALETTE["models"][i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar Chart", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    save(fig, "SLIDE_10_model_comparison", "radar_chart.png")

    return metrics


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 11 — ROC & Precision-Recall Curves
# ════════════════════════════════════════════════════════════════════════════

def slide11_roc_pr(models, X_train, X_test, y_train, y_test):
    print("\n[SLIDE 11] ROC & PR Curves")

    if not models:
        return

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr,  ax_pr  = plt.subplots(figsize=(8, 6))

    for i, (name, m) in enumerate(models.items()):
        m.fit(X_train, y_train)
        if hasattr(m, "predict_proba"):
            y_prob = m.predict_proba(X_test)[:, 1]
        else:
            y_prob = m.decision_function(X_test)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc      = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, color=PALETTE["models"][i],
                    label=f"{name} (AUC={roc_auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc        = auc(rec, prec)
        ax_pr.plot(rec, prec, lw=2, color=PALETTE["models"][i],
                   label=f"{name} (AUC={pr_auc:.3f})")

    # ROC styling
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves — All Models")
    ax_roc.legend(loc="lower right")
    save(fig_roc, "SLIDE_11_roc_pr", "roc_curves.png")

    # PR styling
    baseline = y_test.mean()
    ax_pr.axhline(y=baseline, color="black", linestyle="--",
                  label=f"Random ({baseline:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves — All Models\n(Better metric for imbalanced data)")
    ax_pr.legend()
    save(fig_pr, "SLIDE_11_roc_pr", "pr_curves.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 12 — Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════

def slide12_confusion_matrix(models, X_train, X_test, y_train, y_test):
    print("\n[SLIDE 12] Confusion Matrix")

    if not models:
        return

    best_name = max(
        models,
        key=lambda n: f1_score(y_test, models[n].predict(X_test))
    )
    best = models[best_name]

    y_pred = best.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Flood", "Flood"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    # Annotate with meaning
    labels_meaning = [
        ["True Negative\n(Correct 'safe')", "False Positive\n(False alarm [WARN])"],
        ["False Negative\n(Missed flood! )", "True Positive\n(Caught flood )"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.38, labels_meaning[i][j],
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    ax.set_title(f"Confusion Matrix — {best_name}\n(Best Model by F1 Score)")
    save(fig, "SLIDE_12_confusion_matrix", "confusion_matrix.png")

    # Classification report as heatmap
    report = classification_report(y_test, y_pred,
                                    target_names=["No Flood", "Flood"],
                                    output_dict=True)
    report_df = pd.DataFrame(report).T.iloc[:2, :3]  # precision/recall/f1

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(report_df.astype(float), annot=True, fmt=".3f",
                cmap="YlOrRd", ax=ax, vmin=0, vmax=1,
                linewidths=0.5, cbar=False)
    ax.set_title(f"Classification Report — {best_name}")
    save(fig, "SLIDE_12_confusion_matrix", "classification_report_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 13 — Threshold Optimization
# ════════════════════════════════════════════════════════════════════════════

def slide13_threshold(models, X_train, X_test, y_train, y_test):
    print("\n[SLIDE 13] Threshold Optimization")

    if not models:
        return

    best_name = max(
        models,
        key=lambda n: f1_score(y_test, models[n].predict(X_test))
    )
    best = models[best_name]

    if not hasattr(best, "predict_proba"):
        print(f"  [WARN]  {best_name} has no predict_proba — skipping threshold plot")
        return

    y_prob     = best.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.02)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, lw=2, color=PALETTE["primary"],  label="Precision")
    ax.plot(thresholds, recalls,    lw=2, color=PALETTE["success"],  label="Recall")
    ax.plot(thresholds, f1s,        lw=2, color=PALETTE["danger"],   label="F1 Score",
            linestyle="--")
    ax.axvline(x=best_t, color=PALETTE["warning"], linestyle=":", lw=2,
               label=f"Optimal Threshold = {best_t:.2f}")
    ax.axvline(x=0.5,    color="gray",             linestyle=":", lw=1,
               label="Default Threshold = 0.50")
    ax.set_title(f"Threshold Optimization — {best_name}\n"
                 f"(Default 0.5 may miss floods; optimal threshold = {best_t:.2f})")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    save(fig, "SLIDE_13_threshold", "threshold_optimization.png")

    # Calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color=PALETTE["primary"],
            lw=2, label=f"{best_name}")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_title("Calibration Curve\n(How reliable are the predicted probabilities?)")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.legend()
    save(fig, "SLIDE_13_threshold", "calibration_curve.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 14 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════

def slide14_feature_importance(models, X_train, y_train, feature_names):
    print("\n[SLIDE 14] Feature Importance")

    tree_models = {k: v for k, v in models.items()
                   if hasattr(v, "feature_importances_")}
    if not tree_models:
        print("  [WARN]  No tree models found — skipping")
        return

    for name, m in tree_models.items():
        importances = m.feature_importances_
        fi_df = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        top20 = fi_df.tail(20)

        fig, ax = plt.subplots(figsize=(9, 8))
        colors_fi = [PALETTE["danger"] if v > top20.quantile(0.8) else PALETTE["primary"]
                     for v in top20.values]
        ax.barh(top20.index, top20.values, color=colors_fi, edgecolor="white")
        ax.set_title(f"Feature Importance — {name}\n(Top 20 Features)")
        ax.set_xlabel("Importance Score")
        fname = name.lower().replace(" ", "_")
        save(fig, "SLIDE_14_feature_importance", f"feature_importance_{fname}.png")

    # Combined comparison (if RF and GB both exist)
    if "Random Forest" in tree_models and "Gradient Boost" in tree_models:
        rf_fi = pd.Series(tree_models["Random Forest"].feature_importances_,
                          index=feature_names).sort_values(ascending=False).head(10)
        gb_fi = pd.Series(tree_models["Gradient Boost"].feature_importances_,
                          index=feature_names)[rf_fi.index]

        x   = np.arange(len(rf_fi))
        w   = 0.35
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w/2, rf_fi.values, w, label="Random Forest",   color=PALETTE["primary"])
        ax.bar(x + w/2, gb_fi.values, w, label="Gradient Boost",  color=PALETTE["warning"])
        ax.set_xticks(x)
        ax.set_xticklabels(rf_fi.index, rotation=45, ha="right", fontsize=9)
        ax.set_title("Feature Importance: Random Forest vs Gradient Boost (Top 10)")
        ax.set_ylabel("Importance Score")
        ax.legend()
        plt.tight_layout()
        save(fig, "SLIDE_14_feature_importance", "rf_vs_gb_importance.png")

    # Decision Tree visualization
    if "Decision Tree" in models:
        dt = models["Decision Tree"]
        fig, ax = plt.subplots(figsize=(20, 8))
        plot_tree(dt, ax=ax, max_depth=3, filled=True,
                  feature_names=feature_names,
                  class_names=["No Flood", "Flood"],
                  impurity=False, proportion=True,
                  rounded=True, fontsize=8)
        ax.set_title("Decision Tree Structure (depth = 3 shown)")
        save(fig, "SLIDE_14_feature_importance", "decision_tree_visual.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 15 — SHAP Explainability
# ════════════════════════════════════════════════════════════════════════════

def slide15_shap(models, X_test):
    print("\n[SLIDE 15] SHAP Explainability")

    try:
        import shap
    except ImportError:
        print("  [WARN]  shap not installed. Run:  pip install shap")
        return

    tree_models = {k: v for k, v in models.items()
                   if k in ("Random Forest", "Gradient Boost")}
    if not tree_models:
        print("  [WARN]  No tree model for SHAP — skipping")
        return

    name, m = next(iter(tree_models.items()))
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

    explainer   = shap.TreeExplainer(m)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers shap_values is a list [class0, class1]
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    # 15-a  SHAP Summary Plot (beeswarm)
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, show=False, plot_size=None)
        ax.set_title(f"SHAP Summary Plot — {name}")
        save(fig, "SLIDE_15_shap", "shap_summary_beeswarm.png")
    except Exception as e:
        print(f"  [WARN]  SHAP beeswarm plot failed: {e}")

    # 15-b  SHAP Bar Plot (mean |SHAP|)
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
        ax.set_title(f"SHAP Feature Importance (Mean |SHAP|) — {name}")
        save(fig, "SLIDE_15_shap", "shap_bar_importance.png")
    except Exception as e:
        print(f"  [WARN]  SHAP bar plot failed: {e}")

    # 15-c  SHAP Waterfall for one prediction
    try:
        explanation = explainer(X_sample)
        exp_class1  = (explanation[..., 1]
                       if explanation.values.ndim == 3
                       else explanation)
        shap.plots.waterfall(exp_class1[0], show=False)
        plt.title(f"SHAP Waterfall — Single Prediction ({name})")
        save(plt.gcf(), "SLIDE_15_shap", "shap_waterfall.png")
    except Exception as e:
        print(f"  [WARN]  Waterfall plot failed: {e}")

    # 15-d  SHAP Dependence Plot for top feature
    try:
        mean_abs = np.abs(sv).mean(axis=0)
        top_feat = X_sample.columns[np.argmax(mean_abs)]
        shap.dependence_plot(top_feat, sv, X_sample)
        plt.title(f"SHAP Dependence Plot — {top_feat}")
        save(plt.gcf(), "SLIDE_15_shap", f"shap_dependence_{top_feat}.png")
    except Exception as e:
        print(f"  [WARN]  Dependence plot failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 16 — Learning & Validation Curves
# ════════════════════════════════════════════════════════════════════════════

def slide16_learning_curves(models, X, y):
    print("\n[SLIDE 16] Learning & Validation Curves")

    if "Random Forest" not in models:
        print("  [WARN]  Random Forest not found — skipping")
        return

    rf = models["Random Forest"]
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    skf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(
        rf, X_sc, y, cv=skf, scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-",
            color=PALETTE["primary"], lw=2, label="Training F1")
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.15, color=PALETTE["primary"])
    ax.plot(train_sizes, val_scores.mean(axis=1), "s-",
            color=PALETTE["danger"], lw=2, label="Validation F1")
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    alpha=0.15, color=PALETTE["danger"])
    ax.set_title("Learning Curve — Random Forest\n(Shows overfitting / underfitting)")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score")
    ax.legend()
    save(fig, "SLIDE_16_learning_curves", "learning_curve.png")

    # Validation Curve — n_estimators
    param_range  = [10, 50, 100, 150, 200, 300]
    train_sc_vc, val_sc_vc = validation_curve(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        X_sc, y, param_name="n_estimators", param_range=param_range,
        cv=skf, scoring="f1", n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(param_range, train_sc_vc.mean(axis=1), "o-",
            color=PALETTE["primary"], lw=2, label="Training F1")
    ax.fill_between(param_range,
                    train_sc_vc.mean(axis=1) - train_sc_vc.std(axis=1),
                    train_sc_vc.mean(axis=1) + train_sc_vc.std(axis=1),
                    alpha=0.15, color=PALETTE["primary"])
    ax.plot(param_range, val_sc_vc.mean(axis=1), "s-",
            color=PALETTE["danger"], lw=2, label="Validation F1")
    ax.fill_between(param_range,
                    val_sc_vc.mean(axis=1) - val_sc_vc.std(axis=1),
                    val_sc_vc.mean(axis=1) + val_sc_vc.std(axis=1),
                    alpha=0.15, color=PALETTE["danger"])
    ax.set_title("Validation Curve — n_estimators (Random Forest)")
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("F1 Score")
    ax.legend()
    save(fig, "SLIDE_16_learning_curves", "validation_curve_n_estimators.png")


# ════════════════════════════════════════════════════════════════════════════
#  SLIDE 20 — Deployment: Flood Risk Gauge Chart
# ════════════════════════════════════════════════════════════════════════════

def slide20_gauge(models, X_test, y_test):
    print("\n[SLIDE 20] Deployment — Gauge Chart")

    best_name = max(
        models,
        key=lambda n: f1_score(y_test, models[n].predict(X_test))
    ) if models else None

    if not best_name or not hasattr(models[best_name], "predict_proba"):
        prob = 0.67
    else:
        probs = models[best_name].predict_proba(X_test)[:, 1]
        # Pick a "medium risk" example
        idx  = np.argmin(np.abs(probs - 0.65))
        prob = float(probs[idx])

    def draw_gauge(ax, value, title):
        theta   = np.linspace(np.pi, 0, 200)
        zones   = [(0.0, 0.33, "#2ecc71", "Low Risk"),
                   (0.33, 0.66, "#f39c12", "Medium Risk"),
                   (0.66, 1.0,  "#e74c3c", "High Risk")]

        for lo, hi, color, label in zones:
            t = np.linspace(np.pi * (1 - hi), np.pi * (1 - lo), 50)
            ax.plot(np.cos(t), np.sin(t), color=color, lw=18, solid_capstyle="butt", alpha=0.85)

        # Needle
        angle = np.pi * (1 - value)
        ax.annotate("", xy=(0.55 * np.cos(angle), 0.55 * np.sin(angle)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=3, color="#2c3e50"))
        ax.add_patch(plt.Circle((0, 0), 0.07, color="#2c3e50", zorder=5))

        ax.text(0, -0.25, f"{value*100:.0f}%", ha="center", va="center",
                fontsize=22, fontweight="bold", color="#2c3e50")
        ax.text(0, -0.45, title, ha="center", va="center", fontsize=12, color="#555")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.6, 1.1)
        ax.axis("off")

        # Legend
        for lo, hi, color, label in zones:
            ax.add_patch(mpatches.Rectangle((lo * 2.2 - 1.1, -0.65), 0.33 * 2.2, 0.12,
                                             color=color, alpha=0.8))
            ax.text(lo * 2.2 - 1.1 + 0.17, -0.58, label,
                    ha="center", va="center", fontsize=8)

    fig, ax = plt.subplots(figsize=(7, 5))
    draw_gauge(ax, prob, f"Flood Probability: {prob*100:.0f}%")
    fig.suptitle("Flood Risk Gauge — Sample Prediction", fontsize=13, y=0.98)
    save(fig, "SLIDE_20_deployment", "flood_risk_gauge.png")

    # Prediction confidence distribution
    if models and hasattr(models[best_name], "predict_proba"):
        all_probs = models[best_name].predict_proba(X_test)[:, 1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(all_probs[y_test == 0], bins=40, alpha=0.6,
                color=PALETTE["primary"], label="No Flood (actual)", edgecolor="white")
        ax.hist(all_probs[y_test == 1], bins=40, alpha=0.7,
                color=PALETTE["danger"],  label="Flood (actual)",    edgecolor="white")
        ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold = 0.5")
        ax.set_title("Prediction Confidence Distribution")
        ax.set_xlabel("Predicted Flood Probability")
        ax.set_ylabel("Count")
        ax.legend()
        save(fig, "SLIDE_20_deployment", "confidence_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FLOOD PREDICTION PPT GRAPH GENERATOR")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[SETUP] Loading data...")
    df, X, y = load_data()
    print(f"  Dataset shape : {df.shape}")
    print(f"  Target balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = split_scale(X, y)
    feature_names = list(X.columns)

    # ── Load saved models ─────────────────────────────────────────────────
    print("\n[SETUP] Loading saved models...")
    models = load_models()
    if models:
        for name, m in models.items():
            m.fit(X_train, y_train)    # re-fit on our split
        print(f"  Loaded: {list(models.keys())}")
    else:
        print("  No models found in artifacts/models/ — training fresh models...")
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        models = {
            "Logistic Regression": LogisticRegression(C=0.1, class_weight="balanced",
                                                       max_iter=500, random_state=RANDOM_STATE),
            "Decision Tree":       DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                           random_state=RANDOM_STATE),
            "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                           random_state=RANDOM_STATE, n_jobs=-1),
            "Gradient Boost":      GradientBoostingClassifier(n_estimators=100,
                                                               random_state=RANDOM_STATE),
        }
        for m in models.values():
            m.fit(X_train, y_train)

    # ── Generate all graphs ───────────────────────────────────────────────
    slide3_dataset_overview(df, y)
    slide4_eda_correlation(df, y)
    slide5_eda_boxplots(df, y)
    slide7_transformation(X, y)
    slide8_anomaly_detection(X, y)
    slide9_training_time(X_train, y_train)
    slide10_model_comparison(models, X_train, X_test, y_train, y_test, X, y)
    slide11_roc_pr(models, X_train, X_test, y_train, y_test)
    slide12_confusion_matrix(models, X_train, X_test, y_train, y_test)
    slide13_threshold(models, X_train, X_test, y_train, y_test)
    slide14_feature_importance(models, X_train, y_train, feature_names)
    slide15_shap(models, X_test)
    slide16_learning_curves(models, X, y)
    slide20_gauge(models, X_test, y_test)

    print("\n" + "=" * 60)
    print(f"    ALL GRAPHS SAVED TO:  {OUTPUT_DIR}/")
    print("  📂  Folder structure:")
    for folder in sorted(Path(OUTPUT_DIR).glob("*")):
        files = list(folder.glob("*.png"))
        print(f"      {folder.name}/  ({len(files)} images)")
    print("=" * 60)
    print("\n  QUICK INSTALL (if needed):")
    print("  pip install shap imbalanced-learn matplotlib seaborn scikit-learn pandas numpy joblib")


if __name__ == "__main__":
    main()
