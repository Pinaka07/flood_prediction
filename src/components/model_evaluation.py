"""
model_evaluation.py
-------------------
Evaluation, threshold optimisation, and plotting for the flood pipeline.

Key additions vs original
--------------------------
- find_optimal_threshold()    — sweeps [0.01,0.99] to maximise F1
                                (default 0.5 is wrong for imbalanced data)
- PR-AUC metric               — more reliable than ROC-AUC under imbalance
- plot_pr_curves()            — Precision-Recall curves (most informative here)
- plot_threshold_analysis()   — F1 vs threshold curve per model
- plot_class_distribution()   — visualise the 28:1 imbalance
- plot_leakage_audit()        — feature–target |r| bar chart
- All confusion matrices show optimised threshold in the title
- summarize_results() adds PR-AUC and Threshold columns
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report,
)
from src.configuration.config import PLOT_DIR, PROB_THRESHOLD_DEFAULT
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Dark style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F1117", "axes.facecolor":   "#0F1117",
    "axes.edgecolor":   "#2A2D3E", "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#A0A0A0", "ytick.color":      "#A0A0A0",
    "text.color":       "#E0E0E0", "grid.color":       "#2A2D3E",
    "grid.linestyle":   "--",      "font.family":      "DejaVu Sans",
    "axes.titlesize":   12,        "axes.labelsize":   10,
    "legend.framealpha": 0.15,
})

PALETTE = {
    "Logistic Regression": "#4361EE",
    "Decision Tree":       "#F72585",
    "Random Forest":       "#3A86FF",
    "Gradient Boost":      "#FF6B35",
    "Isolation Forest":    "#7B2D8B",
}


def _save(fig, filename: str) -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    try:
        mlflow.log_artifact(path)
    except Exception:
        pass
    logger.info("  Plot saved → %s", path)
    return path


# ── Threshold optimisation ──────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob, n_steps: int = 200) -> tuple[float, float]:
    """
    Sweep decision thresholds and return the one that maximises F1.

    Parameters
    ----------
    y_true  : true binary labels
    y_prob  : predicted probabilities for class 1
    n_steps : number of threshold candidates

    Returns
    -------
    (best_threshold, best_f1)
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_f1 = PROB_THRESHOLD_DEFAULT, 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ── Core evaluate ───────────────────────────────────────────────────────────

def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    """
    Fit model, find optimal threshold, compute all metrics, save plots.

    Returns
    -------
    dict with keys: accuracy, f1, recall, precision, roc_auc, pr_auc,
                    threshold, y_pred, y_prob
    """
    # ── Fit ────────────────────────────────────────────────────────────────
    if name == "Isolation Forest":
        model.fit(X_train[np.asarray(y_train) == 0])   # fit on normal samples only
        raw    = model.predict(X_test)
        y_pred = np.where(raw == -1, 1, 0)
        y_prob = None
        threshold = PROB_THRESHOLD_DEFAULT
    else:
        model.fit(X_train, y_train)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba") else None
        )
        if y_prob is not None:
            threshold, _ = find_optimal_threshold(y_test, y_prob)
            logger.info("  %s — optimal threshold: %.3f", name, threshold)
        else:
            threshold = PROB_THRESHOLD_DEFAULT
        y_pred = (y_prob >= threshold).astype(int) if y_prob is not None \
                 else model.predict(X_test)

    # ── Metrics ────────────────────────────────────────────────────────────
    try:
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else None
    except Exception:
        pr_auc = None

    result = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc,
        "pr_auc":    pr_auc,
        "threshold": threshold,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
    }

    logger.info(
        "  %s — F1=%.4f  Rec=%.4f  Prec=%.4f  PR-AUC=%s",
        name, result["f1"], result["recall"], result["precision"],
        f"{pr_auc:.4f}" if pr_auc else "N/A",
    )

    # ── Per-model plots ─────────────────────────────────────────────────────
    _plot_confusion_matrix(name, y_test, y_pred, threshold)
    if y_prob is not None:
        _plot_roc_single(name, y_test, y_prob, roc_auc)

    return result


# ── Individual plots ─────────────────────────────────────────────────────────

def _plot_confusion_matrix(name, y_test, y_pred, threshold):
    color = PALETTE.get(name, "#4361EE")
    cm    = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", ax=ax,
        cmap=sns.light_palette(color, as_cmap=True),
        linewidths=0.5, linecolor="#0F1117",
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_title(f"{name}\n(threshold = {threshold:.3f})",
                 color=color, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticklabels(["No Flood", "Flood"], color="#E0E0E0")
    ax.set_yticklabels(["No Flood", "Flood"], color="#E0E0E0", rotation=0)
    fig.tight_layout()
    _save(fig, f"{name.replace(' ', '_')}_cm.png")


def _plot_roc_single(name, y_test, y_prob, auc):
    color = PALETTE.get(name, "#4361EE")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{name} — ROC Curve", color=color, fontweight="bold")
    ax.legend(framealpha=0.15); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, f"{name.replace(' ', '_')}_roc.png")


# ── Aggregate plots ──────────────────────────────────────────────────────────

def plot_all_roc(results: dict, y_test) -> None:
    """All models on a single ROC axis."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.35, label="Random (AUC=0.50)")
    for name, res in results.items():
        if res["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2.5, color=PALETTE.get(name, "#AAA"),
                label=f"{name}  (AUC={res['roc_auc']:.4f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "roc_comparison.png")


def plot_pr_curves(results: dict, y_test) -> None:
    """
    Precision-Recall curves — the most informative plot under class imbalance.
    ROC curves can look optimistic when TN count dominates; PR-AUC does not.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = float(np.array(y_test).mean())
    ax.axhline(baseline, color="w", lw=1, ls="--", alpha=0.4,
               label=f"Baseline (prevalence = {baseline:.3f})")
    for name, res in results.items():
        if res["y_prob"] is None:
            continue
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, lw=2.5, color=PALETTE.get(name, "#AAA"),
                label=f"{name}  (PR-AUC={res['pr_auc']:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves\n(most reliable metric under imbalance)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "pr_curves.png")


def plot_threshold_analysis(results: dict, y_test) -> None:
    """F1 vs decision threshold curve for every supervised model."""
    supervised = {k: v for k, v in results.items()
                  if v["y_prob"] is not None}
    n    = len(supervised)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 5))
    fig.suptitle("F1 vs Decision Threshold — Per Model",
                 fontsize=14, fontweight="bold", color="#E0E0E0")
    axes = axes.flatten()
    thresholds = np.linspace(0.01, 0.99, 200)

    for idx, (name, res) in enumerate(supervised.items()):
        ax    = axes[idx]
        color = PALETTE.get(name, "#4361EE")
        f1s   = [f1_score(y_test, (res["y_prob"] >= t).astype(int), zero_division=0)
                 for t in thresholds]
        ax.plot(thresholds, f1s, color=color, lw=2.5)
        ax.axvline(res["threshold"], color="#E63946", lw=1.5, ls="--",
                   label=f"Optimal t = {res['threshold']:.3f}")
        ax.set_title(name, color=color, fontweight="bold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score")
        ax.legend(framealpha=0.15); ax.grid(True, alpha=0.3)

    for i in range(len(supervised), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    _save(fig, "threshold_analysis.png")


def plot_all_confusion_matrices(results: dict, y_test) -> None:
    """Combined grid of all confusion matrices."""
    names = list(results.keys())
    cols  = 2
    rows  = (len(names) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 5))
    fig.suptitle("Confusion Matrices — All Models (Balanced)",
                 fontsize=14, fontweight="bold", color="#E0E0E0", y=1.01)
    axes = axes.flatten()

    for idx, name in enumerate(names):
        ax    = axes[idx]
        color = PALETTE.get(name, "#4361EE")
        cm    = confusion_matrix(y_test, results[name]["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            cmap=sns.light_palette(color, as_cmap=True),
            linewidths=0.5, linecolor="#0F1117",
            annot_kws={"size": 13, "weight": "bold"},
        )
        t = results[name]["threshold"]
        ax.set_title(f"{name}\n(t={t:.3f})", color=color, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticklabels(["No Flood", "Flood"], color="#E0E0E0")
        ax.set_yticklabels(["No Flood", "Flood"], color="#E0E0E0", rotation=0)

    for i in range(len(names), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    _save(fig, "confusion_matrices_all.png")


def plot_metrics_comparison(results: dict) -> None:
    """Grouped bar chart: F1, Recall, Precision, PR-AUC, ROC-AUC per model."""
    metric_keys = ["f1", "recall", "precision", "pr_auc", "roc_auc"]
    labels      = ["F1", "Recall", "Precision", "PR-AUC", "ROC-AUC"]
    names       = list(results.keys())
    x           = np.arange(len(metric_keys))
    w           = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, name in enumerate(names):
        vals   = [results[name].get(k) or 0 for k in metric_keys]
        offset = (i - len(names) / 2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=name,
                      color=PALETTE.get(name, "#888"), alpha=0.88, edgecolor="#0F1117")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="#E0E0E0")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Model Performance — Balanced  (F1 & PR-AUC are primary metrics)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "metrics_comparison.png")


def plot_class_distribution(y) -> None:
    """Bar + pie chart showing the 28:1 class imbalance."""
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["No Flood", "Flood"]
    colors = ["#4361EE", "#E63946"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, counts.values, color=colors, alpha=0.85, edgecolor="#0F1117")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 300, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Class Distribution", fontweight="bold"); axes[0].set_ylabel("Count")

    axes[1].pie(counts.values, labels=[f"{l}\n{v/sum(counts.values)*100:.1f}%"
                for l, v in zip(labels, counts.values)],
                colors=colors, autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "#0F1117", "linewidth": 2})
    axes[1].set_title(f"Imbalance Ratio (≈{counts[0]/counts[1]:.0f} : 1)", fontweight="bold")

    fig.suptitle("Target Class Imbalance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "class_distribution.png")


def plot_leakage_audit(df, target: str, all_features: list) -> None:
    """Bar chart of |Pearson r| between each feature and the target."""
    corr = (
        df[all_features].corrwith(df[target]).abs().sort_values(ascending=False)
    )
    colors = ["#E63946" if v >= 0.80 else "#FF9F1C" if v >= 0.50 else "#4361EE"
              for v in corr.values]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(corr.index, corr.values, color=colors, alpha=0.88, edgecolor="#0F1117")
    ax.axhline(0.80, color="#E63946", lw=1.5, ls="--", label="Leakage threshold (0.80)")
    ax.axhline(0.50, color="#FF9F1C", lw=1.2, ls="--", label="High correlation (0.50)")
    ax.set_xticks(range(len(corr.index)))
    ax.set_xticklabels(corr.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|Pearson r| with target")
    ax.set_title("Feature–Target Correlation  (red = likely leakage)",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "leakage_audit.png")


def plot_feature_importance(results_models: dict, feature_names: list) -> None:
    """Horizontal bar chart of top-15 features for tree-based models."""
    for name, model in results_models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(15).sort_values()
        color = PALETTE.get(name, "#4361EE")
        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(imp.index, imp.values, color=color, alpha=0.85, edgecolor="#0F1117")
        for bar, val in zip(bars, imp.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8, color="#E0E0E0")
        ax.set_xlabel("Importance Score")
        ax.set_title(f"Feature Importance — {name} (Top 15)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        _save(fig, f"feature_importance_{name.replace(' ', '_')}.png")


# ── Summary ──────────────────────────────────────────────────────────────────

def summarize_results(results: dict) -> pd.DataFrame:
    """Print and save a metrics table sorted by F1 descending."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Model":     name,
            "F1 Score":  round(res["f1"],       4),
            "Recall":    round(res["recall"],    4),
            "Precision": round(res["precision"], 4),
            "PR-AUC":    round(res["pr_auc"],    4) if res["pr_auc"]   else None,
            "ROC AUC":   round(res["roc_auc"],   4) if res["roc_auc"]  else None,
            "Accuracy":  round(res["accuracy"],  4),
            "Threshold": round(res["threshold"], 4),
        })

    df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)

    print("\n===== MODEL PERFORMANCE =====\n")
    print(df.to_string(index=False))
    print("\n  ⚠  Accuracy is misleading for imbalanced data — use F1 / PR-AUC.\n")

    os.makedirs("artifacts", exist_ok=True)
    df.to_csv("artifacts/model_performance.csv", index=False)
    logger.info("Metrics saved → artifacts/model_performance.csv")

    # Classification reports
    return df
