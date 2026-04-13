"""
artifact_entity.py
------------------
Typed dataclasses for pipeline stage outputs.
Each stage returns one of these instead of bare tuples or dicts,
making inter-stage contracts explicit and IDE-navigable.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class DataIngestionArtifact:
    """Output of data_ingestion.load_data()"""
    X:              pd.DataFrame
    y:              pd.Series
    feature_names:  List[str]          # leakage-safe feature list
    leaked_dropped: List[str]          # features removed by leakage audit
    n_samples:      int = 0
    flood_count:    int = 0
    no_flood_count: int = 0

    @property
    def imbalance_ratio(self) -> float:
        return self.no_flood_count / self.flood_count if self.flood_count else float("inf")


@dataclass
class DataTransformationArtifact:
    """Output of data_transformation.split_scale()"""
    X_train:        np.ndarray
    X_test:         np.ndarray
    y_train:        Any                # np.ndarray after SMOTE, pd.Series before
    y_test:         pd.Series
    scaler:         Any                # fitted StandardScaler
    smote_applied:  bool = False
    train_flood_count: int = 0        # flood count AFTER SMOTE


@dataclass
class ModelTrainerArtifact:
    """Output of model training step"""
    models:          Dict[str, Any]   # name → fitted estimator
    best_model_name: str
    best_model:      Any
    param_grids_used: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelEvaluationArtifact:
    """Output of model_evaluation.evaluate() for a single model"""
    model_name:  str
    accuracy:    float
    f1:          float
    recall:      float
    precision:   float
    roc_auc:     Optional[float]
    pr_auc:      Optional[float]
    threshold:   float                # optimal decision threshold (not 0.5)
    y_pred:      np.ndarray
    y_prob:      Optional[np.ndarray]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accuracy":  self.accuracy,
            "f1":        self.f1,
            "recall":    self.recall,
            "precision": self.precision,
            "roc_auc":   self.roc_auc,
            "pr_auc":    self.pr_auc,
            "threshold": self.threshold,
            "y_pred":    self.y_pred,
            "y_prob":    self.y_prob,
        }

    def summary_line(self) -> str:
        auc = f"{self.roc_auc:.4f}" if self.roc_auc else " N/A "
        pr  = f"{self.pr_auc:.4f}"  if self.pr_auc  else " N/A "
        return (
            f"  {self.model_name:<24}  F1={self.f1:.4f}  "
            f"Rec={self.recall:.4f}  Prec={self.precision:.4f}  "
            f"PR-AUC={pr}  ROC-AUC={auc}  t={self.threshold:.3f}"
        )
