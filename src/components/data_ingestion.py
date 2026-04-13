"""
data_ingestion.py
-----------------
Load raw CSV, run leakage audit, return feature matrix X and target y.

Leakage audit
-------------
Computes |Pearson r| between every feature and the target.
Features above LEAKAGE_CORR_THRESHOLD are dropped AND the module-level
FEATURE_NAMES list in config is updated so all downstream components stay
in sync without requiring any manual edits.
"""

import warnings
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
import src.configuration.config as cfg_module
from src.configuration.config import (
    ALL_FEATURE_NAMES,
    LEAKAGE_CORR_THRESHOLD,
    LEAKAGE_SUSPECTS,
)

logger = get_logger(__name__)


def leakage_audit(df: pd.DataFrame, target: str) -> list[str]:
    """
    Return list of feature columns to drop due to likely data leakage.

    Strategy
    --------
    1. Compute |Pearson r| with target for every numeric feature.
    2. Flag columns above LEAKAGE_CORR_THRESHOLD (default 0.80).
    3. Always include LEAKAGE_SUSPECTS (known cumulative rain aggregates).
    """
    corr = (
        df[ALL_FEATURE_NAMES]
        .corrwith(df[target])
        .abs()
        .sort_values(ascending=False)
    )

    logger.info("Feature–target |Pearson r| (top 10):")
    for feat, val in corr.head(10).items():
        flag = "  ← LEAKAGE" if val >= LEAKAGE_CORR_THRESHOLD else ""
        logger.info("  %-26s  %.4f%s", feat, val, flag)

    high_corr = corr[corr >= LEAKAGE_CORR_THRESHOLD].index.tolist()
    to_drop   = sorted(set(high_corr) | set(LEAKAGE_SUSPECTS))

    if to_drop:
        logger.warning("Dropping %d leakage features: %s", len(to_drop), to_drop)
    else:
        logger.info("No leakage features detected above threshold.")

    return to_drop


def load_data(path: str):
    """
    Load CSV, encode target, run leakage audit, return (X, y).

    Side-effect
    -----------
    Updates cfg_module.FEATURE_NAMES in-place so model_trainer,
    prediction_pipeline, and app.py all see the same safe feature list.
    """
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s   nulls: %d", df.shape, df.isnull().sum().sum())

    # Encode target
    df["flash_flood"] = df["flash_flood"].astype(float).fillna(0).astype(int).clip(0, 1)

    # Drop non-feature admin columns
    for col in ["masterTime", "flood_risk_level"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Log class distribution
    counts = df["flash_flood"].value_counts().to_dict()
    total  = len(df)
    n0 = int(counts.get(0, counts.get(0.0, 0)))
    n1 = int(counts.get(1, counts.get(1.0, 0)))
    logger.info(
        "Class distribution — No Flood: %d (%.1f%%)  Flood: %d (%.1f%%)",
        n0, n0 / total * 100,
        n1, n1 / total * 100,
    )
    logger.info("Imbalance ratio: %.1f : 1", n0 / max(n1, 1))

    # Leakage audit — updates config.FEATURE_NAMES for all modules
    leaked = leakage_audit(df, "flash_flood")
    safe_features = [f for f in ALL_FEATURE_NAMES if f not in leaked]
    cfg_module.FEATURE_NAMES = safe_features          # ← propagate to all importers
    logger.info("Safe features after audit: %d / %d", len(safe_features), len(ALL_FEATURE_NAMES))

    X = df[safe_features]
    y = df["flash_flood"]

    return X, y
