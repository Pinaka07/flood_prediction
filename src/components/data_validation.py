"""
data_validation.py
------------------
Schema validation BEFORE leakage audit and preprocessing.

Validates:
- All raw columns listed in config/schema.yaml are present
- Target column exists and contains only {0, 1, True, False}
- No feature column is entirely null
- Class imbalance is within a usable range (warns but does not fail)
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
from src.utils.main_utils import read_yaml
from src.utils.logger import get_logger
from src.exception.exception import CustomException

logger = get_logger(__name__)


def validate_data(df: pd.DataFrame, schema_path: str = "config/schema.yaml") -> bool:
    """
    Validate raw dataframe against schema.

    Parameters
    ----------
    df          : raw dataframe from pd.read_csv()
    schema_path : path to config/schema.yaml

    Returns
    -------
    True on success

    Raises
    ------
    CustomException on any hard validation failure
    """
    try:
        schema = read_yaml(schema_path)
        expected_cols  = list(schema["columns"].keys())
        target_column  = schema["target_column"]

        # ── 1. Required columns ────────────────────────────────────────────
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in dataset: {missing_cols}")
        logger.info("Column check passed (%d / %d present).",
                    len(expected_cols), len(expected_cols))

        # ── 2. Target column ───────────────────────────────────────────────
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        unique_vals = set(df[target_column].dropna().unique())
        allowed     = {0, 1, True, False}
        bad_vals    = unique_vals - allowed
        if bad_vals:
            raise ValueError(
                f"Unexpected values in target '{target_column}': {bad_vals}"
            )
        logger.info("Target column '%s' validated.", target_column)

        # ── 3. Null check ──────────────────────────────────────────────────
        all_null = [c for c in expected_cols if df[c].isnull().all()]
        if all_null:
            raise ValueError(f"Columns are entirely null: {all_null}")

        null_counts = df[expected_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            logger.warning(
                "Columns with nulls (will be handled in preprocessing): %s",
                cols_with_nulls.to_dict(),
            )

        # ── 4. Class imbalance warning ─────────────────────────────────────
        target_enc = df[target_column].astype(float).fillna(-1).astype(int).map({0:0,1:1,-1:0}).fillna(0).astype(int)
        counts     = target_enc.value_counts()
        total      = len(df)

        if 1 not in counts:
            raise ValueError("No positive (Flood) samples found in dataset.")
        if 0 not in counts:
            raise ValueError("No negative (No-Flood) samples found in dataset.")

        flood_pct = counts[1] / total * 100
        ratio     = counts[0] / counts[1]

        logger.info(
            "Class distribution — No Flood: %d (%.1f%%)  Flood: %d (%.1f%%)  ratio: %.1f:1",
            counts[0], 100 - flood_pct, counts[1], flood_pct, ratio,
        )

        if ratio > 10:
            logger.warning(
                "Severe class imbalance detected (%.0f:1). "
                "SMOTE + class_weight='balanced' will be applied. "
                "Use F1 / PR-AUC — NOT accuracy — to evaluate models.",
                ratio,
            )

        # ── 5. Dtype check ─────────────────────────────────────────────────
        non_numeric = [
            c for c in expected_cols
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise ValueError(f"Non-numeric feature columns: {non_numeric}")
        logger.info("Dtype check passed.")

        logger.info("Data validation PASSED.")
        return True

    except Exception as e:
        raise CustomException(e, sys)
