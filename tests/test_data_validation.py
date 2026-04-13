"""
tests/test_data_validation.py
------------------------------
Tests for src/components/data_validation.py
"""
import pytest
import numpy as np
import pandas as pd
from src.configuration.config import ALL_FEATURE_NAMES
from src.components.data_validation import validate_data
from src.exception.exception import CustomException


# ── Fixture ────────────────────────────────────────────────────────────────

def _make_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw DataFrame with all 30 columns including leakage suspects."""
    rng = np.random.default_rng(seed)
    data = {col: rng.random(n) for col in ALL_FEATURE_NAMES}
    # ~3.5 % minority — mirrors real dataset ratio
    data["flash_flood"] = np.where(rng.random(n) < 0.035, 1, 0)
    return pd.DataFrame(data)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_valid_dataframe_passes():
    df = _make_df()
    assert validate_data(df) is True


def test_integer_target_passes():
    df = _make_df()
    df["flash_flood"] = df["flash_flood"].astype(int)
    assert validate_data(df) is True


def test_boolean_target_passes():
    df = _make_df()
    df["flash_flood"] = df["flash_flood"].astype(bool)
    assert validate_data(df) is True


def test_missing_feature_column_raises():
    df = _make_df().drop(columns=["Rain_Flag"])
    with pytest.raises(CustomException, match="Missing columns"):
        validate_data(df)


def test_missing_target_column_raises():
    df = _make_df().drop(columns=["flash_flood"])
    with pytest.raises(CustomException, match="Target column"):
        validate_data(df)


def test_invalid_target_values_raises():
    df = _make_df()
    df["flash_flood"] = 99
    with pytest.raises(CustomException, match="Unexpected values"):
        validate_data(df)


def test_all_null_column_raises():
    df = _make_df()
    df["Pressure_hPa"] = np.nan
    with pytest.raises(CustomException, match="entirely null"):
        validate_data(df)


def test_non_numeric_feature_raises():
    df = _make_df()
    df["Rain_Flag"] = "bad_string"
    with pytest.raises(CustomException, match="Non-numeric"):
        validate_data(df)
