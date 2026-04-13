"""
tests/test_data_ingestion.py
-----------------------------
Tests for src/components/data_ingestion.py
"""
import pytest
import numpy as np
import pandas as pd
import tempfile, os

from src.configuration.config import ALL_FEATURE_NAMES, LEAKAGE_SUSPECTS
from src.components.data_ingestion import leakage_audit, load_data
import src.configuration.config as cfg_module


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_csv(n: int = 500, seed: int = 0, add_leakage: bool = False) -> str:
    """Write a temporary CSV and return its path."""
    rng = np.random.default_rng(seed)
    data = {col: rng.random(n) for col in ALL_FEATURE_NAMES}
    data["flash_flood"] = np.where(rng.random(n) < 0.035, 1, 0)

    if add_leakage:
        # Make RR_3hr perfectly correlated with target → guaranteed leakage hit
        data["RR_3hr"] = data["flash_flood"].astype(float)

    df = pd.DataFrame(data)
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


# ── leakage_audit tests ────────────────────────────────────────────────────

def test_leakage_audit_returns_list():
    rng = np.random.default_rng(0)
    n = 400
    data = {col: rng.random(n) for col in ALL_FEATURE_NAMES}
    data["flash_flood"] = np.where(rng.random(n) < 0.035, 1, 0)
    df = pd.DataFrame(data)
    dropped = leakage_audit(df, "flash_flood")
    assert isinstance(dropped, list)


def test_leakage_audit_catches_high_correlation():
    """A feature perfectly correlated with target must be flagged."""
    rng = np.random.default_rng(1)
    n = 400
    data = {col: rng.random(n) for col in ALL_FEATURE_NAMES}
    data["flash_flood"] = np.where(rng.random(n) < 0.035, 1, 0)
    data["RR_3hr"] = data["flash_flood"].astype(float)   # perfect correlation
    df = pd.DataFrame(data)
    dropped = leakage_audit(df, "flash_flood")
    assert "RR_3hr" in dropped


def test_leakage_audit_always_drops_suspects():
    """Known suspects are always dropped regardless of computed correlation."""
    rng = np.random.default_rng(2)
    n = 400
    data = {col: rng.random(n) for col in ALL_FEATURE_NAMES}
    data["flash_flood"] = np.where(rng.random(n) < 0.035, 1, 0)
    df = pd.DataFrame(data)
    dropped = leakage_audit(df, "flash_flood")
    for s in LEAKAGE_SUSPECTS:
        assert s in dropped, f"Known suspect '{s}' was not dropped"


# ── load_data tests ────────────────────────────────────────────────────────

def test_load_data_returns_correct_shapes():
    path = _make_csv(n=500)
    try:
        X, y = load_data(path)
        assert X.shape[0] == 500
        assert y.shape[0] == 500
        assert X.shape[1] == len(cfg_module.FEATURE_NAMES)
    finally:
        os.unlink(path)


def test_load_data_removes_leakage_suspects():
    path = _make_csv(n=500)
    try:
        X, _ = load_data(path)
        for suspect in LEAKAGE_SUSPECTS:
            assert suspect not in X.columns, f"Suspect '{suspect}' not removed"
    finally:
        os.unlink(path)


def test_load_data_updates_config_feature_names():
    """load_data must update cfg_module.FEATURE_NAMES in-place."""
    path = _make_csv(n=500)
    try:
        X, _ = load_data(path)
        assert cfg_module.FEATURE_NAMES == list(X.columns)
    finally:
        os.unlink(path)


def test_load_data_encodes_target_as_int():
    path = _make_csv(n=300)
    try:
        _, y = load_data(path)
        assert set(y.unique()).issubset({0, 1})
        assert y.dtype in (int, "int64", "int32")
    finally:
        os.unlink(path)


def test_load_data_file_not_found():
    with pytest.raises(Exception):
        load_data("nonexistent_file.csv")
