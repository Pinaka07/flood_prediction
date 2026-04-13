"""
tests/test_config.py
--------------------
Tests for src/configuration/config.py
"""
import pytest
from src.configuration import config as cfg


def test_leakage_suspects_not_in_feature_names():
    """Leakage suspects must be removed from FEATURE_NAMES."""
    for suspect in cfg.LEAKAGE_SUSPECTS:
        assert suspect not in cfg.FEATURE_NAMES, (
            f"Leakage suspect '{suspect}' still present in FEATURE_NAMES"
        )


def test_feature_names_subset_of_all():
    """Every safe feature must exist in the full raw feature list."""
    for f in cfg.FEATURE_NAMES:
        assert f in cfg.ALL_FEATURE_NAMES, (
            f"'{f}' in FEATURE_NAMES but not in ALL_FEATURE_NAMES"
        )


def test_counts():
    assert len(cfg.ALL_FEATURE_NAMES) == 30, "Expected 30 raw features"
    assert len(cfg.FEATURE_NAMES)     == 25, "Expected 25 safe features (30 - 5 suspects)"
    assert len(cfg.LEAKAGE_SUSPECTS)  ==  5, "Expected 5 leakage suspects"


def test_constants_in_range():
    assert 0.0 < cfg.TEST_SIZE < 1.0
    assert 0.0 < cfg.SMOTE_SAMPLING_STRATEGY < 1.0
    assert 0.0 < cfg.LEAKAGE_CORR_THRESHOLD <= 1.0
    assert 0.0 < cfg.PROB_THRESHOLD_DEFAULT  < 1.0


def test_no_duplicate_features():
    assert len(cfg.FEATURE_NAMES)     == len(set(cfg.FEATURE_NAMES))
    assert len(cfg.ALL_FEATURE_NAMES) == len(set(cfg.ALL_FEATURE_NAMES))
