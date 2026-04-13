"""
config_entity.py
----------------
Typed configuration dataclass that mirrors src/configuration/config.py.
Used to pass a single config object between pipeline components instead
of importing individual constants from multiple places.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    data_path:               str
    model_dir:               str
    plot_dir:                str
    test_size:               float
    random_state:            int
    smote_sampling_strategy: float
    leakage_corr_threshold:  float
    prob_threshold_default:  float
    cv_folds:                int = 5
    cv_scoring:              str = "f1"
    feature_names:           List[str] = field(default_factory=list)
    leakage_suspects:        List[str] = field(default_factory=list)


def build_training_config() -> TrainingConfig:
    """
    Construct TrainingConfig from the flat constants in configuration/config.py.
    Centralises config access so components only need to import this function.
    """
    from src.configuration.config import (
        DATA_PATH, MODEL_DIR, PLOT_DIR,
        TEST_SIZE, RANDOM_STATE,
        SMOTE_SAMPLING_STRATEGY,
        LEAKAGE_CORR_THRESHOLD,
        PROB_THRESHOLD_DEFAULT,
        FEATURE_NAMES,
        LEAKAGE_SUSPECTS,
    )
    return TrainingConfig(
        data_path=DATA_PATH,
        model_dir=MODEL_DIR,
        plot_dir=PLOT_DIR,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        smote_sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        leakage_corr_threshold=LEAKAGE_CORR_THRESHOLD,
        prob_threshold_default=PROB_THRESHOLD_DEFAULT,
        feature_names=list(FEATURE_NAMES),
        leakage_suspects=list(LEAKAGE_SUSPECTS),
    )
