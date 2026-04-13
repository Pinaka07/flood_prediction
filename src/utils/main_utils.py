"""
main_utils.py
-------------
General utility functions shared across pipeline components.
"""

import os
import yaml
import joblib
from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path: str) -> dict:
    """
    Load and return a YAML file as a dict.

    Raises
    ------
    FileNotFoundError  if the path does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, "r") as f:
        content = yaml.safe_load(f)
    logger.debug("Loaded YAML: %s", path)
    return content


def save_object(path: str, obj) -> None:
    """Persist any Python object with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved object → %s", path)


def load_object(path: str):
    """
    Load a joblib-serialised object.

    Raises
    ------
    FileNotFoundError  if the file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    logger.info("Loaded object ← %s", path)
    return obj
