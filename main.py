"""
main.py
-------
Entry point for the flash flood prediction training pipeline.

Usage
-----
    # Full pipeline (with SMOTE + GridSearchCV)
    python main.py

    # Skip SMOTE (use class_weight only)
    python main.py --no-smote

    # Fast run — skip GridSearchCV (for quick testing)
    python main.py --skip-tuning

    # Both
    python main.py --no-smote --skip-tuning
"""

import argparse
import sys
from src.pipeline.training_pipeline import run_training
from src.exception.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flash Flood Prediction — Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling (use class_weight='balanced' only)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip GridSearchCV for a fast test run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        logger.info("Starting Flash Flood Prediction Pipeline")
        logger.info("  use_smote   = %s", not args.no_smote)
        logger.info("  skip_tuning = %s", args.skip_tuning)
        run_training(
            use_smote=not args.no_smote,
            skip_tuning=args.skip_tuning,
        )
    except Exception as e:
        raise CustomException(e, sys)
