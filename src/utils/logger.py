"""
logger.py
---------
Configures logging to both a timestamped file under logs/ AND stdout,
so you can see progress in the terminal while a permanent record is kept.
"""

import logging
import os
import sys
from datetime import datetime

LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

_FORMAT = "[%(asctime)s]  %(levelname)-8s  %(name)s  —  %(message)s"
_DATE   = "%H:%M:%S"

# Root logger — configure once
logging.basicConfig(
    level=logging.INFO,
    format=_FORMAT,
    datefmt=_DATE,
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),   # ← also print to terminal
    ],
    force=True,   # override any previous basicConfig calls
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the root handler config."""
    return logging.getLogger(name)
