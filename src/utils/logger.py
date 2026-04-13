"""
Centralised logger — import get_logger() from any module.
Produces colourised, levelled output; INFO by default.
"""
from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(_LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_LOG_LEVEL)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger