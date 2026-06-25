"""
logger_config.py — Centralized logging for the Face Emotion Detector.

Provides structured console logging for the Flask API and WebSocket
handlers, with configurable levels per environment.
"""

import logging
import sys
from os import getenv


def configure_logging(app_name: str = "emotion-detector") -> logging.Logger:
    """
    Configure application logging.

    LOG_LEVEL env var overrides the default:
    - development: INFO
    - production: WARNING
    """
    env = getenv("FLASK_ENV", "production")
    default_level = "INFO" if env == "development" else "WARNING"
    log_level = getenv("LOG_LEVEL", default_level)

    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logger.level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger(name: str = "emotion-detector") -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)
