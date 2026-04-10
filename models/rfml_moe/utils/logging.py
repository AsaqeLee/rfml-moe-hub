"""Logging setup with file and console handlers."""

import logging
import sys
from pathlib import Path
from datetime import datetime

_initialized = False


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Configure root logger with console and file output."""
    global _initialized
    if _initialized:
        return logging.getLogger("rfml")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rfml_{timestamp}.log"

    logger = logging.getLogger("rfml")
    logger.setLevel(getattr(logging, level.upper()))

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _initialized = True
    return logger


def get_logger(name: str = "rfml") -> logging.Logger:
    """Get a named child logger."""
    return logging.getLogger(name)
