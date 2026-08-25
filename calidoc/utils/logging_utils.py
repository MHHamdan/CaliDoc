"""
Logging Utilities
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: Optional[str] = None
):
    """
    Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional file to log to
        format_str: Log format string
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
