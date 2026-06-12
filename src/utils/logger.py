"""Centralised logging configuration for AayuSense."""
import logging
import sys
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create and return a configured Logger instance.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        log_file: Optional file path; if provided, a FileHandler is added.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Cannot open log file '%s': %s", log_file, exc)

    return logger
