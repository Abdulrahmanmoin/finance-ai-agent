"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Final

from finance_agent.config import get_settings

_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured: bool = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging exactly once per process."""
    global _configured
    if _configured:
        return

    resolved = (level or get_settings().log_level).upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    # Tame noisy third-party loggers.
    for noisy in ("urllib3", "httpx", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
