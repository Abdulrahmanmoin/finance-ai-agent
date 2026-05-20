"""Standalone CLI to (re)build the FAISS index over the configured filing.

Usage:
    python scripts/build_index.py                # uses FILINGS_TEXT_PATH from .env
    python scripts/build_index.py path/to.txt    # override source filing
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the filings FAISS index.")
    parser.add_argument(
        "filings_path",
        nargs="?",
        type=Path,
        help="Optional path to override FILINGS_TEXT_PATH from the environment.",
    )
    args = parser.parse_args(argv)

    from finance_agent.config import get_settings
    from finance_agent.logging_setup import configure_logging
    from finance_agent.tools.filings import FilingsRetriever

    configure_logging()
    logger = logging.getLogger("scripts.build_index")

    settings = get_settings()
    if args.filings_path is not None:
        settings = settings.model_copy(update={"filings_text_path": args.filings_path})

    retriever = FilingsRetriever(settings=settings)
    retriever.build()
    logger.info("Index built at %s", retriever.index_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
