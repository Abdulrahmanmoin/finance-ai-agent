"""Entrypoint for ``python -m finance_agent``."""

from __future__ import annotations

from finance_agent.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
