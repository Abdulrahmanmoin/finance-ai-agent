"""Typer CLI for the finance-agent."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from finance_agent.config import get_settings
from finance_agent.graph import build_graph
from finance_agent.logging_setup import configure_logging
from finance_agent.schemas import FinalReport
from finance_agent.tools.filings import FilingsRetriever

app = typer.Typer(
    add_completion=False,
    help="Multi-agent equity research powered by LangGraph + Google Gemini.",
)
console = Console()


def _render_report(report: FinalReport) -> None:
    color = {"BUY": "green", "HOLD": "yellow", "SELL": "red"}.get(report.recommendation, "white")
    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold")
    header.add_column()
    header.add_row("Ticker", report.ticker)
    header.add_row("Recommendation", f"[{color}]{report.recommendation}[/{color}]")
    header.add_row("Confidence", report.confidence)
    console.print(Panel(header, title="Final Report", border_style=color))

    console.print(Panel(Markdown(report.rationale), title="Rationale", border_style="cyan"))

    if report.key_risks:
        risks = Table(title="Key Risks", show_header=False, border_style="magenta")
        risks.add_column(style="magenta")
        for risk in report.key_risks:
            risks.add_row(f"- {risk}")
        console.print(risks)

    if report.citations:
        cites = Table(title="Citations", show_header=False, border_style="dim")
        cites.add_column(style="dim")
        for citation in report.citations:
            cites.add_row(f"> {citation}")
        console.print(cites)


@app.command("run")
def run(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Stock ticker, e.g. NVDA.")],
    question: Annotated[str, typer.Option("--question", "-q", help="User question.")],
    thread_id: Annotated[
        str | None,
        typer.Option(
            "--thread-id",
            help="Conversation thread for memory; reuse to continue a session.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Stream node updates."),
    ] = False,
) -> None:
    """Run the full multi-agent workflow on TICKER for QUESTION."""
    configure_logging()
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id or f"cli-{uuid.uuid4().hex[:8]}"}}
    payload = {"ticker": ticker.upper(), "question": question}

    if verbose:
        for update in graph.stream(payload, config=config, stream_mode="updates"):
            for node, delta in update.items():
                console.log(f"[blue]{node}[/blue] produced keys: {list(delta or {})}")
        final_state = graph.get_state(config).values
        report = final_state.get("final_report")
    else:
        with console.status("[bold cyan]Running multi-agent research…", spinner="dots"):
            result = graph.invoke(payload, config=config)
        report = result.get("final_report")

    if report is None:
        console.print("[red]No final report was produced. Check logs.[/red]")
        raise typer.Exit(code=1)

    _render_report(report)


@app.command("build-index")
def build_index(
    filings_path: Annotated[
        Path | None,
        typer.Option(
            "--filings-path",
            help="Override FILINGS_TEXT_PATH from .env.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """(Re)build the FAISS index over the configured filing."""
    configure_logging()
    settings = get_settings()
    if filings_path is not None:
        settings = settings.model_copy(update={"filings_text_path": filings_path})

    retriever = FilingsRetriever(settings=settings)
    retriever.build()
    console.print(f"[green]Index built at[/green] {retriever.index_dir}")


@app.callback()
def _callback() -> None:
    """Entrypoint group."""


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
