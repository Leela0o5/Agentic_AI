from __future__ import annotations

import argparse
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    from src.ingest import ingest_file

    console.print()
    console.print(Panel.fit("[bold]RAG CLI[/bold] — Ingest", border_style="cyan"))

    try:
        result = ingest_file(args.file_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    console.print()
    console.print(Panel(
        f"[bold green]Done![/bold green]\n\n"
        f"  File:   [cyan]{result['filename']}[/cyan]\n"
        f"  Pages:  {result['pages_processed']}\n"
        f"  Chunks: {result['chunks_ingested']}",
        border_style="green",
        expand=False,
    ))
    console.print()


def cmd_query(args: argparse.Namespace) -> None:
    from src.query import run_query

    console.print()
    console.print(Panel.fit("[bold]RAG CLI[/bold] — Query", border_style="cyan"))
    console.print(f"\n[bold]Question:[/bold] {args.question}\n")

    if args.verbose:
        console.print("[dim]Verbose mode on — showing re-ranking process.[/dim]\n")

    try:
        result = run_query(args.question, verbose=args.verbose)
    except RuntimeError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    console.print(Panel(
        result["answer"],
        title="[bold green]Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    sources = result.get("sources", [])
    if sources:
        table = Table(
            title="Sources",
            box=box.SIMPLE_HEAD,
            title_style="bold yellow",
            header_style="bold dim",
        )
        table.add_column("Filename", style="cyan")
        table.add_column("Page", justify="center")
        for src in sources:
            table.add_row(src["filename"], str(src["page_number"]))
        console.print()
        console.print(table)

    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="Framework-free RAG CLI — index docs, ask questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ingest report.pdf\n"
            "  python main.py query \"What are the key findings?\"\n"
            "  python main.py query \"Summarize section 2\" --verbose\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    ingest_p = sub.add_parser("ingest", help="Index a .pdf or .txt file.")
    ingest_p.add_argument("file_path", help="Path to the file to index.")

    query_p = sub.add_parser("query", help="Ask a question against indexed docs.")
    query_p.add_argument("question", help="Your question (use quotes).")
    query_p.add_argument("--verbose", "-v", action="store_true",
                         help="Show re-ranking process in detail.")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "query": cmd_query}[args.command](args)


if __name__ == "__main__":
    main()
