"""
main.py — Entry point for the Agentic Go Contributor.

Usage:
    python main.py \\
        --repo https://github.com/gin-gonic/gin \\
        --issue https://github.com/gin-gonic/gin/issues/1234

    python main.py \\
        --issue https://github.com/gin-gonic/gin/issues/1234
        # (repo URL is inferred from the issue URL)

The script:
  1. Validates inputs
  2. Configures logging
  3. Runs the LangGraph workflow
  4. Writes outputs to the ./outputs/ directory
  5. Prints a rich summary
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from config.settings import settings

# ── Force UTF-8 output on Windows to handle emoji/unicode in rich output ──────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass  # Python < 3.7 or non-TextIOWrapper stream

# ── Configure logging early ───────────────────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="agentic-go-contributor",
    help="AI agent that autonomously solves GitHub issues in Go repositories.",
    add_completion=False,
)
console = Console(highlight=True)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _write_outputs(state: dict, run_id: str) -> Path:
    """Write all agent outputs to the outputs/ directory."""
    out_dir = settings.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # PR title and body
    pr_title = state.get("pr_title", "")
    pr_body = state.get("pr_body", "")
    (out_dir / "pr_title.txt").write_text(pr_title, encoding="utf-8")
    (out_dir / "pr_body.md").write_text(pr_body, encoding="utf-8")

    # Implementation plan
    plan = state.get("implementation_plan", "")
    if plan:
        (out_dir / "implementation_plan.md").write_text(plan, encoding="utf-8")

    # Proposed changes (write each file)
    proposed = state.get("proposed_changes", {})
    if proposed:
        changes_dir = out_dir / "proposed_changes"
        changes_dir.mkdir(exist_ok=True)
        for rel_path, content in proposed.items():
            safe_name = rel_path.replace("/", "__").replace("\\", "__")
            (changes_dir / safe_name).write_text(content, encoding="utf-8")

    # Test + lint results
    test_res = state.get("test_results", "")
    lint_res = state.get("lint_results", "")
    (out_dir / "test_results.txt").write_text(test_res, encoding="utf-8")
    (out_dir / "lint_results.txt").write_text(lint_res, encoding="utf-8")

    # Full run log
    logs = state.get("logs", [])
    (out_dir / "run_log.txt").write_text("\n".join(logs), encoding="utf-8")

    # Full state (JSON)
    safe_state = {
        k: v for k, v in state.items()
        if isinstance(v, (str, bool, int, float, list, dict, type(None)))
        and k != "repository_map"  # can be very large
    }
    (out_dir / "full_state.json").write_text(
        json.dumps(safe_state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return out_dir


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]🤖 Agentic Go Contributor[/bold cyan]\n"
            "[dim]Powered by LangGraph + Kimi K2 via OpenRouter[/dim]",
            border_style="cyan",
        )
    )


def _print_summary(state: dict, out_dir: Path) -> None:
    """Print a rich summary table of the run results."""
    table = Table(title="Run Summary", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Issue", state.get("issue_url", ""))
    table.add_row("Title", state.get("issue_title", ""))
    table.add_row("Repo", state.get("repo_path", ""))
    table.add_row("Files retrieved", str(len(state.get("retrieved_files", []))))
    table.add_row("Files changed", str(len(state.get("proposed_changes", {}))))
    table.add_row("Validation", "✅ PASSED" if state.get("validation_passed") else "❌ FAILED")
    table.add_row("PR Title", state.get("pr_title", ""))
    table.add_row("Output dir", str(out_dir))

    console.print(table)

    pr_body = state.get("pr_body", "")
    if pr_body:
        console.print(
            Panel(
                Markdown(pr_body[:3000]),
                title="📝 PR Description",
                border_style="green",
            )
        )


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@app.command()
def run(
    issue: str = typer.Option(
        ...,
        "--issue",
        "-i",
        help="GitHub issue URL (e.g. https://github.com/gin-gonic/gin/issues/1234)",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="GitHub repository URL. Inferred from issue URL if not provided.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory. Defaults to ./outputs/<run-id>/",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip go test and golangci-lint (faster, but no quality check).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging.",
    ),
) -> None:
    """
    Run the Agentic Go Contributor on a GitHub issue.

    The agent will:
    1. Read the issue
    2. Clone and analyse the repository
    3. Retrieve relevant files
    4. Generate an implementation plan
    5. Write code changes
    6. Validate (go test + golangci-lint)
    7. Repair failures (up to 3 iterations)
    8. Generate a PR title and description
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    _print_banner()

    # ── Validate API key ──────────────────────────────────────────────────────
    if not settings.openrouter_api_key or settings.openrouter_api_key.startswith("your_"):
        console.print(
            "[bold red]ERROR:[/bold red] OPENROUTER_API_KEY is not set.\n"
            "Copy .env.example → .env and fill in your key."
        )
        raise typer.Exit(code=1)

    # ── Override output dir if provided ──────────────────────────────────────
    if output_dir:
        settings.output_dir = output_dir

    # ── Build initial state ───────────────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_state: dict = {
        "issue_url": issue,
        "repo_path": repo or "",   # Repository Agent will infer if empty
        "logs": [f"Run started at {datetime.now().isoformat()}"],
        "validation_passed": False,
        "retrieved_files": [],
        "proposed_changes": {},
        "repository_map": {},
    }

    # ── Import workflow here to avoid circular imports ────────────────────────
    from graph.workflow import get_workflow

    workflow = get_workflow()

    # ── Run the workflow ──────────────────────────────────────────────────────
    console.print(f"\n[bold]Starting run:[/bold] {run_id}")
    console.print(f"[dim]Issue: {issue}[/dim]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Running agents…", total=None)

            final_state = workflow.invoke(
                initial_state,
                config={"recursion_limit": 50},
            )
            progress.update(task, description="[green]Complete!")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=1)
    except Exception as exc:
        logger.exception("Workflow failed")
        console.print(f"\n[bold red]Workflow error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir = _write_outputs(final_state, run_id)
    console.print(f"\n[bold green]✅ Run complete![/bold green] Outputs written to: {out_dir}")

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(final_state, out_dir)

    # Exit with non-zero if validation failed (useful for CI)
    if not final_state.get("validation_passed", False):
        console.print(
            "\n[yellow]⚠️  Validation did not pass. Review the outputs carefully.[/yellow]"
        )
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
