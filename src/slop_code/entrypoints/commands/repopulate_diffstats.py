"""Repopulate diff_diffstat.json files for checkpoint snapshots (CQB-commensurable)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from slop_code.common import SNAPSHOT_DIR_NAME
from slop_code.entrypoints.utils import discover_checkpoints
from slop_code.entrypoints.utils import discover_problems
from slop_code.execution.diffstat import git_diffstat_between_dirs
from slop_code.logging import get_logger

logger = get_logger(__name__)

DIFFSTAT_FILENAME = "diff_diffstat.json"


def register(app: typer.Typer, name: str):
    app.command(
        name,
        help="Repopulate diff_diffstat.json files (CQB-commensurable git+diffstat stats).",
    )(repopulate_diffstats)


def _process_problem(problem_dir: Path) -> dict[str, dict]:
    checkpoints = discover_checkpoints(problem_dir)
    results: dict[str, dict] = {}
    prev_snapshot_dir: Path | None = None

    for checkpoint_dir in checkpoints:
        checkpoint_name = checkpoint_dir.name
        snapshot_dir = checkpoint_dir / SNAPSHOT_DIR_NAME

        if not snapshot_dir.exists():
            logger.warning(
                "Snapshot directory not found, skipping checkpoint",
                checkpoint=checkpoint_name,
                problem=problem_dir.name,
            )
            prev_snapshot_dir = None
            continue

        result = git_diffstat_between_dirs(prev_snapshot_dir, snapshot_dir)

        out_path = checkpoint_dir / DIFFSTAT_FILENAME
        out_path.write_text(json.dumps(result.to_dict(), indent=2))

        logger.info(
            "Wrote diff_diffstat.json",
            checkpoint=checkpoint_name,
            problem=problem_dir.name,
            files_changed=result.files_changed,
            lines_changed=result.lines_changed,
        )

        results[checkpoint_name] = result.to_dict()
        prev_snapshot_dir = snapshot_dir

    return results


def repopulate_diffstats(
    run_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the run directory",
            exists=True,
            dir_okay=True,
            file_okay=False,
        ),
    ],
    problem_name: Annotated[
        str | None,
        typer.Option(
            "-p",
            "--problem-name",
            help="Filter to a specific problem name.",
        ),
    ] = None,
) -> None:
    """Regenerate diff_diffstat.json files for all checkpoints in a run.

    Uses `git diff --no-index -M --ignore-space-change | diffstat -tm` — the same
    flags as CQB's scoring pipeline — so churn numbers are commensurable across
    the two benchmarks.

    Requires `git` and `diffstat` on the host PATH.
    """
    console = Console()

    problems = discover_problems(run_dir)
    if not problems:
        console.print(f"[red]No problems found in {run_dir}[/red]")
        sys.exit(1)

    if problem_name is not None:
        problems = [p for p in problems if p.name == problem_name]
        if not problems:
            console.print(f"[red]Problem '{problem_name}' not found in {run_dir}[/red]")
            sys.exit(1)

    console.print(f"[green]Found {len(problems)} problem(s) in {run_dir}[/green]")

    all_results: dict[str, dict[str, dict]] = {}
    for problem_dir in problems:
        console.print(f"\n[bold]Processing: {problem_dir.name}[/bold]")
        all_results[problem_dir.name] = _process_problem(problem_dir)

    console.print("\n[bold]Summary[/bold]")
    table = Table(show_header=True, show_lines=True)
    table.add_column("Problem")
    table.add_column("Checkpoint")
    table.add_column("Files Changed")
    table.add_column("Lines Changed")

    for prob_name, checkpoints in all_results.items():
        for i, (cp_name, stats) in enumerate(checkpoints.items()):
            table.add_row(
                prob_name if i == 0 else "",
                cp_name,
                str(stats.get("files_changed", 0)),
                str(stats.get("lines_changed", 0)),
            )

    console.print(table)
