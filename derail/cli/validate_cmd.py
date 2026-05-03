"""``derail validate scenario.yaml [...]`` — schema check, no execution.

Wraps the existing ``load_scenario`` / ``ScenarioValidationError`` plumbing
so users can sanity-check scenarios before paying the cost of an LLM run.
Designed to be cheap enough to drop into pre-commit / CI.

Exits with status 0 if every file validates, 1 if any file fails. Errors
are rendered with rich formatting; the file path is always shown so it's
obvious which scenario broke when validating many at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from derail.scenario import ScenarioValidationError, load_scenario

_console = Console()
_error_console = Console(stderr=True, style="bold red")


def validate_command(
    scenarios: Annotated[
        list[Path],
        typer.Argument(
            help="One or more scenario YAML files to validate.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Validate one or more scenario YAML files against the schema."""
    failures = 0
    for path in scenarios:
        try:
            scenario = load_scenario(path)
        except ScenarioValidationError as exc:
            failures += 1
            _error_console.print(f"[red]✗[/red] {path}")
            for err in exc.errors:
                _error_console.print(f"    [red]•[/red] {err}")
            continue
        except Exception as exc:
            failures += 1
            _error_console.print(f"[red]✗[/red] {path}: {exc.__class__.__name__}: {exc}")
            continue
        _console.print(
            f"[green]✓[/green] {path} "
            f"[dim](mcps={','.join(scenario.mcps)}, "
            f"utility={len(scenario.utility)}, security={len(scenario.security)})[/dim]"
        )

    if failures:
        raise typer.Exit(code=1)
