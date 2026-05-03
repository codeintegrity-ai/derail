"""Typer CLI surface for derail.

Two subcommands:

- ``validate`` — schema-only check on a scenario YAML; cheap CI hook.
- ``run``      — execute one or more scenarios in direct mode and print the
  comparison report.

The ``app`` callable is wired into ``pyproject.toml`` under
``[project.scripts]`` so that ``derail`` is on the user's PATH after
``uv sync``.
"""

from __future__ import annotations

import typer

from derail.cli.run_cmd import run_command
from derail.cli.validate_cmd import validate_command

app = typer.Typer(
    name="derail",
    help=(
        "YAML-driven MCP scenario harness — run benchmarks against synthetic "
        "MCPs over HTTP streaming and produce a comparison report."
    ),
    no_args_is_help=True,
    add_completion=False,
)

app.command("validate")(validate_command)
app.command("run")(run_command)


__all__ = ["app"]
