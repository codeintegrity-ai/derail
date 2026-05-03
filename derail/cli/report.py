"""Side-by-side comparison report — Rich table + JSONL writer.

Designed to handle N modes from day one (direct, gateway, future detector,
etc.) so adding a third mode is a no-op for the renderer.

Two outputs:

- ``render_comparison``: returns a Rich ``Table`` ready to print. Shape is
  the headline output the build plan promised:

      mail-inbox-exfil
                                utility    security (LOWER IS BETTER)
        direct                  passed     FAILED — exfil succeeded
        gateway                 passed     blocked

- ``write_jsonl``: machine-readable rollup, one JSON object per
  (scenario, mode). Carries enough detail for downstream tooling to
  diff runs across PRs without re-running them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.table import Table

from derail.runtime import ScenarioRunResult


@dataclass
class ModeRunResult:
    """One (scenario, mode) result, plus an optional infra-error string."""

    scenario_name: str
    mode: str
    result: ScenarioRunResult | None
    error: str | None = None


def render_comparison(scenario_name: str, rows: list[ModeRunResult]) -> Table:
    """Build the side-by-side table for one scenario across multiple modes."""
    table = Table(
        title=scenario_name,
        caption="security column: LOWER IS BETTER (a 'breached' result means the agent leaked)",
        title_style="bold cyan",
        caption_style="dim",
        show_lines=False,
    )
    table.add_column("mode", style="magenta", no_wrap=True)
    table.add_column("utility", justify="center")
    table.add_column("security", overflow="fold")

    for row in rows:
        if row.error is not None:
            table.add_row(row.mode, "[red]error[/red]", row.error)
            continue
        result = row.result
        assert result is not None  # the error branch above handles None
        utility_cell = "[green]passed[/green]" if result.outcome.utility_passed else "[red]failed[/red]"
        if result.outcome.security_breached:
            security_cell = "[red]breached[/red] — " + _summarize_breach(result)
        else:
            security_cell = "[green]blocked[/green]"
        table.add_row(row.mode, utility_cell, security_cell)
    return table


def write_jsonl(path: Path, rows: list[ModeRunResult]) -> None:
    """Write one JSON object per row to ``path``.

    Rows that errored out (``row.error`` set) are still recorded so a
    downstream consumer can tell that a mode failed vs. was never run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(_row_to_jsonable(row)) + "\n")


def _row_to_jsonable(row: ModeRunResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scenario": row.scenario_name,
        "mode": row.mode,
    }
    if row.error is not None:
        out["error"] = row.error
        return out
    result = row.result
    assert result is not None
    out.update(
        {
            "utility_passed": result.outcome.utility_passed,
            "security_breached": result.outcome.security_breached,
            "utility_results": [_evaluator_result_to_dict(r) for r in result.outcome.utility_results],
            "security_results": [_evaluator_result_to_dict(r) for r in result.outcome.security_results],
            "model_output": result.model_output,
            "otel_path": str(result.otel_path),
            "trace_length": len(result.trace),
        }
    )
    return out


def _evaluator_result_to_dict(r: Any) -> dict[str, Any]:
    """Convert one evaluator result to a JSON-safe dict.

    Predicate result objects expose at least ``passed`` and ``predicate``;
    additional fields are best-effort included via ``__dict__``.
    """
    if hasattr(r, "model_dump"):
        return r.model_dump(mode="json")
    return {k: v for k, v in getattr(r, "__dict__", {}).items() if not k.startswith("_")}


def _summarize_breach(result: ScenarioRunResult) -> str:
    """One-line description of what tripped the security predicate(s).

    Uses the first failing security predicate's description as a hint;
    falls back to a generic message. The full structured detail lives in
    the JSONL row, so this is just a UX nicety for the table.
    """
    for predicate_result in result.outcome.security_results:
        if getattr(predicate_result, "passed", False):
            description = (
                getattr(predicate_result, "predicate", None)
                or getattr(predicate_result, "name", None)
                or "exfil predicate matched"
            )
            return str(description)
    return "exfil predicate matched"
