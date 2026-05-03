"""``derail run scenario.yaml [...] [--mode direct[,direct+<classifier>]]`` ­— headline command.

Runs each scenario through each requested transport mode (direct, optionally
augmented with an inline classifier), collects the ``ScenarioRunResult`` per
(scenario, mode) pair, and prints the comparison table. Optionally writes a
JSONL report (one row per (scenario, mode)) for downstream tooling.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from derail.cli.config_file import (
    EndpointConfig,
    EndpointConfigError,
    load_endpoint_config,
)
from derail.cli.report import ModeRunResult, render_comparison, write_jsonl
from derail.runtime import (
    ScenarioRunResult,
    run_scenario_direct,
)
from derail.scenario import Scenario, ScenarioValidationError, load_scenario

_console = Console()
_error_console = Console(stderr=True, style="bold red")

_SUPPORTED_MODES = {"direct"}


def run_command(
    scenarios: Annotated[
        list[Path],
        typer.Argument(
            help="One or more scenario YAML files to run.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help=(
                "Comma-separated list of transport modes. Each mode is either "
                "``direct`` or ``direct+<classifier>`` where ``<classifier>`` "
                "names an entry from the config's ``classifiers:`` block."
            ),
        ),
    ] = "direct",
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=(
                "Path to an endpoint-config YAML. Defaults to ``.derail.yaml`` walked up from cwd to repo root."
            ),
            exists=False,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Override the default model from the config file.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=("Path to write the JSONL report. Default: ``runs/<scenario>-<timestamp>.jsonl``."),
        ),
    ] = None,
    work_dir: Annotated[
        Path,
        typer.Option(
            "--work-dir",
            help="Directory for per-run OTEL traces.",
        ),
    ] = Path("runs"),
) -> None:
    """Run scenarios across modes and print the comparison report."""
    requested_modes = _parse_modes(mode)

    try:
        endpoint_config = load_endpoint_config(config_path)
    except EndpointConfigError as exc:
        _error_console.print(f"[red]✗[/red] config error: {exc}")
        raise typer.Exit(code=2) from exc

    if model:
        endpoint_config = endpoint_config.with_model(model)

    parsed: list[Scenario] = []
    for path in scenarios:
        try:
            parsed.append(load_scenario(path))
        except ScenarioValidationError as exc:
            _error_console.print(f"[red]✗[/red] {path}")
            for err in exc.errors:
                _error_console.print(f"    [red]•[/red] {err}")
            raise typer.Exit(code=1) from exc

    asyncio.run(
        _run_all(
            scenarios=parsed,
            modes=requested_modes,
            endpoint_config=endpoint_config,
            work_dir=work_dir,
            output=output,
        )
    )


async def _run_all(
    *,
    scenarios: list[Scenario],
    modes: list[str],
    endpoint_config: EndpointConfig,
    work_dir: Path,
    output: Path | None,
) -> None:
    all_rows: list[ModeRunResult] = []
    for scenario in scenarios:
        rows: list[ModeRunResult] = []
        for mode in modes:
            _console.print(f"[bold]→[/bold] running [cyan]{scenario.name}[/cyan] in [magenta]{mode}[/magenta] mode")
            try:
                result = await _run_one(
                    scenario=scenario,
                    mode=mode,
                    endpoint_config=endpoint_config,
                    work_dir=work_dir,
                )
            except Exception as exc:
                _error_console.print(f"[red]✗[/red] {scenario.name} / {mode}: {exc.__class__.__name__}: {exc}")
                rows.append(
                    ModeRunResult(
                        scenario_name=scenario.name,
                        mode=mode,
                        result=None,
                        error=str(exc),
                    )
                )
                continue
            rows.append(
                ModeRunResult(
                    scenario_name=scenario.name,
                    mode=mode,
                    result=result,
                    error=None,
                )
            )
        _console.print(render_comparison(scenario.name, rows))
        all_rows.extend(rows)

    if output is None:
        timestamp = _timestamp()
        work_dir.mkdir(parents=True, exist_ok=True)
        output = work_dir / f"report-{timestamp}.jsonl"
    write_jsonl(output, all_rows)
    _console.print(f"[dim]Report written to {output}[/dim]")


async def _run_one(
    *,
    scenario: Scenario,
    mode: str,
    endpoint_config: EndpointConfig,
    work_dir: Path,
) -> ScenarioRunResult:
    base_mode, classifier_name = _split_mode(mode)
    if base_mode != "direct":
        raise ValueError(f"Unsupported mode {mode!r}")
    config = endpoint_config.build_direct_config(scenario.mcps, classifier_name=classifier_name)
    return await run_scenario_direct(scenario, config, work_dir)


def _split_mode(raw: str) -> tuple[str, str | None]:
    """Split ``direct+judge`` into ``("direct", "judge")``.

    Plain ``direct`` yields ``("direct", None)``. The parser is intentionally
    lenient on unknown classifier names — the name is validated when
    ``EndpointConfig.build_direct_config`` looks it up in the resolved
    classifiers dict.
    """
    if "+" not in raw:
        return raw, None
    base, _, classifier_name = raw.partition("+")
    if not classifier_name:
        raise typer.BadParameter(f"mode {raw!r} ends with ``+`` but no classifier name follows.")
    return base, classifier_name


def _parse_modes(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("--mode must list at least one transport.")
    unknown = []
    for part in parts:
        base, _ = _split_mode(part)
        if base not in _SUPPORTED_MODES:
            unknown.append(part)
    if unknown:
        raise typer.BadParameter(
            f"unknown modes: {unknown}. Supported base modes: "
            f"{sorted(_SUPPORTED_MODES)} (optionally with ``+<classifier>`` suffix)."
        )
    return parts


def _timestamp() -> str:
    """Sortable timestamp suffix for default report paths."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
