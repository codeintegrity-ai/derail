"""derail — YAML-driven MCP scenario harness.

Two pathways feed traces to the evaluator:

- ``process_spans`` (``metrics.py``): reduces OTEL JSONL spans into
  ``list[ToolCall]``. This is the production path — ``mcp-agent`` runs
  the agent against each upstream MCP, emits OTEL spans for each call,
  and this function reduces them to the canonical shape the evaluator
  consumes.
- ``TraceRecorder`` (``trace.py``): appends ``ToolCall`` records directly.
  Test-only path: unit tests construct traces by hand without any agent
  or MCP in the loop.

Both paths produce identical ``list[ToolCall]`` input for the evaluator.
"""

from derail.metrics import TestMetrics, ToolCall, TraceSpan, process_spans
from derail.runtime import (
    DirectModeConfig,
    MCPRunner,
    ScenarioRunResult,
    build_direct_settings,
    run_cleanup,
    run_scenario_direct,
    run_seed,
)
from derail.scenario.loader import (
    Scenario,
    ScenarioValidationError,
    load_scenario,
    load_scenario_from_string,
    validate_scenario_dict,
)
from derail.scenario.scoring import (
    EvaluationOutcome,
    evaluate_scenario,
    evaluate_scenario_async,
)
from derail.trace import TraceRecorder

__all__ = [
    "DirectModeConfig",
    "EvaluationOutcome",
    "MCPRunner",
    "Scenario",
    "ScenarioRunResult",
    "ScenarioValidationError",
    "TestMetrics",
    "ToolCall",
    "TraceRecorder",
    "TraceSpan",
    "build_direct_settings",
    "evaluate_scenario",
    "evaluate_scenario_async",
    "load_scenario",
    "load_scenario_from_string",
    "process_spans",
    "run_cleanup",
    "run_scenario_direct",
    "run_seed",
    "validate_scenario_dict",
]
