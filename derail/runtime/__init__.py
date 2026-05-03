"""Runtime — live scenario execution against real MCP servers.

Builds on the data plane (metrics, evaluators, scenario loader) and adds the
pieces a real run needs:

- ``config``: turn a ``Scenario`` + endpoint map into an ``mcp-agent``
  ``Settings`` instance with OTEL routed to a JSONL file.
- ``mcp_runner.MCPRunner``: thin async-context wrapper around
  ``MCPApp`` + ``Agent`` + ``AugmentedLLM`` that exposes just what the
  orchestrator needs (direct tool calls for seed/cleanup, prompted agent
  turns for the user task, and a ``read_spans`` helper).
- ``scenario_run``: ``run_scenario_direct`` glues it all together into a
  single ``EvaluationOutcome``.
"""

from derail.runtime.config import (
    DirectModeConfig,
    build_direct_settings,
)
from derail.runtime.mcp_runner import MCPRunner
from derail.runtime.scenario_run import (
    ScenarioRunResult,
    run_scenario_direct,
)
from derail.runtime.seed_cleanup import (
    extract_captured_id,
    render_bindings,
    run_cleanup,
    run_seed,
)

__all__ = [
    "DirectModeConfig",
    "MCPRunner",
    "ScenarioRunResult",
    "build_direct_settings",
    "extract_captured_id",
    "render_bindings",
    "run_cleanup",
    "run_scenario_direct",
    "run_seed",
]
