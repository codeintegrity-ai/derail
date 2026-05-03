from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mcp_agent.agents.agent_spec import AgentSpec

from derail.evaluators._matchers import (
    ArgContains,
    ArgEquals,
    ArgInAllowlist,
    ArgMatches,
    ArgNotInAllowlist,
    CallMatcher,
    WhereAll,
    WhereAny,
)
from derail.metrics import ToolCall, process_spans
from derail.runtime.config import (
    DirectModeConfig,
    build_direct_settings,
)
from derail.runtime.mcp_runner import MCPRunner
from derail.runtime.seed_cleanup import (
    render_bindings,
    run_cleanup,
    run_seed,
)
from derail.scenario import EvaluationOutcome, Scenario, evaluate_scenario_async


@dataclass
class ScenarioRunResult:

    outcome: EvaluationOutcome
    trace: list[ToolCall]
    model_output: str | None
    otel_path: Path
    bindings: dict[str, Any] = field(default_factory=dict)


async def run_scenario_direct(
    scenario: Scenario,
    config: DirectModeConfig,
    work_dir: Path,
) -> ScenarioRunResult:

    work_dir.mkdir(parents=True, exist_ok=True)
    label_suffix = f"+{config.classifier_name}" if config.classifier_name else ""
    transport_label = f"direct{label_suffix}"
    otel_path = work_dir / f"{scenario.name}-{transport_label}.jsonl"
    _reset_trace_file(otel_path)

    settings = build_direct_settings(scenario, config, otel_path)
    agent_spec = AgentSpec(
        name=f"{scenario.name}-{transport_label}",
        instruction=config.system_prompt,
        server_names=list(scenario.mcps),
    )
    runner = MCPRunner(
        settings=settings,
        agent_spec=agent_spec,
        provider=config.provider,
        model=config.model,
        otel_path=otel_path,
        classifier=config.classifier,
        scenario_name=scenario.name,
    )
    return await _run_scenario_loop(scenario=scenario, runner=runner, transport_label=transport_label)


async def _run_scenario_loop(
    *,
    scenario: Scenario,
    runner: MCPRunner,
    transport_label: str,
) -> ScenarioRunResult:
    """Shared seed → prompt → evaluate → cleanup flow."""
    bindings: dict[str, Any] = {}
    response: str | None = None
    seed_call_count = 0
    async with runner:
        try:
            await run_seed(runner, scenario.seed, bindings)
            # Capture how many tool calls belong to seed so we can drop them
            # from the scored trace — only agent behaviour is evaluated.
            pre_agent_spans = runner.read_spans()
            seed_call_count = _count_tool_call_spans(pre_agent_spans)

            # Bindings captured by seed (e.g. ``issue_id``) are rendered into
            # the prompt AND into utility/security matcher needles, so a
            # predicate like ``args.issueId_contains: "{issue_id}"`` can match
            # against the runtime UUID rather than the literal placeholder.
            _render_predicate_bindings(scenario.utility, bindings)
            _render_predicate_bindings(scenario.security, bindings)
            prompt = render_bindings(scenario.prompt, bindings)
            response = await runner.run_prompt(prompt)
        finally:
            # Cleanup always runs, even if the agent crashed mid-loop.
            try:
                await run_cleanup(runner, scenario.cleanup, bindings)
            except Exception:  # pragma: no cover - soft-fail already inside
                pass

            spans = runner.read_spans()

    metrics = process_spans(spans)
    agent_calls = _drop_seed_calls(metrics.tool_calls, seed_call_count)
    agent_calls = [_stamp_transport(c, transport_label) for c in agent_calls]

    outcome = await evaluate_scenario_async(scenario, agent_calls, model_output=response)
    return ScenarioRunResult(
        outcome=outcome,
        trace=agent_calls,
        model_output=response,
        otel_path=runner.otel_path,
        bindings=bindings,
    )


def _reset_trace_file(otel_path: Path) -> None:
    """Starting fresh means process_spans sees only this run's spans."""
    if otel_path.exists():
        otel_path.unlink()


def _count_tool_call_spans(spans: list) -> int:
    """Count the MCPAggregator.call_tool spans in ``spans``.

    Mirrors ``metrics._is_tool_call_span`` without importing it (it's private).
    Good enough for seed-step bookkeeping.
    """
    return sum(
        1
        for s in spans
        if s.name == "MCPAggregator.call_tool"
        and (s.attributes.get("parsed_tool_name") or s.attributes.get("gen_ai.tool.name"))
    )


def _drop_seed_calls(trace: list[ToolCall], seed_count: int) -> list[ToolCall]:
    """Drop the first ``seed_count`` entries and re-number ``seq`` from 1.

    Uses ``dataclasses.replace`` so future ``ToolCall`` fields don't have
    to be re-listed here every time the dataclass grows.
    """
    remaining = trace[seed_count:] if seed_count > 0 else list(trace)
    return [replace(call, seq=new_seq) for new_seq, call in enumerate(remaining, start=1)]


def _stamp_transport(call: ToolCall, label: str) -> ToolCall:
    return replace(call, transport=label)


def _render_predicate_bindings(predicates: list, bindings: dict[str, Any]) -> None:
    """Substitute ``{name}`` placeholders into matcher needles in place.

    Predicates are built once at scenario-load time, so a matcher like
    ``args.issueId_contains: "{issue_id}"`` ends up with the literal
    string ``"{issue_id}"`` baked in. After seed runs, ``bindings``
    holds the actual captured ID — we walk each predicate's matcher
    tree and rewrite the needles so the comparison works.

    Mutates in place. ``Scenario`` is freshly loaded per run so this
    doesn't leak across runs.
    """
    if not bindings:
        return
    for evaluator in predicates:
        matcher = getattr(evaluator, "matcher", None)
        if isinstance(matcher, CallMatcher):
            _render_matcher(matcher, bindings)


def _render_matcher(matcher: CallMatcher, bindings: dict[str, Any]) -> None:
    """Mutate ``matcher`` in place: render bindings into its string fields."""
    if isinstance(matcher, ArgEquals):
        matcher.expected = _render_scalar(matcher.expected, bindings)
    elif isinstance(matcher, ArgContains):
        matcher.needle = render_bindings(matcher.needle, bindings)
    elif isinstance(matcher, ArgMatches):
        matcher.pattern = render_bindings(matcher.pattern, bindings)
    elif isinstance(matcher, ArgInAllowlist | ArgNotInAllowlist):
        matcher.allowed = [_render_scalar(v, bindings) for v in matcher.allowed]
    elif isinstance(matcher, WhereAny | WhereAll):
        for child in matcher.children:
            _render_matcher(child, bindings)


def _render_scalar(value: Any, bindings: dict[str, Any]) -> Any:
    """Render bindings into a value if it's a string; pass through otherwise."""
    return render_bindings(value, bindings) if isinstance(value, str) else value
