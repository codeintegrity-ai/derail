from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field

from derail.evaluators import Evaluator, EvaluatorContext
from derail.metrics import TestMetrics, ToolCall
from derail.scenario.loader import Scenario


@dataclass
class PredicateOutcome:
    predicate: str
    passed: bool
    expected: object | None = None
    actual: object | None = None
    details: dict | None = None
    error: str | None = None


@dataclass
class EvaluationOutcome:
    scenario_name: str
    utility_results: list[PredicateOutcome] = field(default_factory=list)
    security_results: list[PredicateOutcome] = field(default_factory=list)
    model_output: str | None = None

    @property
    def utility_passed(self) -> bool:
        return all(r.passed for r in self.utility_results)

    @property
    def security_breached(self) -> bool:
        return any(r.passed for r in self.security_results)


def evaluate_scenario(
    scenario: Scenario,
    trace: Iterable[ToolCall],
    *,
    model_output: str | None = None,
) -> EvaluationOutcome:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop → sync caller → safe to spin one up.
        return asyncio.run(evaluate_scenario_async(scenario, list(trace), model_output=model_output))
    raise RuntimeError(
        "evaluate_scenario was called from a running event loop. Use `await evaluate_scenario_async(...)` instead."
    )


async def evaluate_scenario_async(
    scenario: Scenario,
    trace: Iterable[ToolCall],
    *,
    model_output: str | None = None,
) -> EvaluationOutcome:
    """Async evaluator entry point — the one to use from inside an event loop."""
    return await _evaluate_async(scenario, list(trace), model_output=model_output)


async def _evaluate_async(
    scenario: Scenario,
    trace: list[ToolCall],
    *,
    model_output: str | None,
) -> EvaluationOutcome:
    metrics = TestMetrics(tool_calls=trace)
    ctx: EvaluatorContext = EvaluatorContext(
        inputs=scenario.prompt,
        output=model_output,
        expected_output=None,
        metadata={"scenario": scenario.name},
        metrics=metrics,
    )
    return EvaluationOutcome(
        scenario_name=scenario.name,
        utility_results=[await _run(p, ctx) for p in scenario.utility],
        security_results=[await _run(p, ctx) for p in scenario.security],
        model_output=model_output,
    )


async def _run(predicate: Evaluator, ctx: EvaluatorContext) -> PredicateOutcome:
    try:
        result = await predicate.evaluate(ctx)
    except Exception as exc:  # pragma: no cover - safety net
        return PredicateOutcome(
            predicate=predicate.__class__.__name__,
            passed=False,
            error=f"{exc.__class__.__name__}: {exc}",
        )
    return PredicateOutcome(
        predicate=predicate.__class__.__name__,
        passed=result.passed,
        expected=result.expected,
        actual=result.actual,
        details=result.details,
        error=result.error,
    )
