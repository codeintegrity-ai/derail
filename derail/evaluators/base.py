"""Base evaluator classes and the EvaluatorContext bag passed to each call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from derail.metrics import TestMetrics

if TYPE_CHECKING:
    from derail.evaluators.shared import EvaluatorResult


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


@dataclass
class EvaluatorContext(Generic[InputType, OutputType]):
    """Everything an evaluator might need to make a decision.

    ``inputs``: the prompt or input handed to the agent.
    ``output``: the assistant's final textual response (may be ``None``).
    ``metrics``: the trace summarized into a TestMetrics object.
    """

    inputs: InputType
    output: OutputType
    expected_output: OutputType | None
    metadata: dict[str, Any] | None
    metrics: TestMetrics

    @property
    def tool_calls(self):
        return self.metrics.tool_calls


class Evaluator(ABC, Generic[InputType, OutputType]):
    """Async evaluator base. Subclasses override ``evaluate``.

    ``requires_final_metrics`` defaults to True for predicates that read the
    full trace; immediate-only checks may set it False.
    """

    requires_final_metrics: bool = True

    @abstractmethod
    async def evaluate(self, ctx: EvaluatorContext[InputType, OutputType]) -> EvaluatorResult: ...

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evaluator:
        return cls(**data)


class SyncEvaluator(Evaluator[InputType, OutputType]):
    """Convenience base for predicates that don't need async."""

    async def evaluate(self, ctx: EvaluatorContext[InputType, OutputType]) -> EvaluatorResult:
        return self.evaluate_sync(ctx)

    @abstractmethod
    def evaluate_sync(self, ctx: EvaluatorContext[InputType, OutputType]) -> EvaluatorResult: ...
