"""``count_calls`` predicate — count of matching calls compared to a value."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from derail.evaluators._matchers import (
    AlwaysTrue,
    CallMatcher,
    matching_calls,
)
from derail.evaluators.base import EvaluatorContext, SyncEvaluator
from derail.evaluators.call_made import _matcher_from_dict
from derail.evaluators.shared import EvaluatorResult

_OPS: dict[str, callable] = {
    "==": lambda a, b: a == b,
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


@dataclass
class CountCalls(SyncEvaluator):
    """Compare the count of matching calls to ``value`` using ``op``.

    YAML form::

        count_calls:
          tool: director.execute_vos_command
          op: "<="
          value: 1
          args:
            command_matches: "^clear bgp neighbor .* soft$"
    """

    op: str = "=="
    value: int = 0
    tool: str | None = None
    matcher: CallMatcher = field(default_factory=AlwaysTrue)

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        if self.op not in _OPS:
            return EvaluatorResult(
                passed=False,
                expected=f"op in {sorted(_OPS)}",
                actual=self.op,
                error=f"Unknown comparator {self.op!r}",
            )
        hits = matching_calls(ctx.tool_calls, self.tool, self.matcher)
        actual = len(hits)
        passed = _OPS[self.op](actual, self.value)
        return EvaluatorResult(
            passed=passed,
            expected=f"count {self.op} {self.value}",
            actual=actual,
            details={
                "tool": self.tool,
                "matcher": self.matcher.describe(),
                "op": self.op,
                "value": self.value,
                "matching_calls": [{"seq": c.seq, "tool": c.tool, "args": c.args} for c in hits],
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CountCalls:
        if "op" not in data or "value" not in data:
            raise ValueError("count_calls requires 'op' and 'value'")
        return cls(
            op=str(data["op"]),
            value=int(data["value"]),
            tool=data.get("tool"),
            matcher=_matcher_from_dict({k: v for k, v in data.items() if k not in {"op", "value"}}),
        )
