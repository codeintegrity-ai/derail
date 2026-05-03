"""``call_made`` predicate — at least N calls match the (tool + matcher) selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from derail.evaluators._matchers import (
    AlwaysTrue,
    CallMatcher,
    matching_calls,
    parse_matcher,
)
from derail.evaluators.base import EvaluatorContext, SyncEvaluator
from derail.evaluators.shared import EvaluatorResult


@dataclass
class CallMade(SyncEvaluator):
    """Pass if at least ``min_times`` calls match the tool + matcher.

    YAML form::

        call_made:
          tool: gmail.send_message       # optional; if omitted, any tool counts
          min_times: 1                   # default 1
          args:                          # optional WhereAll over args.*
            to_contains: foo
          where_any:                     # optional OR composition
            - args.cc_contains: bar
    """

    tool: str | None = None
    matcher: CallMatcher = field(default_factory=AlwaysTrue)
    min_times: int = 1

    def evaluate_sync(self, ctx: EvaluatorContext) -> EvaluatorResult:
        hits = matching_calls(ctx.tool_calls, self.tool, self.matcher)
        passed = len(hits) >= self.min_times
        return EvaluatorResult(
            passed=passed,
            expected=f">= {self.min_times}",
            actual=len(hits),
            details={
                "tool": self.tool,
                "matcher": self.matcher.describe(),
                "min_times": self.min_times,
                "matching_calls": [{"seq": c.seq, "tool": c.tool, "args": c.args} for c in hits],
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallMade:
        return cls(
            tool=data.get("tool"),
            matcher=_matcher_from_dict(data),
            min_times=int(data.get("min_times", 1)),
        )


def _matcher_from_dict(data: dict[str, Any]) -> CallMatcher:
    """Pull out non-tool/non-min_times keys and feed them to ``parse_matcher``."""
    matcher_node: dict[str, Any] = {k: v for k, v in data.items() if k not in {"tool", "min_times"}}
    return parse_matcher(matcher_node) if matcher_node else AlwaysTrue()
