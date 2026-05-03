"""Predicate evaluators + registry for the derail scenario DSL.

Three top-level predicates ship in Phase 1, all sharing the ``CallMatcher``
sub-DSL (``args.<field>_<op>``, ``where_any``, ``where_all``):

  - ``call_made``       — at least N calls match the selector.
  - ``no_call_made``    — no calls match the selector.
  - ``count_calls``     — call count satisfies a comparator (``op`` + ``value``).

Future phases can register additional predicates (LLM judges, span-tree
analyzers, etc.) via ``register_evaluator``.
"""

from __future__ import annotations

from typing import Any

from derail.evaluators.base import Evaluator, EvaluatorContext, SyncEvaluator
from derail.evaluators.call_made import CallMade
from derail.evaluators.count_calls import CountCalls
from derail.evaluators.no_call_made import NoCallMade
from derail.evaluators.shared import EvaluationRecord, EvaluatorResult

_EVALUATOR_REGISTRY: dict[str, type[Evaluator]] = {
    "call_made": CallMade,
    "no_call_made": NoCallMade,
    "count_calls": CountCalls,
}


def get_evaluator_by_name(name: str, config: dict[str, Any]) -> Evaluator:
    """Resolve a YAML predicate name + config to an Evaluator instance.

    Raises ``ValueError`` if the name isn't registered. Phase 1 keeps the
    surface small so unknown predicates fail loudly rather than silently
    returning ``None`` (mcp-eval's behaviour).
    """
    cls = _EVALUATOR_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown predicate {name!r}. Registered: {sorted(_EVALUATOR_REGISTRY)}.")
    return cls.from_dict(config)


def register_evaluator(name: str, evaluator_class: type[Evaluator]) -> None:
    _EVALUATOR_REGISTRY[name] = evaluator_class


def registered_predicates() -> list[str]:
    return sorted(_EVALUATOR_REGISTRY)


__all__ = [
    "CallMade",
    "CountCalls",
    "EvaluationRecord",
    "Evaluator",
    "EvaluatorContext",
    "EvaluatorResult",
    "NoCallMade",
    "SyncEvaluator",
    "get_evaluator_by_name",
    "register_evaluator",
    "registered_predicates",
]
