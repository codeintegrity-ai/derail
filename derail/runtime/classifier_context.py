from __future__ import annotations

import json
import warnings
from contextvars import ContextVar
from typing import Any

from mcp.types import CallToolRequest, CallToolResult, TextContent

from derail.classifiers.base import (
    Classifier,
    ClassifierContext,
    ClassifierVerdict,
)

_active_classifier: ContextVar[Classifier | None] = ContextVar("derail_active_classifier", default=None)

_active_scenario_name: ContextVar[str | None] = ContextVar("derail_active_scenario_name", default=None)


def get_active_classifier() -> Classifier | None:
    return _active_classifier.get()


def set_active_classifier(classifier: Classifier | None):
    """Returns the ContextVar token so the caller can reset on exit."""
    return _active_classifier.set(classifier)


def reset_active_classifier(token) -> None:
    _active_classifier.reset(token)


def set_active_scenario_name(name: str | None):
    return _active_scenario_name.set(name)


def reset_active_scenario_name(token) -> None:
    _active_scenario_name.reset(token)


# ---- pre / post adapters ---------------------------------------------------


async def apply_pre_tool_call(
    classifier: Classifier,
    request: CallToolRequest,
) -> CallToolRequest | bool:

    if not classifier.inspects_args:
        return request
    try:
        content = _serialize_args(request.params.arguments or {})
        ctx = _ctx_for("pre_tool_call", request.params.name)
        verdict = await classifier.classify(content, context=ctx)
    except Exception as exc:
        warnings.warn(
            f"classifier pre_tool_call adapter raised; treating as safe. tool={request.params.name!r} error={exc!r}",
            stacklevel=2,
        )
        return request
    if verdict.safe:
        return request
    return False


async def apply_post_tool_call(
    classifier: Classifier,
    request: CallToolRequest,
    result: CallToolResult,
) -> CallToolResult | None:

    if not classifier.inspects_results:
        return None
    try:
        content = _serialize_result(result)
        ctx = _ctx_for("post_tool_call", request.params.name)
        verdict = await classifier.classify(content, context=ctx)
    except Exception as exc:
        warnings.warn(
            f"classifier post_tool_call adapter raised; treating as safe. tool={request.params.name!r} error={exc!r}",
            stacklevel=2,
        )
        return None
    if verdict.safe:
        return None
    return _build_blocked_result(verdict)


# ---- internals -------------------------------------------------------------


def _ctx_for(phase: str, tool_name: str) -> ClassifierContext:
    return ClassifierContext(
        phase=phase,  # type: ignore[arg-type]
        tool_name=tool_name,
        mcp_server=None,
        scenario_name=_active_scenario_name.get(),
    )


def _serialize_args(args: dict[str, Any]) -> str:
    """Args → deterministic JSON. Stable for snapshot tests; the classifier
    sees the same byte sequence for the same args."""
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(args)


def _serialize_result(result: CallToolResult) -> str:
    parts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        try:
            parts.append(json.dumps(structured, sort_keys=True, default=str))
        except (TypeError, ValueError):
            parts.append(str(structured))
    return "\n".join(parts)


def _build_blocked_result(verdict: ClassifierVerdict) -> CallToolResult:
    text = f"Tool result was blocked by an inline content classifier. reason={verdict.reason!r}"
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent={
            "blocked": True,
            "reason": verdict.reason,
            "score": verdict.score,
        },
        isError=True,
    )
