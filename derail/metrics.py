from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def unflatten_attributes(attributes: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Reconstruct nested dicts/lists from flattened OTEL span attributes.

    Example: ``{"result.content.0.text": "hi"}`` becomes
    ``{"result": {"content": [{"text": "hi"}]}}``.
    """

    def _ensure_list_size(lst: list, index: int) -> None:
        if index >= len(lst):
            lst.extend([None] * (index + 1 - len(lst)))

    def _set_path(root: Any, parts: list[str], value: Any) -> Any:
        if not parts:
            return root

        key = parts[0]
        is_index = key.isdigit()
        idx = int(key) if is_index else None

        if len(parts) == 1:
            if is_index:
                if not isinstance(root, list):
                    root = []
                _ensure_list_size(root, idx)
                root[idx] = value
            else:
                if not isinstance(root, dict):
                    root = {}
                root[key] = value
            return root

        if is_index:
            if not isinstance(root, list):
                root = []
            _ensure_list_size(root, idx)
            next_val = root[idx]
            if next_val is None:
                next_val = [] if parts[1].isdigit() else {}
            root[idx] = _set_path(next_val, parts[1:], value)
            return root

        if not isinstance(root, dict):
            root = {}
        next_val = root.get(key)
        if next_val is None:
            next_val = [] if (len(parts) > 1 and parts[1].isdigit()) else {}
        root[key] = _set_path(next_val, parts[1:], value)
        return root

    result: dict[str, Any] = {}
    for full_key, value in attributes.items():
        if not full_key.startswith(prefix):
            continue
        path = full_key[len(prefix) :].split(".")
        result = _set_path(result, path, value)
    return result


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    result: Any
    start_time: float  # epoch seconds
    end_time: float  # epoch seconds
    seq: int = 0
    transport: str = ""
    mcp_server: str | None = None
    is_error: bool = False
    error_message: str | None = None


@dataclass
class TestMetrics:
    """Metrics derived from a recorded trace."""

    __test__ = False  # pytest: not a test class

    tool_calls: list[ToolCall] = field(default_factory=list)


# ---------- OTEL trace ingestion -------------------------------------------


@dataclass
class TraceSpan:
    """A single OTEL span loaded from a JSONL trace file."""

    name: str
    context: dict[str, str]
    parent: dict[str, str] | None
    start_time: int  # nanoseconds since epoch
    end_time: int
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_line: str) -> TraceSpan:
        data = json.loads(json_line)

        def parse_timestamp(ts: Any) -> int:
            if isinstance(ts, int | float):
                return int(ts)
            if isinstance(ts, str):
                if "T" in ts and ts.endswith("Z"):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    return int(dt.timestamp() * 1e9)
                return int(float(ts))
            return 0

        return cls(
            name=data.get("name", ""),
            context=data.get("context", {}),
            parent=data.get("parent"),
            start_time=parse_timestamp(data.get("start_time", 0)),
            end_time=parse_timestamp(data.get("end_time", 0)),
            attributes=data.get("attributes", {}),
            events=data.get("events", []),
        )


def process_spans(spans: list[TraceSpan]) -> TestMetrics:
    """Reduce raw OTEL spans into a TestMetrics summary."""
    metrics = TestMetrics()
    if not spans:
        return metrics

    tool_calls: list[ToolCall] = []
    for seq, span in enumerate(_sorted_tool_spans(spans), start=1):
        call = _extract_tool_call(span, seq)
        if call is not None:
            tool_calls.append(call)
    metrics.tool_calls = tool_calls
    return metrics


def _sorted_tool_spans(spans: list[TraceSpan]) -> list[TraceSpan]:
    return sorted(
        (s for s in spans if _is_tool_call_span(s)),
        key=lambda s: s.start_time,
    )


def _is_tool_call_span(span: TraceSpan) -> bool:
    if span.name == "MCPAggregator.call_tool":
        return (
            span.attributes.get("gen_ai.tool.name") is not None or span.attributes.get("parsed_tool_name") is not None
        )
    return False


def _extract_tool_call(span: TraceSpan, seq: int) -> ToolCall | None:
    try:
        tool_name = span.attributes.get("parsed_tool_name")
        server_name = span.attributes.get("parsed_server_name")
        if not tool_name:
            tool_name = span.attributes.get("mcp.tool.name")
        if not tool_name:
            gen_ai_tool_name = span.attributes.get("gen_ai.tool.name")
            if gen_ai_tool_name and "_" in gen_ai_tool_name:
                if server_name and gen_ai_tool_name.startswith(server_name + "_"):
                    tool_name = gen_ai_tool_name[len(server_name) + 1 :]
                else:
                    parts = gen_ai_tool_name.split("_", 1)
                    tool_name = parts[1] if len(parts) > 1 else gen_ai_tool_name
                    if not server_name and len(parts) > 1:
                        server_name = parts[0]
            elif gen_ai_tool_name:
                tool_name = gen_ai_tool_name

        arguments = unflatten_attributes(span.attributes, "arguments.")

        # When mcp-agent's ``post_tool_call`` (or our classifier wrapper)
        # returned a replacement ``CallToolResult``, mcp-agent annotates
        # the span with a parallel ``processed.result.*`` set of
        # attributes. The agent saw that processed view, so predicates
        # should score against it. Fall back to the raw result when no
        # processed view is present.
        has_processed = any(key.startswith("processed.result.") for key in span.attributes)
        if has_processed:
            result = _extract_result_view(span, prefix="processed.result")
            is_error = bool(
                span.attributes.get(
                    "processed.result.isError",
                    span.attributes.get("result.isError", False),
                )
            )
        else:
            result = _extract_result_view(span, prefix="result")
            is_error = bool(span.attributes.get("result.isError", False))

        return ToolCall(
            tool=tool_name or "",
            args=arguments,
            result=result,
            start_time=span.start_time / 1e9,
            end_time=span.end_time / 1e9,
            seq=seq,
            transport="otel",
            mcp_server=server_name,
            is_error=is_error,
            error_message=span.attributes.get("error.message"),
        )
    except Exception:
        return None


def _extract_result_view(span: TraceSpan, *, prefix: str) -> dict[str, Any]:
    """Build a result dict from span attributes under ``prefix``.

    Used twice: once with ``"result"`` for the raw view, once with
    ``"processed.result"`` when a post-hook override happened. Pulls
    structuredContent + meta_json from their JSON-string side-attributes
    so the dict reflects the full ``CallToolResult`` shape downstream
    predicates may expect.
    """
    view = unflatten_attributes(span.attributes, f"{prefix}.")
    structured = span.attributes.get(f"{prefix}.structuredContent")
    if structured and isinstance(structured, str):
        try:
            view["structuredContent"] = json.loads(structured)
        except (json.JSONDecodeError, TypeError):
            view["structuredContent"] = structured
    meta_json = span.attributes.get(f"{prefix}.meta_json")
    if meta_json and isinstance(meta_json, str):
        try:
            view["meta"] = json.loads(meta_json)
        except (json.JSONDecodeError, TypeError):
            view["meta"] = meta_json
    return view
