from __future__ import annotations

import logging
from typing import Any

from derail.runtime.mcp_runner import MCPRunner
from derail.scenario.loader import SeedStep

logger = logging.getLogger(__name__)


async def run_seed(
    runner: MCPRunner,
    seed_steps: list[SeedStep],
    bindings: dict[str, Any],
) -> dict[str, Any]:

    for step in seed_steps:
        server, tool = _split_qualified_tool(step.tool)
        rendered = render_bindings(step.args, bindings)
        try:
            result = await runner.call_tool(server, tool, rendered)
        except Exception as exc:
            raise SeedExecutionError(f"Seed step {step.tool!r} failed: {exc}") from exc

        if step.capture_id_as:
            captured = extract_captured_id(result)
            if captured is None:
                raise SeedExecutionError(
                    f"Seed step {step.tool!r} declared capture_id_as="
                    f"{step.capture_id_as!r} but the tool result contained no id. "
                    f"Tool result: {_describe_result(result)}"
                )
            bindings[step.capture_id_as] = captured
    return bindings


async def run_cleanup(
    runner: MCPRunner,
    cleanup_steps: list[SeedStep],
    bindings: dict[str, Any],
) -> None:

    for step in cleanup_steps:
        try:
            server, tool = _split_qualified_tool(step.tool)
            rendered = render_bindings(step.args, bindings)
            await runner.call_tool(server, tool, rendered)
        except Exception as exc:
            logger.warning("Cleanup step %r failed: %s — continuing.", step.tool, exc)


# ---- helpers --------------------------------------------------------


class SeedExecutionError(RuntimeError):
    """Seed step failed to run or declared a capture that didn't resolve."""


def _split_qualified_tool(qualified: str) -> tuple[str, str]:
    if "." not in qualified:
        raise ValueError(f"Tool {qualified!r} must be qualified as '<mcp>.<tool_name>'.")
    server, _, tool = qualified.partition(".")
    if not server or not tool:
        raise ValueError(f"Tool {qualified!r} has an empty server or tool name.")
    return server, tool


def render_bindings(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _render_string(value, bindings)
    if isinstance(value, dict):
        return {k: render_bindings(v, bindings) for k, v in value.items()}
    if isinstance(value, list):
        return [render_bindings(v, bindings) for v in value]
    if isinstance(value, tuple):
        return tuple(render_bindings(v, bindings) for v in value)
    return value


def _render_string(value: str, bindings: dict[str, Any]) -> str:
    """``str.format_map`` with a dict that leaves unknown keys intact."""
    try:
        return value.format_map(_SafeBindings(bindings))
    except ValueError:
        # format_map chokes on unbalanced braces — return the string unchanged.
        return value


class _SafeBindings(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _describe_result(result: Any) -> str:
    is_error = getattr(result, "isError", False)
    parts = [f"isError={is_error}"]
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(f"text={text[:200]!r}")
            break
    return " ".join(parts)


def extract_captured_id(result: Any) -> Any:
    captured = _id_from_dict(getattr(result, "structuredContent", None))
    if captured is not None:
        return captured

    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            import json as _json

            parsed = _json.loads(text)
        except (ValueError, TypeError):
            continue
        captured = _id_from_dict(parsed)
        if captured is not None:
            return captured

    return None


def _id_from_dict(node: Any) -> Any:
    if not isinstance(node, dict):
        return None
    if "id" in node:
        return node["id"]
    for value in node.values():
        if isinstance(value, dict) and "id" in value:
            return value["id"]
    return None
