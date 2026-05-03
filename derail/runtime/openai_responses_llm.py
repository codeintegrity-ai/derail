from __future__ import annotations

import json
from typing import Any

from mcp.types import CallToolRequest, CallToolRequestParams
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from openai import AsyncOpenAI

_FUNCTION_CALL_TYPE = "function_call"
_MESSAGE_TYPE = "message"


class OpenAIResponsesAugmentedLLM(OpenAIAugmentedLLM):
    """Drop-in mcp-agent LLM that uses ``/v1/responses`` instead of chat completions."""

    async def generate_str(self, message, request_params=None) -> str:
        params = self.get_request_params(request_params)
        model = await self.select_model(params)
        instructions = self.instruction or params.systemPrompt or ""

        tools_listing = await self.agent.list_tools(tool_filter=params.tool_filter)
        tools_param = [_function_tool_param(t) for t in tools_listing.tools]

        input_items: list[dict[str, Any]] = _initial_input_items(message)

        # Tool loop. Each iteration calls /v1/responses; if the model emits
        # function_call items, we dispatch them and append the function_call
        # + function_call_output items into the next request's input list,
        # per the Responses API stateless-continuation pattern.
        last_text = ""
        for _ in range(params.max_iterations):
            response = await self._request_responses(
                model=model,
                instructions=instructions,
                input_items=input_items,
                tools=tools_param,
            )

            tool_calls: list[Any] = []
            for item in response.output or []:
                item_type = getattr(item, "type", None)
                if item_type == _FUNCTION_CALL_TYPE:
                    tool_calls.append(item)
                elif item_type == _MESSAGE_TYPE:
                    extracted = _extract_message_text(item)
                    if extracted:
                        last_text = extracted
                # Reasoning / refusal / other item types: deliberately ignored.

            if not tool_calls:
                break

            for tc in tool_calls:
                # Mirror the function_call back into input so the next
                # request's context includes it (Responses API requires the
                # call item to precede its output for matching by call_id).
                input_items.append(
                    {
                        "type": _FUNCTION_CALL_TYPE,
                        "call_id": tc.call_id,
                        "name": tc.name,
                        "arguments": tc.arguments or "",
                    }
                )

                args = _safe_parse_json_args(tc.arguments)
                tool_request = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name=tc.name, arguments=args),
                )
                # Goes through AugmentedLLM.call_tool → agent.call_tool →
                # MCPAggregator.call_tool. Same span shape as chat-completions
                # path; instrumentation.py captures Meta/structuredContent
                # off the result identically.
                tool_result = await self.call_tool(request=tool_request, tool_call_id=tc.call_id)

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": _render_tool_result_text(tool_result),
                    }
                )

        return last_text

    async def _request_responses(
        self,
        *,
        model: str,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ):
        config = self.context.config.openai
        async with AsyncOpenAI(
            api_key=getattr(config, "api_key", None),
            base_url=getattr(config, "base_url", None),
        ) as client:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": input_items,
            }
            if instructions:
                kwargs["instructions"] = instructions
            if tools:
                kwargs["tools"] = tools
            return await client.responses.create(**kwargs)

def _function_tool_param(tool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
        "strict": False,
    }


def _initial_input_items(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, str):
        return [{"role": "user", "content": message, "type": "message"}]
    if isinstance(message, list):
        return list(message)
    # Single dict / message-like object: wrap defensively.
    return [message] if message is not None else []


def _safe_parse_json_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_message_text(item) -> str:
    """Pull the assistant text out of a Responses-API ``message`` output item."""
    parts: list[str] = []
    for content in getattr(item, "content", None) or []:
        ctype = getattr(content, "type", None)
        text = getattr(content, "text", None)
        # Responses API uses "output_text"; older betas used "text". Accept both.
        if ctype in ("output_text", "text") and isinstance(text, str):
            parts.append(text)
    return "\n".join(p for p in parts if p)


def _render_tool_result_text(result) -> str:

    text_parts: list[str] = []
    for c in getattr(result, "content", None) or []:
        text = getattr(c, "text", None)
        if isinstance(text, str) and text:
            text_parts.append(text)
    if text_parts:
        return "\n".join(text_parts)

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        try:
            return json.dumps(structured)
        except (TypeError, ValueError):
            return str(structured)
    return ""
