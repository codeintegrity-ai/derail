from __future__ import annotations

import re
from collections.abc import Callable

from mcp_agent.agents.agent import Agent
from mcp_agent.core.context import Context
from mcp_agent.workflows.factory import _llm_factory as _mcp_agent_llm_factory
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

from derail.runtime.openai_responses_llm import OpenAIResponsesAugmentedLLM

_RESPONSES_API_MODEL_RE = re.compile(r"^(gpt-5|o[1-9])(?:[-\.]|$)")

def is_responses_api_model(model: str | None) -> bool:
    """Return True if ``model`` should be driven through ``/v1/responses``."""
    if not model:
        return False
    return _RESPONSES_API_MODEL_RE.match(model) is not None


def llm_factory(
    *,
    provider: str,
    model: str,
    context: Context,
) -> Callable[[Agent], AugmentedLLM]:
    """Return a callable ``agent -> AugmentedLLM`` for ``provider``/``model``.

    For OpenAI Responses-API models, returns a factory that builds our
    ``OpenAIResponsesAugmentedLLM`` adapter directly. For everything else,
    delegates to mcp-agent's own ``_llm_factory`` so all the existing
    provider-selection, model-preference, and request-param plumbing keeps
    working unchanged.
    """
    if provider == "openai" and is_responses_api_model(model):
        return lambda agent: OpenAIResponsesAugmentedLLM(
            agent=agent,
            default_model=model,
            context=context,
        )
    return _mcp_agent_llm_factory(
        provider=provider,
        model=model,
        context=context,
    )
