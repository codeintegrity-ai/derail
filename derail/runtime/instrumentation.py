"""Runtime patches to mcp-agent for OTEL capture and inline content classification.

This module installs two monkey-patches

1. **OTEL annotator wrappers** — mcp-agent's annotators record only
   ``result.isError`` and ``result.content.*.{type,text}``, dropping
   ``CallToolResult.meta`` and ``CallToolResult.structuredContent``. Both
   of those carry data our harness needs. We wrap mcp-agent's annotators
   so the dropped fields land on the span as JSON strings.

2. **Inline content classification** — ``AugmentedLLM.pre_tool_call`` and
   ``post_tool_call`` are mcp-agent's intended extension points for
   intercepting tool calls. We wrap them so an active classifier (set
   via ``classifier_context._active_classifier``) inspects outgoing
   args and incoming results. Block-or-allow only; failure is treated
   as allow with a warning.

There are two OTEL annotator entry points in mcp-agent and both must be
patched:

1. ``annotate_span_for_call_tool_result`` — a module-level function
   defined in ``mcp_agent.tracing.telemetry`` and **imported by name**
   into several call sites. ``from x import y`` captures the binding at
   import time, so patching just the source module has no effect on call
   sites that already imported. We rebind the name in every importing
   module.

2. ``AugmentedLLM._annotate_span_for_call_tool_result`` — a method on
   the LLM base class, called by every LLM subclass during the agent
   loop. Patching the class method via attribute assignment works
   because subclass lookups resolve via MRO at call time.

"""

from __future__ import annotations

import importlib
import json
from contextlib import suppress
from threading import Lock
from typing import Any

from mcp_agent.workflows.llm import augmented_llm as _augmented_llm

from derail.runtime import classifier_context as _classifier_context


# Concrete LLM subclasses define their own ``pre_tool_call``/``post_tool_call``
# overrides (no-op pass-throughs upstream). Patching only ``AugmentedLLM`` is
# dead code for those subclasses because Python's MRO resolves the subclass
# method first. We collect every subclass that has an override and patch it
# alongside the base class.
def _llm_classes_with_tool_call_overrides() -> list[type]:
    """Concrete LLM classes that shadow ``AugmentedLLM.pre/post_tool_call``.
    """
    classes: list[type] = [_augmented_llm.AugmentedLLM]
    for module_name, attr in (
        ("mcp_agent.workflows.llm.augmented_llm_openai", "OpenAIAugmentedLLM"),
        ("mcp_agent.workflows.llm.augmented_llm_anthropic", "AnthropicAugmentedLLM"),
        ("mcp_agent.workflows.llm.augmented_llm_google", "GoogleAugmentedLLM"),
        ("mcp_agent.workflows.llm.augmented_llm_bedrock", "BedrockAugmentedLLM"),
        ("derail.runtime.openai_responses_llm", "OpenAIResponsesAugmentedLLM"),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, attr, None)
        if cls is None or cls in classes:
            continue
        # Only patch classes that DECLARE the method themselves — patching a
        # class that just inherits would override the inherited slot with our
        # bound function, which is fine but unnecessary clutter.
        if "pre_tool_call" in cls.__dict__ or "post_tool_call" in cls.__dict__:
            classes.append(cls)
    return classes


# Modules that ``from mcp_agent.tracing.telemetry import
# annotate_span_for_call_tool_result``. The function is captured at import
# time, so patching the source module isn't enough — we have to rebind the
# attribute on each of these too.
_FUNCTION_IMPORT_SITES: tuple[str, ...] = (
    "mcp_agent.tracing.telemetry",
    "mcp_agent.agents.agent",
    "mcp_agent.mcp.mcp_aggregator",
    "mcp_agent.workflows.llm.augmented_llm",
    "mcp_agent.workflows.llm.augmented_llm_openai",
)

_FUNCTION_NAME = "annotate_span_for_call_tool_result"

_originals: dict[str, Any] = {}
_original_method: Any = None
_install_count = 0
_lock = Lock()

# Saved references to mcp-agent's original ``AugmentedLLM.pre_tool_call``
# and ``post_tool_call`` methods. ``None`` means "not installed".
_original_pre_tool_call: Any = None
_original_post_tool_call: Any = None

# Per-class originals so uninstall can restore each subclass to whatever it
# had before — `dict.get(name)` returns ``None`` if the class only inherited
# the method, in which case uninstall just deletes our injected attribute.
_original_subclass_pre: dict[type, Any] = {}
_original_subclass_post: dict[type, Any] = {}


def install() -> None:
    """Wrap mcp-agent's annotators + classifier hooks.

    Reference-counted so nested ``MCPRunner`` contexts share a single
    install. The first call patches; later calls just bump the counter.
    """
    global \
        _original_method, \
        _original_pre_tool_call, \
        _original_post_tool_call, \
        _install_count
    with _lock:
        if _install_count == 0:
            # Patch the module-level function in every importer's namespace.
            for module_name in _FUNCTION_IMPORT_SITES:
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    continue
                if not hasattr(module, _FUNCTION_NAME):
                    continue
                _originals[module_name] = getattr(module, _FUNCTION_NAME)
                setattr(module, _FUNCTION_NAME, _patched_module_annotator)

            # Patch the AugmentedLLM method too (separate code path).
            _original_method = _augmented_llm.AugmentedLLM._annotate_span_for_call_tool_result
            _augmented_llm.AugmentedLLM._annotate_span_for_call_tool_result = _patched_method_annotator

            # Wrap pre/post tool-call hooks for inline classifier mode.
            # Without a classifier active in the contextvar, the wrappers
            # delegate to the original methods (no-op pass-through). We
            # patch every concrete LLM subclass that defines its own
            # override — patching only the base class is dead code because
            # Python's MRO resolves the subclass first.
            _original_pre_tool_call = _augmented_llm.AugmentedLLM.pre_tool_call
            _original_post_tool_call = _augmented_llm.AugmentedLLM.post_tool_call
            for cls in _llm_classes_with_tool_call_overrides():
                _original_subclass_pre[cls] = cls.__dict__.get("pre_tool_call")
                _original_subclass_post[cls] = cls.__dict__.get("post_tool_call")
                cls.pre_tool_call = _patched_pre_tool_call
                cls.post_tool_call = _patched_post_tool_call
        _install_count += 1


def uninstall() -> None:
    """Restore mcp-agent's originals once all installers have left."""
    global \
        _original_method, \
        _original_pre_tool_call, \
        _original_post_tool_call, \
        _install_count
    with _lock:
        if _install_count == 0:
            return
        _install_count -= 1
        if _install_count == 0:
            for module_name, original in list(_originals.items()):
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    continue
                setattr(module, _FUNCTION_NAME, original)
            _originals.clear()
            if _original_method is not None:
                _augmented_llm.AugmentedLLM._annotate_span_for_call_tool_result = _original_method
                _original_method = None
            if _original_pre_tool_call is not None:
                _augmented_llm.AugmentedLLM.pre_tool_call = _original_pre_tool_call
                _original_pre_tool_call = None
            if _original_post_tool_call is not None:
                _augmented_llm.AugmentedLLM.post_tool_call = _original_post_tool_call
                _original_post_tool_call = None
            # Restore per-subclass overrides. ``None`` means the class only
            # inherited the method; deleting our injected attribute lets MRO
            # fall back to the parent (now-restored) original.
            for cls, original in _original_subclass_pre.items():
                if cls is _augmented_llm.AugmentedLLM:
                    continue  # already restored above
                if original is None:
                    with suppress(AttributeError):
                        del cls.pre_tool_call
                else:
                    cls.pre_tool_call = original
            for cls, original in _original_subclass_post.items():
                if cls is _augmented_llm.AugmentedLLM:
                    continue
                if original is None:
                    with suppress(AttributeError):
                        del cls.post_tool_call
                else:
                    cls.post_tool_call = original
            _original_subclass_pre.clear()
            _original_subclass_post.clear()


def _patched_module_annotator(span, result) -> None:
    """Module-level annotator wrapper: original first, then record extras.

    Routes through the *current* attribute of telemetry rather than a
    captured reference so multiple installers stack cleanly.
    """
    original = _originals.get("mcp_agent.tracing.telemetry")
    if original is not None:
        original(span, result)
    _record_extras(span, result, prefix="result")


def _patched_method_annotator(self, span, result, processed: bool = False) -> None:
    """``AugmentedLLM`` method wrapper: original first, then record extras."""
    if _original_method is not None:
        _original_method(self, span, result, processed)
    if not getattr(self.context, "tracing_enabled", True):
        return
    prefix = "processed.result" if processed else "result"
    _record_extras(span, result, prefix=prefix)


def _record_extras(span, result, *, prefix: str) -> None:
    meta = getattr(result, "meta", None)
    if meta:
        with suppress(Exception):
            span.set_attribute(f"{prefix}.meta_json", json.dumps(_jsonify(meta)))
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        with suppress(Exception):
            span.set_attribute(f"{prefix}.structuredContent", json.dumps(_jsonify(structured)))


async def _patched_pre_tool_call(self, tool_call_id, request):
    """Wrap mcp-agent's pre_tool_call to consult an active classifier.

    With no classifier active, delegates straight to the original (which
    the user may have overridden in their own LLM subclass). With a
    classifier active and ``inspects_args=True``, runs it on the
    serialized args; if the verdict is unsafe, returns ``False`` and
    mcp-agent's ``call_tool`` short-circuits to a refusal result the
    agent sees in place of the call's actual outcome.
    """
    classifier = _classifier_context.get_active_classifier()
    if classifier is None:
        if _original_pre_tool_call is None:
            return request
        return await _original_pre_tool_call(self, tool_call_id, request)

    decision = await _classifier_context.apply_pre_tool_call(classifier, request)
    if decision is False:
        return False
    # Pass through to the upstream hook so any user-defined override on
    # the LLM subclass still runs after our classifier check.
    if _original_pre_tool_call is None:
        return decision
    return await _original_pre_tool_call(self, tool_call_id, decision)


async def _patched_post_tool_call(self, tool_call_id, request, result):
    """Wrap mcp-agent's post_tool_call to consult an active classifier.

    Order: original hook runs first (preserving any user-defined
    ``post_tool_call`` override), then classifier runs on whichever
    result the original returned. If the classifier flags it, we return
    a synthetic blocked ``CallToolResult`` and mcp-agent annotates the
    span with ``processed.result.*`` — which our metrics path then reads
    when scoring predicates.
    """
    if _original_post_tool_call is None:
        post_result = result
    else:
        post_result = await _original_post_tool_call(self, tool_call_id, request, result)

    classifier = _classifier_context.get_active_classifier()
    if classifier is None:
        return post_result

    override = await _classifier_context.apply_post_tool_call(classifier, request, post_result)
    if override is None:
        return post_result
    return override


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion to JSON-safe shapes.

    Handles pydantic models (``model_dump``), datetimes, and anything that
    ``json.dumps`` accepts natively. Unknown objects fall back to ``str`` so
    the attribute always serializes — better to lose precision on one leaf
    than to lose a full trace.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonify(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
