"""Pluggable classifiers for inline detector mode.

A classifier inspects a piece of text — either the JSON-serialized
arguments going *out* to a tool, or the response coming *back* from one
— and returns a verdict (safe / unsafe). The runtime intercepts MCP tool
calls via ``AugmentedLLM.pre_tool_call`` / ``post_tool_call`` (see
``derail.runtime.instrumentation``) and uses the verdict to either
block the call or let it through.

One reference classifier ships:

- ``LLMJudgeClassifier`` — asks a small LLM whether the content looks
  like a prompt-injection attempt or an agent-manipulation attack.

Failure-tolerance is uniform: any exception raised by a classifier
implementation is logged and treated as ``safe=True`` so a flaky
classifier never fails a scenario run.
"""

from derail.classifiers.base import (
    Classifier,
    ClassifierContext,
    ClassifierVerdict,
)
from derail.classifiers.llm_judge import LLMJudgeClassifier

__all__ = [
    "Classifier",
    "ClassifierContext",
    "ClassifierVerdict",
    "LLMJudgeClassifier",
]
