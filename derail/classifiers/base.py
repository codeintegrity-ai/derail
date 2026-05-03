from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class ClassifierVerdict:
    safe: bool
    reason: str
    score: float | None = None


@dataclass
class ClassifierContext:
    phase: Literal["pre_tool_call", "post_tool_call"]
    tool_name: str
    mcp_server: str | None = None
    scenario_name: str | None = None


class Classifier(ABC):
    inspects_args: bool = True
    inspects_results: bool = True

    @abstractmethod
    async def classify(
        self,
        content: str,
        *,
        context: ClassifierContext,
    ) -> ClassifierVerdict: ...
