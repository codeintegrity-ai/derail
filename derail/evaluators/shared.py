"""Shared types for evaluators."""

from typing import Any

from pydantic import BaseModel, Field


class EvaluatorResult(BaseModel):
    """Standard return type for any evaluator."""

    passed: bool = Field(description="Whether the evaluation passed")
    expected: str | int | float | list[Any] | dict[str, Any] | None = Field(
        default=None, description="What was expected"
    )
    actual: str | int | float | list[Any] | dict[str, Any] | None = Field(
        default=None, description="What was actually received"
    )
    score: float | None = Field(default=None, ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    details: dict[str, Any] | None = Field(default=None, description="Additional context-specific information")
    error: str | None = Field(default=None, description="Error message if applicable")

    class Config:
        extra = "forbid"


class EvaluationRecord(BaseModel):
    """A named evaluation result, suitable for aggregation in reports."""

    name: str = Field(description="Name of the evaluator")
    result: EvaluatorResult = Field(description="The evaluation result")
    passed: bool = Field(description="Whether the evaluation passed")
    error: str | None = Field(default=None, description="Error message if applicable")

    class Config:
        extra = "forbid"
