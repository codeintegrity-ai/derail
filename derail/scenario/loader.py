"""Parse + validate scenario YAML files into ``Scenario`` objects."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from derail.evaluators import (
    Evaluator,
    get_evaluator_by_name,
    registered_predicates,
)
from derail.scenario.schema import SCENARIO_JSON_SCHEMA

_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class ScenarioValidationError(ValueError):
    """Raised when a scenario YAML fails schema or semantic validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Scenario validation failed:\n" + "\n".join(f"- {e}" for e in errors))


@dataclass(frozen=True)
class SeedStep:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    capture_id_as: str | None = None


@dataclass(frozen=True)
class InjectTarget:
    """Where evolution-driven runs splice the candidate text into a scenario.

    Used by ``evolution/inject.py`` to know which seed step's which arg
    field to overwrite per iteration. Optional in the YAML; absent means
    fall back to the legacy default of ``seed[0].args["description"]``.
    """

    seed_index: int
    field_name: str


@dataclass
class Scenario:
    name: str
    mcps: list[str]
    prompt: str
    utility: list[Evaluator]
    security: list[Evaluator]
    seed: list[SeedStep] = field(default_factory=list)
    cleanup: list[SeedStep] = field(default_factory=list)
    description: str | None = None
    inject: InjectTarget | None = None
    attacker_goal: str | None = None

    @property
    def utility_predicates_yaml(self) -> list[dict[str, Any]]:
        return [{e.__class__.__name__: e.to_dict()} for e in self.utility]


def validate_scenario_dict(data: dict[str, Any]) -> None:
    """Schema-only validation. Does not instantiate predicates.

    Use this for the cheap CI hook (``derail validate scenario.yaml``);
    use ``load_scenario`` when you also want to instantiate the predicate tree.
    """
    validator = Draft202012Validator(SCENARIO_JSON_SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        raise ScenarioValidationError([_format_jsonschema_error(e) for e in errors])

    # Cross-check predicate names against the live registry. This catches the
    # case where someone uses a name that's syntactically valid per the schema
    # but not actually registered (e.g. a planned-but-unimplemented predicate).
    allowed = set(registered_predicates())
    semantic_errors: list[str] = []
    for section in ("utility", "security"):
        for idx, predicate in enumerate(data.get(section, [])):
            if not isinstance(predicate, dict) or len(predicate) != 1:
                continue  # caught by schema
            (name,) = predicate.keys()
            if name not in allowed:
                semantic_errors.append(
                    f"{section}[{idx}]: predicate {name!r} is not registered (allowed: {sorted(allowed)})"
                )
    if semantic_errors:
        raise ScenarioValidationError(semantic_errors)


def _format_jsonschema_error(err) -> str:
    path = ".".join(str(p) for p in err.absolute_path) or "<root>"
    return f"{path}: {err.message}"


def load_scenario(path: str | Path) -> Scenario:
    """Load and fully instantiate a scenario from a YAML file path."""
    text = Path(path).read_text()
    return load_scenario_from_string(text)


def load_scenario_from_string(text: str) -> Scenario:
    """Same as ``load_scenario`` but takes the raw YAML text."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ScenarioValidationError(["<root>: scenario file must parse to a YAML mapping"])
    data = _interpolate_env(data)
    validate_scenario_dict(data)
    return Scenario(
        name=data["name"],
        description=data.get("description"),
        mcps=list(data["mcps"]),
        prompt=data["prompt"],
        utility=[_build_predicate(p) for p in data["utility"]],
        security=[_build_predicate(p) for p in data["security"]],
        seed=[_build_seed(s) for s in data.get("seed") or []],
        cleanup=[_build_seed(s) for s in data.get("cleanup") or []],
        inject=_build_inject(data.get("inject")),
        attacker_goal=data.get("attacker_goal"),
    )


def _build_predicate(node: dict[str, Any]) -> Evaluator:
    (name, config) = next(iter(node.items()))
    return get_evaluator_by_name(name, config or {})


def _build_seed(node: dict[str, Any]) -> SeedStep:
    return SeedStep(
        tool=node["tool"],
        args=dict(node.get("args") or {}),
        capture_id_as=node.get("capture_id_as"),
    )


def _build_inject(node: dict[str, Any] | None) -> InjectTarget | None:
    if node is None:
        return None
    return InjectTarget(
        seed_index=int(node["seed_index"]),
        field_name=str(node["field_name"]),
    )


def _interpolate_env(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` placeholders from the environment.

    Mirrors the topology-config loader's behavior so scenarios can pull
    workspace IDs / tokens from env vars instead of hard-coding them.
    Missing vars surface as a ``ScenarioValidationError`` rather than
    silently leaving the literal placeholder, which would otherwise be
    sent to the upstream MCP and rejected with a cryptic error.
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(_env_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _env_sub(match: re.Match[str]) -> str:
    var = match.group(1)
    replacement = os.environ.get(var)
    if replacement is None:
        raise ScenarioValidationError([f"environment variable {var} referenced in scenario but not set."])
    return replacement
