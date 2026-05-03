"""JSON Schema for scenario YAML files.

Validates the *shape* of a scenario (required fields, types, allowed
predicate names). The set of allowed predicate names is derived from
``derail.evaluators.registered_predicates()`` at import time, so the
schema and the live registry can't drift.
"""

from __future__ import annotations

from derail.evaluators import registered_predicates

PREDICATE_NAMES: list[str] = registered_predicates()
_PREDICATE_PATTERN = "^(" + "|".join(PREDICATE_NAMES) + ")$"

SCENARIO_JSON_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "derail scenario",
    "type": "object",
    "required": ["name", "mcps", "prompt", "utility", "security"],
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[a-z0-9][a-z0-9_-]*$",
            "description": "Lowercase, kebab- or snake-case identifier.",
        },
        "description": {"type": "string"},
        "mcps": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
            "description": "Names of MCP servers from the gateway/direct catalog.",
        },
        "seed": {
            "type": "array",
            "items": {"$ref": "#/$defs/seedStep"},
        },
        "prompt": {"type": "string", "minLength": 1},
        "utility": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/predicate"},
        },
        "security": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/predicate"},
        },
        "cleanup": {
            "type": "array",
            "items": {"$ref": "#/$defs/seedStep"},
        },
        "inject": {
            "type": "object",
            "additionalProperties": False,
            "required": ["seed_index", "field_name"],
            "properties": {
                "seed_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Index into seed[] of the step whose args field gets the evolved injection text.",
                },
                "field_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Name of the field inside seed[seed_index].args to overwrite (e.g. 'description', 'body', 'content', 'bio').",
                },
            },
            "description": "Where evolution-driven runs (evolution/inject.py) should plant the candidate text. Optional; defaults to seed_index=0, field_name='description'.",
        },
        "attacker_goal": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Per-scenario description of what the attacker wants in the agent's "
                "output. Used by benchmarks/run_benchmark.py to build a scenario-specific "
                "openevolve mutator system message. Without this, every scenario gets the "
                "same generic URL-injection guidance from openevolve_base.yaml — which is "
                "wrong for non-URL-shaped scenarios. Optional; if absent, the runner falls "
                "back to the base openevolve config's system_message verbatim."
            ),
        },
    },
    "$defs": {
        "seedStep": {
            "type": "object",
            "required": ["tool"],
            "additionalProperties": False,
            "properties": {
                "tool": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_-]+\\.[a-zA-Z0-9_-]+$",
                    "description": "Qualified tool name <mcp>.<tool>",
                },
                "args": {"type": "object"},
                "capture_id_as": {
                    "type": "string",
                    "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
                },
            },
        },
        "predicate": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "patternProperties": {
                _PREDICATE_PATTERN: {"type": "object"},
            },
            "additionalProperties": False,
        },
    },
}
