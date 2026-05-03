from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from mcp_agent.config import MCPServerSettings

from derail.classifiers import (
    Classifier,
    LLMJudgeClassifier,
)
from derail.runtime import DirectModeConfig

CONFIG_FILE_NAME = ".derail.yaml"

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class EndpointConfigError(ValueError):
    """Raised when the config file is missing, malformed, or incomplete."""


@dataclass
class EndpointConfig:
    """Resolved endpoint configuration ready to be turned into mode configs."""

    provider: str
    model: str
    direct_endpoints: dict[str, MCPServerSettings] = field(default_factory=dict)
    # Named classifiers from the top-level ``classifiers:`` block, ready
    # for the ``direct+<name>`` CLI dispatch. Built once at config-load
    # time so a misconfigured classifier (bad endpoint URL, unsupported
    # type) fails fast instead of crashing mid-scenario.
    classifiers: dict[str, Classifier] = field(default_factory=dict)

    def with_model(self, model: str) -> EndpointConfig:
        return replace(self, model=model)

    def build_direct_config(
        self,
        mcps: list[str],
        *,
        classifier_name: str | None = None,
    ) -> DirectModeConfig:
        missing = [m for m in mcps if m not in self.direct_endpoints]
        if missing:
            raise EndpointConfigError(
                f"direct mode: scenario references mcps without configured "
                f"endpoints: {missing}. Add them under modes.direct.endpoints."
            )
        classifier: Classifier | None = None
        if classifier_name is not None:
            if classifier_name not in self.classifiers:
                raise EndpointConfigError(
                    f"unknown classifier {classifier_name!r}. Define it under "
                    f"top-level ``classifiers:`` in the config. "
                    f"Known: {sorted(self.classifiers)}"
                )
            classifier = self.classifiers[classifier_name]
        return DirectModeConfig(
            endpoints=dict(self.direct_endpoints),
            provider=self.provider,
            model=self.model,
            classifier=classifier,
            classifier_name=classifier_name,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )


def load_endpoint_config(path: Path | None) -> EndpointConfig:
    """Load + resolve the endpoint config from ``path`` or by cwd walk-up."""
    resolved_path = path if path is not None else _find_config_file()
    if resolved_path is None:
        raise EndpointConfigError(
            f"no {CONFIG_FILE_NAME} found in cwd or any parent directory; "
            f"either create one or pass --config explicitly."
        )
    if not resolved_path.exists():
        raise EndpointConfigError(f"config file not found: {resolved_path}")

    raw = yaml.safe_load(resolved_path.read_text())
    if raw is None:
        raise EndpointConfigError(f"config file is empty: {resolved_path}")
    if not isinstance(raw, dict):
        raise EndpointConfigError(f"config file must be a YAML mapping at the top level: {resolved_path}")
    return _build_from_mapping(raw, source=resolved_path)


def _find_config_file(start: Path | None = None) -> Path | None:
    """Walk cwd → / looking for ``.derail.yaml``."""
    cur = (start or Path.cwd()).resolve()
    while True:
        candidate = cur / CONFIG_FILE_NAME
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            return None
        cur = cur.parent


def _build_from_mapping(raw: dict[str, Any], *, source: Path) -> EndpointConfig:
    raw = _interpolate(raw)

    defaults = raw.get("defaults") or {}
    provider = defaults.get("provider")
    model = defaults.get("model")
    if not provider:
        raise EndpointConfigError(f"{source}: defaults.provider is required (e.g. 'openai').")
    if not model:
        raise EndpointConfigError(f"{source}: defaults.model is required (e.g. 'gpt-4o-2024-11-20').")

    modes = raw.get("modes") or {}
    direct = modes.get("direct") or {}
    direct_endpoints_raw = direct.get("endpoints") or {}
    if not isinstance(direct_endpoints_raw, dict):
        raise EndpointConfigError(f"{source}: modes.direct.endpoints must be a mapping.")

    direct_endpoints = {
        str(k): _build_endpoint_settings(name=str(k), value=v, source=source)
        for k, v in direct_endpoints_raw.items()
    }

    classifiers = _build_classifiers(raw.get("classifiers"), source=source)

    return EndpointConfig(
        provider=str(provider),
        model=str(model),
        direct_endpoints=direct_endpoints,
        classifiers=classifiers,
    )


def _build_endpoint_settings(*, name: str, value: Any, source: Path) -> MCPServerSettings:
    """Normalize one endpoint entry into a fully-resolved MCPServerSettings.

    Endpoints are URL strings — synthetic MCPs over HTTP streaming, no auth.
    """
    if not isinstance(value, str):
        raise EndpointConfigError(
            f"{source}: modes.direct.endpoints.{name} must be a URL string; got {type(value).__name__}."
        )
    return MCPServerSettings(transport="streamable_http", url=value)


_CLASSIFIER_TYPES: dict[str, type[Classifier]] = {
    "llm_judge": LLMJudgeClassifier,
}


def _build_classifiers(raw: Any, *, source: Path) -> dict[str, Classifier]:
    """Resolve the top-level ``classifiers:`` block.

    Each named entry has a ``type`` field selecting one of the built-in
    classifier classes; remaining fields are passed verbatim to the
    class constructor. Construction errors fail fast so a misconfigured
    classifier (bad URL, missing model name) surfaces at config-load
    rather than mid-scenario.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise EndpointConfigError(f"{source}: top-level classifiers must be a mapping; got {type(raw).__name__}.")
    out: dict[str, Classifier] = {}
    for name, entry in raw.items():
        name_str = str(name)
        if not isinstance(entry, dict):
            raise EndpointConfigError(
                f"{source}: classifiers.{name_str} must be a mapping; got {type(entry).__name__}."
            )
        spec = dict(entry)
        type_key = spec.pop("type", None)
        if not isinstance(type_key, str):
            raise EndpointConfigError(
                f"{source}: classifiers.{name_str}.type is required (one of: {sorted(_CLASSIFIER_TYPES)})."
            )
        cls = _CLASSIFIER_TYPES.get(type_key)
        if cls is None:
            raise EndpointConfigError(
                f"{source}: classifiers.{name_str}.type={type_key!r} is not a "
                f"known classifier. Supported: {sorted(_CLASSIFIER_TYPES)}."
            )
        try:
            out[name_str] = cls(**spec)
        except Exception as exc:
            raise EndpointConfigError(
                f"{source}: classifiers.{name_str}: failed to construct {cls.__name__}: {exc}"
            ) from exc
    return out


def _interpolate(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` placeholders from os.environ.

    Missing env vars become empty strings — surface that as a config error
    rather than silently passing the literal placeholder onto a tool that
    would reject it cryptically downstream.
    """
    if isinstance(value, str):
        return _interpolate_string(value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _interpolate_string(value: str) -> str:
    def _sub(match: re.Match[str]) -> str:
        var = match.group(1)
        replacement = os.environ.get(var)
        if replacement is None:
            raise EndpointConfigError(f"environment variable {var} referenced in config but not set.")
        return replacement

    return _ENV_VAR_RE.sub(_sub, value)
