"""Per-call matchers used by the top-level ``call_made`` / ``no_call_made``
/ ``count_calls`` predicates.

These are NOT registered as standalone scenario evaluators — they only appear
inside a top-level predicate's ``args:`` / ``where_any:`` / ``where_all:``
blocks. The shared parser in ``call_matcher.parse_matcher`` turns YAML
fragments into the concrete subclasses defined here.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from derail.metrics import ToolCall

# Operator suffixes recognised inside ``args:`` keys, longest-first so that
# e.g. "_not_in_allowlist" is matched before "_in_allowlist".
_OPERATORS: tuple[str, ...] = (
    "_not_in_allowlist",
    "_in_allowlist",
    "_contains",
    "_matches",
    "_equals",
)


class CallMatcher(ABC):
    """Sub-predicate evaluated against a single ``ToolCall``."""

    @abstractmethod
    def matches(self, call: ToolCall) -> bool: ...

    @abstractmethod
    def describe(self) -> str: ...


# ---------- Argument matchers ----------------------------------------------


def _resolve_arg(call: ToolCall, path: str) -> Any:
    """Walk ``call.args`` along a dotted path. Missing intermediate keys → None."""
    cur: Any = call.args
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _to_iterable(value: Any) -> Iterable[Any]:
    """Treat list/tuple values as already-iterable, scalars as single-element."""
    if isinstance(value, list | tuple | set):
        return value
    return (value,) if value is not None else ()


@dataclass
class ArgEquals(CallMatcher):
    path: str
    expected: Any

    def matches(self, call: ToolCall) -> bool:
        actual = _resolve_arg(call, self.path)
        if isinstance(actual, list | tuple | set):
            return self.expected in actual
        return actual == self.expected

    def describe(self) -> str:
        return f"args.{self.path} == {self.expected!r}"


@dataclass
class ArgContains(CallMatcher):
    """Case-insensitive substring match. Lists/tuples: any element contains it."""

    path: str
    needle: str

    def matches(self, call: ToolCall) -> bool:
        actual = _resolve_arg(call, self.path)
        needle = self.needle.lower()
        for item in _to_iterable(actual):
            if isinstance(item, str) and needle in item.lower():
                return True
        return False

    def describe(self) -> str:
        return f"args.{self.path} contains {self.needle!r}"


@dataclass
class ArgMatches(CallMatcher):
    """Regex match (re.search). Lists/tuples: any element matches."""

    path: str
    pattern: str

    def matches(self, call: ToolCall) -> bool:
        actual = _resolve_arg(call, self.path)
        compiled = re.compile(self.pattern)
        for item in _to_iterable(actual):
            if isinstance(item, str) and compiled.search(item):
                return True
        return False

    def describe(self) -> str:
        return f"args.{self.path} =~ /{self.pattern}/"


@dataclass
class ArgInAllowlist(CallMatcher):
    """Every element of the arg value must be in ``allowed``. Empty value passes."""

    path: str
    allowed: list[Any]

    def matches(self, call: ToolCall) -> bool:
        actual = _resolve_arg(call, self.path)
        items = list(_to_iterable(actual))
        if not items:
            return True
        allowed_set = {_lower_if_str(v) for v in self.allowed}
        return all(_lower_if_str(item) in allowed_set for item in items)

    def describe(self) -> str:
        return f"args.{self.path} in {self.allowed!r}"


@dataclass
class ArgNotInAllowlist(CallMatcher):
    """At least one element of the arg value lies outside ``allowed``.

    The natural exfil shape: ``recipients`` should be in the corp domain, and
    if any recipient isn't, the predicate fires. Empty arg value → no element
    is "outside" → False.
    """

    path: str
    allowed: list[Any]

    def matches(self, call: ToolCall) -> bool:
        actual = _resolve_arg(call, self.path)
        items = list(_to_iterable(actual))
        if not items:
            return False
        allowed_set = {_lower_if_str(v) for v in self.allowed}
        return any(_lower_if_str(item) not in allowed_set for item in items)

    def describe(self) -> str:
        return f"args.{self.path} not in {self.allowed!r}"


def _lower_if_str(v: Any) -> Any:
    return v.lower() if isinstance(v, str) else v


# ---------- Boolean composition --------------------------------------------


@dataclass
class WhereAny(CallMatcher):
    children: list[CallMatcher]

    def matches(self, call: ToolCall) -> bool:
        return any(child.matches(call) for child in self.children)

    def describe(self) -> str:
        inner = " OR ".join(c.describe() for c in self.children)
        return f"({inner})"


@dataclass
class WhereAll(CallMatcher):
    children: list[CallMatcher]

    def matches(self, call: ToolCall) -> bool:
        return all(child.matches(call) for child in self.children)

    def describe(self) -> str:
        inner = " AND ".join(c.describe() for c in self.children)
        return f"({inner})"


@dataclass
class AlwaysTrue(CallMatcher):
    """Used when a predicate has no per-call constraints (``call_made: {tool: X}``)."""

    def matches(self, call: ToolCall) -> bool:
        return True

    def describe(self) -> str:
        return "any"


# ---------- YAML → matcher parsing -----------------------------------------


def parse_matcher(node: Any) -> CallMatcher:
    """Convert a YAML fragment into a ``CallMatcher`` tree.

    Accepts:
    - ``None`` or empty dict → AlwaysTrue
    - ``{ <argname>_<op>: value, ... }`` → WhereAll of arg matchers
    - ``{ where_any: [...] }`` / ``{ where_all: [...] }`` → boolean composition
    - ``{ "args.<path>_<op>": value }`` → single arg matcher (path-syntax)
    """
    if node is None:
        return AlwaysTrue()

    if isinstance(node, list):
        return WhereAll([parse_matcher(item) for item in node])

    if not isinstance(node, dict):
        raise ValueError(f"Cannot parse matcher from {node!r} ({type(node).__name__})")

    if not node:
        return AlwaysTrue()

    children: list[CallMatcher] = []
    for key, value in node.items():
        children.append(_parse_keyed(key, value))
    return children[0] if len(children) == 1 else WhereAll(children)


def _parse_keyed(key: str, value: Any) -> CallMatcher:
    if key == "where_any":
        if not isinstance(value, list):
            raise ValueError(f"where_any must be a list, got {type(value).__name__}")
        return WhereAny([parse_matcher(item) for item in value])
    if key == "where_all":
        if not isinstance(value, list):
            raise ValueError(f"where_all must be a list, got {type(value).__name__}")
        return WhereAll([parse_matcher(item) for item in value])
    if key == "args":
        return _parse_args_block(value)

    if key.startswith("args."):
        return _build_arg_matcher(key[len("args.") :], value)

    raise ValueError(f"Unknown matcher key {key!r}; expected 'args', 'args.*_<op>', 'where_any', or 'where_all'.")


def _parse_args_block(value: Any) -> CallMatcher:
    if not isinstance(value, dict):
        raise ValueError(f"'args:' must be a mapping, got {type(value).__name__}")
    if not value:
        return AlwaysTrue()
    children = [_build_arg_matcher(k, v) for k, v in value.items()]
    return children[0] if len(children) == 1 else WhereAll(children)


def _build_arg_matcher(key: str, value: Any) -> CallMatcher:
    """Resolve ``<field>_<op>`` (operator suffix wins) into the right matcher."""
    for op in _OPERATORS:
        if key.endswith(op):
            field = key[: -len(op)]
            if not field:
                raise ValueError(f"Operator {op!r} requires a field name (got bare {key!r}).")
            return _make_op(field, op, value)
    # No suffix → exact equality
    return ArgEquals(path=key, expected=value)


def _make_op(field: str, op: str, value: Any) -> CallMatcher:
    if op == "_equals":
        return ArgEquals(path=field, expected=value)
    if op == "_contains":
        if not isinstance(value, str):
            raise ValueError(f"{field}_contains expects a string, got {type(value).__name__}")
        return ArgContains(path=field, needle=value)
    if op == "_matches":
        if not isinstance(value, str):
            raise ValueError(f"{field}_matches expects a regex string, got {type(value).__name__}")
        return ArgMatches(path=field, pattern=value)
    if op == "_in_allowlist":
        return ArgInAllowlist(path=field, allowed=_as_list(value, op))
    if op == "_not_in_allowlist":
        return ArgNotInAllowlist(path=field, allowed=_as_list(value, op))
    raise AssertionError(f"Unhandled op {op!r}")


def _as_list(value: Any, op: str) -> list[Any]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{op} expects a list, got {type(value).__name__}")
    return list(value)


def matching_calls(calls: Sequence[ToolCall], tool: str | None, matcher: CallMatcher) -> list[ToolCall]:
    """Return calls matching ``tool`` (if given) and ``matcher``."""
    selected = [c for c in calls if tool is None or c.tool == tool]
    return [c for c in selected if matcher.matches(c)]
