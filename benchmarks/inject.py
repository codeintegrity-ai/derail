from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from derail.scenario import Scenario, load_scenario

_DEFAULT_SEED_INDEX = 0
_DEFAULT_FIELD_NAME = "description"


def build_scenario_with_injection(
    base_scenario_path: str | Path,
    injection_text: str,
    *,
    seed_index: int | None = None,
    field_name: str | None = None,
) -> Scenario:

    scenario = load_scenario(Path(base_scenario_path))

    if seed_index is None:
        seed_index = scenario.inject.seed_index if scenario.inject else _DEFAULT_SEED_INDEX
    if field_name is None:
        field_name = scenario.inject.field_name if scenario.inject else _DEFAULT_FIELD_NAME

    if seed_index >= len(scenario.seed):
        raise IndexError(
            f"scenario {scenario.name!r} has {len(scenario.seed)} seed steps; "
            f"can't replace seed[{seed_index}].args[{field_name!r}]."
        )
    target = scenario.seed[seed_index]
    if field_name not in target.args:
        # Not strictly an error — the field gets created — but warn loudly
        # because it usually means the inject metadata is wrong and the
        # candidate text will land in a field the upstream tool ignores.
        import warnings

        warnings.warn(
            f"scenario {scenario.name!r}: inject target seed[{seed_index}].args[{field_name!r}] "
            f"does not exist in the seed step; existing arg keys are {sorted(target.args)}. "
            f"The candidate text will be added as a new field but the upstream tool may ignore it.",
            stacklevel=2,
        )
    new_args = {**target.args, field_name: injection_text}
    new_seed = list(scenario.seed)
    new_seed[seed_index] = replace(target, args=new_args)
    scenario.seed = new_seed
    return scenario
