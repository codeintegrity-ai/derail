"""Parse openevolve's per-run output directory into harness-friendly fields."""

from __future__ import annotations

import json
from pathlib import Path


def read_best(out_dir: Path) -> tuple[dict[str, float], int | None]:
    """Pull metrics from the best-program info file.

    Returns ``(metrics_dict, iteration_of_best)``. If openevolve produced
    no output, returns ``({}, None)``.
    """
    info_path = out_dir / "best" / "best_program_info.json"
    if not info_path.exists():
        return {}, None
    info = json.loads(info_path.read_text())
    metrics = info.get("metrics", {}) or {}
    iteration = info.get("iteration")
    return metrics, iteration


def scan_first_breach(out_dir: Path) -> int | None:
    """Walk per-iteration program records to find the earliest iteration
    whose metrics show ``security_breached >= 1.0``.

    Falls back to the latest checkpoint if the top-level programs.jsonl
    isn't available.
    """
    db_path = out_dir / "database" / "programs.jsonl"
    if not db_path.exists():
        checkpoints = sorted(
            (out_dir / "checkpoints").glob("checkpoint_*"),
            key=lambda p: int(p.name.split("_")[-1]),
        )
        if not checkpoints:
            return None
        db_path = checkpoints[-1] / "programs.jsonl"
        if not db_path.exists():
            return None
    earliest = None
    with db_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = row.get("metrics") or {}
            if metrics.get("security_breached", 0.0) >= 1.0:
                iter_idx = row.get("iteration")
                if iter_idx is not None and (earliest is None or iter_idx < earliest):
                    earliest = iter_idx
    return earliest
