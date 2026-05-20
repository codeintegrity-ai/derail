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

    OpenEvolve writes one JSON file per program under
    ``checkpoints/checkpoint_<N>/programs/<uuid>.json`` — not a single
    ``programs.jsonl``. Each checkpoint only contains the current
    population (~10 programs), so a single checkpoint misses anything
    selected out of the population before the run ended. We walk every
    checkpoint, dedupe by program ``id``, and take the minimum
    ``iteration_found`` over breach hits.
    """
    ckpt_root = out_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    seen_ids: set[str] = set()
    earliest: int | None = None
    for ckpt in sorted(
        ckpt_root.glob("checkpoint_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    ):
        programs_dir = ckpt / "programs"
        if not programs_dir.is_dir():
            continue
        for program_file in programs_dir.glob("*.json"):
            try:
                row = json.loads(program_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            pid = row.get("id")
            if pid in seen_ids:
                continue
            if pid is not None:
                seen_ids.add(pid)
            metrics = row.get("metrics") or {}
            if metrics.get("security_breached", 0.0) >= 1.0:
                iter_idx = row.get("iteration_found")
                if iter_idx is not None and (earliest is None or iter_idx < earliest):
                    earliest = iter_idx
    return earliest
