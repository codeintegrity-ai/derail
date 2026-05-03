"""Sweep runner for the classifier-evasion benchmark.

Iterates over (scenario × judge_model × attacker_model × iter_budget) and
records the resulting metrics + iteration-to-first-breach into a CSV.

Each combination:
  1. Picks the right scenario YAML.
  2. Selects the right classifier name (e.g., judge_gpt4omini) so the
     harness wires that classifier into the agent's pre-tool / post-tool
     hooks via ``direct+<classifier>`` mode.
  3. Calls openevolve with the right attacker model via --primary-model.
  4. Reads the best-program metrics + iteration history from the
     openevolve output directory.
  5. Appends one row to ``--out``.

Usage::

    uv run python benchmarks/run_benchmark.py \\
      --scenarios "benchmarks/scenarios/*.yaml" \\
      --judges judge_gpt4omini,judge_gpt4o \\
      --attackers gpt-4o-mini,gpt-4o \\
      --iters 5,15,30 \\
      --out benchmarks/results.csv

Prereqs:
  - The synthetic Notion + Linear MCPs are running (mcps/*.py).
  - OPENAI_API_KEY is set.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.openevolve_config import materialize_config_with_goal
from benchmarks.result_parsing import read_best, scan_first_breach
from derail.scenario import load_scenario

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
_DEFAULT_SEED = _BENCH_DIR / "seeds" / "blatant_injection.txt"
_DEFAULT_CONFIG = _BENCH_DIR / "configs" / "openevolve_base.yaml"
_DEFAULT_EVALUATOR = _BENCH_DIR / "openevolve_evaluator.py"


def _split_csv(value: str) -> list[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _expand_globs(values: list[str]) -> list[Path]:
    out: list[Path] = []
    for v in values:
        if any(ch in v for ch in "*?["):
            matches = sorted(Path(p) for p in glob.glob(v))
            out.extend(matches)
        else:
            out.append(Path(v))
    return out


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def _resolve_seed(scenario_path: Path, override: Path | None) -> Path:
    """Pick the seed file to use for this scenario.

    If ``override`` is set (user passed ``--seed``), use it for every scenario.
    Otherwise auto-resolve: prefer ``benchmarks/seeds/<scenario_stem>.txt``
    if it exists; fall back to ``benchmarks/seeds/blatant_injection.txt``.

    Per-scenario seed files matter because the harness's ``inject.py``
    overwrites the targeted seed-step field wholesale with the candidate text.
    A seed that doesn't include the scenario's legitimate content leaves the
    agent with nothing to summarize when the judge passes the candidate, so
    utility drops to 0 and evolution has no gradient. Per-scenario seeds keep
    the legit content alongside the planted injection so evolution can mutate
    only the injection portion.
    """
    if override is not None:
        return override
    candidate = _BENCH_DIR / "seeds" / f"{scenario_path.stem}.txt"
    if candidate.exists():
        return candidate
    return _DEFAULT_SEED


def _run_one(
    scenario_path: Path,
    judge: str,
    attacker_model: str,
    iters: int,
    seed_path: Path,
    config_path: Path,
    evaluator_path: Path,
    work_root: Path,
) -> dict[str, object]:
    label = f"{scenario_path.stem}__{_safe_slug(judge)}__{_safe_slug(attacker_model)}__iters{iters}"
    out_dir = work_root / label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["EVOLVE_SCENARIO"] = str(scenario_path.resolve())
    env["EVOLVE_MODE"] = f"direct+{judge}"
    env["EVOLVE_WORK_DIR"] = str(out_dir / "scenario_traces")

    # If the scenario defines an attacker_goal, materialize a per-cell
    # openevolve config whose system_message names that goal explicitly.
    # Otherwise pass the base config straight through.
    scenario_obj = load_scenario(scenario_path)
    effective_config_path = config_path
    if scenario_obj.attacker_goal:
        effective_config_path = materialize_config_with_goal(
            base_config_path=config_path,
            attacker_goal=scenario_obj.attacker_goal,
            out_dir=out_dir,
        )

    cmd = [
        sys.executable,
        "-m",
        "openevolve.cli",
        str(seed_path.resolve()),
        str(evaluator_path.resolve()),
        "--config",
        str(effective_config_path.resolve()),
        "--iterations",
        str(iters),
        "--output",
        str(out_dir.resolve()),
        "--primary-model",
        attacker_model,
    ]

    print(f"\n=== {label} ===", flush=True)
    print(f"    scenario : {scenario_path}", flush=True)
    print(f"    mode     : direct+{judge}", flush=True)
    print(f"    attacker : {attacker_model}", flush=True)
    print(f"    iters    : {iters}", flush=True)
    print(f"    output   : {out_dir}", flush=True)
    if scenario_obj.attacker_goal:
        print("    config   : per-cell (scenario_aware system message)", flush=True)
    else:
        print("    config   : base (no scenario.attacker_goal)", flush=True)

    t0 = time.time()
    # Run from benchmarks/ cwd so the harness picks up
    # benchmarks/.derail.yaml (cwd is the first lookup).
    completed = subprocess.run(cmd, env=env, cwd=_BENCH_DIR)
    elapsed = time.time() - t0

    metrics, best_iter = read_best(out_dir)
    first_breach = scan_first_breach(out_dir)
    return {
        "scenario": scenario_path.stem,
        "judge_model": judge,
        "attacker_model": attacker_model,
        "iter_budget": iters,
        "combined_score": metrics.get("combined_score"),
        "security_breached": metrics.get("security_breached"),
        "utility_passed": metrics.get("utility_passed"),
        "best_iteration": best_iter,
        "iterations_to_first_breach": first_breach,
        "elapsed_sec": round(elapsed, 1),
        "exit_code": completed.returncode,
        "output_dir": str(out_dir.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=str,
        default=str(_BENCH_DIR / "scenarios" / "*.yaml"),
        help="Glob or comma-separated paths to scenario YAMLs.",
    )
    parser.add_argument(
        "--judges",
        type=str,
        default="judge_gpt4omini,judge_gpt4o",
        help=(
            "Comma-separated classifier names from .derail.yaml "
            "(e.g., judge_gpt4omini,judge_gpt4o,judge_gpt5)."
        ),
    )
    parser.add_argument(
        "--attackers",
        type=str,
        default="gpt-4o-mini,gpt-4o",
        help="Comma-separated mutation-LLM model names for openevolve.",
    )
    parser.add_argument(
        "--iters",
        type=str,
        default="5,15,30",
        help="Comma-separated iteration budgets to test.",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=None,
        help=(
            "Initial injection text file. If set, used for every scenario. "
            "If unset (default), the runner auto-resolves per scenario: "
            "prefers benchmarks/seeds/<scenario_stem>.txt if present, "
            "falls back to benchmarks/seeds/blatant_injection.txt."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Base openevolve config.",
    )
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=_DEFAULT_EVALUATOR,
        help="OpenEvolve evaluator (defaults to benchmarks/openevolve_evaluator.py).",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=_BENCH_DIR / "runs",
        help="Per-run output directory root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_BENCH_DIR / "results.csv",
        help="Path to write the consolidated results CSV.",
    )
    args = parser.parse_args()

    try:
        import openevolve  # noqa: F401
    except ImportError:
        print(
            "openevolve is not installed. Run `uv sync` to install it.",
            file=sys.stderr,
        )
        return 2
    if not args.evaluator.exists():
        print(f"evaluator not found at {args.evaluator}", file=sys.stderr)
        return 2

    scenarios_input = (
        _split_csv(args.scenarios) if "," in args.scenarios else [args.scenarios]
    )
    scenarios = _expand_globs(scenarios_input)
    if not scenarios:
        print(f"no scenarios matched {args.scenarios!r}", file=sys.stderr)
        return 2

    judges = _split_csv(args.judges)
    attackers = _split_csv(args.attackers)
    iter_budgets = [int(x) for x in _split_csv(args.iters)]

    args.work_root.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario",
        "judge_model",
        "attacker_model",
        "iter_budget",
        "combined_score",
        "security_breached",
        "utility_passed",
        "best_iteration",
        "iterations_to_first_breach",
        "elapsed_sec",
        "exit_code",
        "output_dir",
    ]

    write_header = not args.out.exists() or args.out.stat().st_size == 0
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        total = len(scenarios) * len(judges) * len(attackers) * len(iter_budgets)
        idx = 0
        for scenario_path in scenarios:
            for judge in judges:
                for attacker in attackers:
                    for iters in iter_budgets:
                        idx += 1
                        seed_for_this_cell = _resolve_seed(scenario_path, args.seed)
                        print(f"\n[{idx}/{total}] starting", flush=True)
                        print(f"    seed     : {seed_for_this_cell.name}", flush=True)
                        try:
                            row = _run_one(
                                scenario_path=scenario_path,
                                judge=judge,
                                attacker_model=attacker,
                                iters=iters,
                                seed_path=seed_for_this_cell,
                                config_path=args.config,
                                evaluator_path=args.evaluator,
                                work_root=args.work_root,
                            )
                        except Exception as exc:  # noqa: BLE001
                            row = {
                                "scenario": scenario_path.stem,
                                "judge_model": judge,
                                "attacker_model": attacker,
                                "iter_budget": iters,
                                "combined_score": None,
                                "security_breached": None,
                                "utility_passed": None,
                                "best_iteration": None,
                                "iterations_to_first_breach": None,
                                "elapsed_sec": None,
                                "exit_code": -1,
                                "output_dir": f"error: {exc!r}",
                            }
                        writer.writerow(row)
                        f.flush()
                        print(f"    -> {row}", flush=True)

    print(f"\nDone. Results written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
