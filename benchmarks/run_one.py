"""CLI: kick off an openevolve run that refines a prompt-injection
string against a single derail scenario.

Sets the env vars ``openevolve_evaluator.py`` reads, then shells out
to the installed ``openevolve.cli`` module with our config + initial
seed.

Example:
    OPENAI_API_KEY=sk-... \\
    uv run python benchmarks/run_one.py \\
      --scenario benchmarks/scenarios/11_revenue_metrics_exfil.yaml \\
      --mode direct \\
      --iters 20

Prerequisites:
    - synthetic Notion + Linear MCPs running (see benchmarks/mcps/)
    - benchmarks/.derail.yaml configured for linear + notion
    - OPENAI_API_KEY set (used by both the mutation LLM and the agent)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
_DEFAULT_SEED = _BENCH_DIR / "seeds" / "blatant_injection.txt"
_DEFAULT_CONFIG = _BENCH_DIR / "configs" / "openevolve_base.yaml"
_DEFAULT_EVALUATOR = _BENCH_DIR / "openevolve_evaluator.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to the base scenario YAML (e.g., benchmarks/scenarios/11_revenue_metrics_exfil.yaml)",
    )
    parser.add_argument(
        "--mode",
        default="direct",
        help='Target mode: "direct" or "direct+<classifier_name>" (e.g., direct+judge).',
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Total evolution iterations (overrides config.yaml).",
    )
    parser.add_argument(
        "--initial",
        type=Path,
        default=_DEFAULT_SEED,
        help="Initial seed injection text file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="OpenEvolve config YAML.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / "runs" / "evolution",
        help="Parent directory for per-iteration OTEL traces.",
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
    scenario_path = args.scenario.resolve()
    if not scenario_path.exists():
        print(f"scenario not found: {scenario_path}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["EVOLVE_SCENARIO"] = str(scenario_path)
    env["EVOLVE_MODE"] = args.mode
    env["EVOLVE_WORK_DIR"] = str(args.work_dir.resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "openevolve.cli",
        str(args.initial.resolve()),
        str(_DEFAULT_EVALUATOR.resolve()),
        "--config",
        str(args.config.resolve()),
        "--iterations",
        str(args.iters),
    ]
    print(f"→ scenario:     {scenario_path}")
    print(f"→ target mode:  {args.mode}")
    print(f"→ iterations:   {args.iters}")
    print(f"→ work dir:     {args.work_dir}")
    print(f"→ command:      {' '.join(cmd)}")
    print()
    completed = subprocess.run(cmd, env=env, cwd=_REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
