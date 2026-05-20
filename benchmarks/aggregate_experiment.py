"""Post-hoc summary for ``run_experiment.py`` output.

Reads the raw per-run CSV produced by ``run_experiment.py`` and emits a
per-(judge, scenario) summary table:

  * breach_rate    — fraction of repeats whose final best-program score
                     reached the early-stop threshold (target-score),
                     i.e. how often the attacker succeeded at all.
  * n_breached     — count of repeats that breached.
  * n_runs         — total repeats in the cell.
  * median_iter    — median ``first_breach_iter`` across breached runs.
  * iqr_iter       — (q1, q3) of ``first_breach_iter`` across breached
                     runs (Tukey hinge, inclusive).
  * n_censored     — repeats that hit the iteration cap with no breach.

Censored runs are reported as a count rather than imputed with the cap.
A judge that censors 9/10 with one outlier breach at iter 8 looks very
different from a judge that breaches all 10 at iter 240 — flattening
them with an imputed mean loses that signal.

Usage::

    uv run python benchmarks/aggregate_experiment.py \\
      --in  benchmarks/experiment.csv \\
      --out benchmarks/experiment_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def _coerce_int(value: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _coerce_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _quartiles(values: list[int]) -> tuple[float, float]:
    """Return (q1, q3) using the same method as ``statistics.quantiles``
    with ``method='inclusive'``. Returns ``(NaN, NaN)`` for fewer than
    two samples; for exactly one sample, returns ``(v, v)``.
    """
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    qs = statistics.quantiles(values, n=4, method="inclusive")
    return qs[0], qs[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="src",
        type=Path,
        required=True,
        help="Raw per-run CSV from run_experiment.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output summary CSV. If omitted, derived from --in by "
            "appending _summary.csv."
        ),
    )
    args = parser.parse_args()

    if not args.src.exists():
        print(f"input not found: {args.src}", file=sys.stderr)
        return 2

    out_path = args.out
    if out_path is None:
        stem = args.src.stem
        out_path = args.src.with_name(f"{stem}_summary.csv")

    # group rows by (judge, scenario, attacker)
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    with args.src.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("judge", ""),
                row.get("scenario", ""),
                row.get("attacker", ""),
            )
            groups[key].append(row)

    summary_rows: list[dict[str, object]] = []
    for (judge, scenario, attacker), rows in sorted(groups.items()):
        n_runs = len(rows)
        breach_iters: list[int] = []
        censored = 0
        for r in rows:
            iter_val = _coerce_int(r.get("first_breach_iter", ""))
            breached = _coerce_float(r.get("security_breached", ""))
            if iter_val is not None and breached and breached >= 1.0:
                breach_iters.append(iter_val)
            else:
                censored += 1

        n_breached = len(breach_iters)
        median = statistics.median(breach_iters) if breach_iters else None
        q1, q3 = _quartiles(breach_iters)
        summary_rows.append(
            {
                "judge": judge,
                "scenario": scenario,
                "attacker": attacker,
                "n_runs": n_runs,
                "n_breached": n_breached,
                "n_censored": censored,
                "breach_rate": round(n_breached / n_runs, 4) if n_runs else None,
                "median_iter": median,
                "q1_iter": None if q1 != q1 else q1,  # filter NaN
                "q3_iter": None if q3 != q3 else q3,
                "min_iter": min(breach_iters) if breach_iters else None,
                "max_iter": max(breach_iters) if breach_iters else None,
            }
        )

    fieldnames = [
        "judge",
        "scenario",
        "attacker",
        "n_runs",
        "n_breached",
        "n_censored",
        "breach_rate",
        "median_iter",
        "q1_iter",
        "q3_iter",
        "min_iter",
        "max_iter",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"wrote summary for {len(summary_rows)} cells to {out_path}")
    for row in summary_rows:
        print(
            f"  {row['judge']:<20s} {row['scenario']:<40s} "
            f"breach={row['n_breached']}/{row['n_runs']}  "
            f"median_iter={row['median_iter']}  "
            f"IQR=({row['q1_iter']}, {row['q3_iter']})  "
            f"censored={row['n_censored']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
