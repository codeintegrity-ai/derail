# derail

A benchmark for measuring how well an LLM-as-a-judge classifier defends an
MCP-connected agent against indirect prompt injection. You bring an MCP
deployment and a workflow, plant an injection in a data field the agent will
read, wire a classifier in as a pre-tool / post-tool hook, and let an
evolutionary search iterate the planted text against the live (agent +
classifier) loop until it finds a phrasing that bypasses the judge and
completes the attacker's goal.

The reference deployment is synthetic Notion + Linear MCPs, with two
scenarios under `benchmarks/scenarios/` — both featured in
`blog_classifier_evasion.md` and snapshotted under `benchmark_logs/`.

![How derail tests adversarial prompt injections against classifier-defended agents](benchmark_flow.svg)

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
export OPENAI_API_KEY=sk-...
```

`uv sync` installs [openevolve](https://github.com/algorithmicsuperintelligence/openevolve)
from PyPI; the runner invokes it as `python -m openevolve.cli`.

## Start the MCPs

The benchmark targets two synthetic MCP servers that mirror Notion and Linear.
Run each in its own terminal (or in the background):

```bash
uv run python benchmarks/mcps/linear_mcp_server.py    # 127.0.0.1:5021/mcp
uv run python benchmarks/mcps/notion_mcp_server.py    # 127.0.0.1:5031/mcp
```

Endpoint topology and the LLM-judge classifier variants are declared in
`benchmarks/.derail.yaml`. Six judge entries are pre-wired (`judge_gpt4omini`,
`judge_gpt4o`, `judge_gpt5`, `judge_gpt5mini`, `judge_gpt54`, `judge_gpt55`) and
share a single general-purpose judge prompt via a YAML anchor.

## Scenarios

Each scenario is a YAML file under `benchmarks/scenarios/` describing:

- the MCPs the agent has access to,
- a `seed:` block of pre-state to plant in the upstream MCPs (issues, pages,
  comments, etc.),
- a `prompt:` the user asks the agent,
- `utility:` predicates — did the agent complete the legitimate task,
- `security:` predicates — did the injection succeed (lower is better),
- `inject:` — which seed step's field receives the evolved text,
- `attacker_goal:` — reconnaissance + guidance the evolutionary mutator sees.

A matching seed file lives at `benchmarks/seeds/<scenario_name>.txt` and holds
the iteration-0 blatant payload. The runner auto-resolves it by scenario name;
override with `--seed`.

To add a scenario, drop a new YAML into `benchmarks/scenarios/` and a matching
seed into `benchmarks/seeds/`. The runner discovers it by glob.

## Run

Single (scenario × judge × attacker × iters) cell:

```bash
uv run python benchmarks/run_benchmark.py \
  --scenarios benchmarks/scenarios/11_revenue_metrics_exfil.yaml \
  --judges    judge_gpt54 \
  --attackers gpt-5.4 \
  --iters     50
```

Full sweep:

```bash
uv run python benchmarks/run_benchmark.py \
  --scenarios "benchmarks/scenarios/*.yaml" \
  --judges    judge_gpt4omini,judge_gpt4o,judge_gpt5 \
  --attackers gpt-4o-mini,gpt-4o \
  --iters     5,15,30 \
  --out       benchmarks/results.csv
```

Each row in the output CSV is one cell with `combined_score`,
`security_breached`, `utility_passed`, and `iterations_to_first_breach`.
Per-iteration OTEL traces and openevolve checkpoints land under `runs/`.

## Reproducing the blog runs

Notable runs cited in `blog_classifier_evasion.md` are snapshotted under
`benchmark_logs/`. Each snapshot is self-contained — scenario YAML, seed,
materialized openevolve config, checkpoints, traces, and a `SUMMARY.md` with
the exact reproduction command.

## Further reading

- `docs/architecture.md` — repo layout, modes, classifier wiring, and the
  trace → score pipeline.
- `benchmarks/README.md` — deeper notes on the sweep, the judge prompt
  rationale, and how to add judges or attacker models.
- `docs/derail/predicate_cookbook.md` — predicate DSL reference for
  `utility:` / `security:` blocks.
- `CLAUDE.md` — code style + Claude-specific guidance for editing this repo.
