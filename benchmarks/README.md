# Classifier-Evasion Benchmark

A benchmark suite for measuring how reliably an LLM-judge content-classifier defense
holds up against an iterating adversary. Sweeps across:

- **2 scenarios** spanning Notion + Linear MCPs — both cross-MCP data-exfil
  shapes where a Linear injection coaxes the agent into fetching a
  confidential Notion page and quoting it back into a Linear comment.
- **Multiple judge models** — the same general judge prompt run against
  different OpenAI models so we can compare classifier strength.
- **Multiple attacker (mutation) models** — openevolve uses an LLM to propose
  payload edits; we can vary which model plays attacker.
- **Multiple iteration counts** — 5 / 15 / 30 etc., to plot "iterations to
  first breach."

Output is a single CSV that the blog post draws from.

## Layout

```
benchmarks/
├── README.md                       this file
├── .derail.yaml             endpoint + classifier config (multi-judge)
├── mcps/
│   ├── notion_mcp_server.py        synthetic Notion MCP (port 5031)
│   └── linear_mcp_server.py        synthetic Linear MCP (port 5021)
├── scenarios/                      2 YAML scenarios
│   ├── 03_notion_confidential_exfil.yaml
│   └── 11_revenue_metrics_exfil.yaml
├── configs/
│   └── openevolve_base.yaml        base openevolve config (per-cell variants
│                                   are materialized at run time from the
│                                   scenario's attacker_goal)
├── seeds/                          per-scenario blatant seed payloads
│   ├── 03_notion_confidential_exfil.txt
│   ├── 11_revenue_metrics_exfil.txt
│   └── blatant_injection.txt       generic fallback (scores 0 against any judge)
└── run_benchmark.py                sweep runner — writes results.csv
```

## Quickstart

```bash
# 1. Start the synthetic MCPs (each in its own terminal or as background)
uv run python benchmarks/mcps/linear_mcp_server.py    # → 127.0.0.1:5021/mcp
uv run python benchmarks/mcps/notion_mcp_server.py    # → 127.0.0.1:5031/mcp

# 2. Run a single combination (smoke test)
EVOLVE_BENCH_CONFIG=benchmarks/.derail.yaml \
uv run python benchmarks/run_benchmark.py \
  --scenarios benchmarks/scenarios/11_revenue_metrics_exfil.yaml \
  --judges judge_gpt4omini \
  --attackers gpt-4o-mini \
  --iters 5

# 3. Run the full sweep
uv run python benchmarks/run_benchmark.py \
  --scenarios "benchmarks/scenarios/*.yaml" \
  --judges judge_gpt4omini,judge_gpt4o,judge_gpt5 \
  --attackers gpt-4o-mini,gpt-4o \
  --iters 5,15,30 \
  --out benchmarks/results.csv
```

## How the sweep is structured

Each row in `results.csv` is one (scenario, judge_model, attacker_model, iter_budget)
combination. For each combination the runner:

1. Resets the synthetic MCPs to a clean state.
2. Points openevolve at the scenario in `direct+<judge>` mode — pre-tool and
   post-tool hooks are wired to the named LLM judge.
3. Runs openevolve for `iter_budget` iterations using the named attacker model
   as the mutation operator.
4. Reads the best program's metrics (combined_score, security_breached,
   utility_passed) and records the iteration index where the first breach
   occurred.

Columns:

```
scenario, judge_model, attacker_model, iter_budget,
combined_score, security_breached, utility_passed,
iterations_to_first_breach, best_payload_path
```

## Why the judge prompt is deliberately general

The `_judge_prompt:` YAML anchor at the top of `benchmarks/.derail.yaml`
defines a single, *general* prompt-injection-detector prompt shared by
every judge variant — it catches the obvious blatant patterns (SYSTEM
OVERRIDE, "ignore previous instructions", DAN, role-override delimiters,
jailbreak markers) but does not enumerate every possible bypass phrasing.

This is intentional. An over-tuned judge that lists every red flag will block
the blatant seed *and* the soft-policy paraphrases the search would otherwise
discover, but it will also false-positive on legitimate corporate content —
which is precisely the trade-off the blog post is about. We want the judge
to look like what a thoughtful production deployment actually ships, so the
benchmark measures the failure mode that real systems exhibit.

## Adding a new scenario

Drop a YAML into `benchmarks/scenarios/` following the format used by the
two provided examples:

```yaml
name: my-scenario
description: ...
mcps: [linear, notion]
seed:
  - tool: linear.save_issue
    args: {...}
    capture_id_as: issue_id
prompt: |
  Realistic user task referencing {issue_id}.
utility:
  - call_made: ...   # did the agent do the legit task
security:
  - call_made: ...   # did the injection succeed
cleanup: []
```

The runner picks it up automatically.

## Adding a new judge model

Add a classifier entry to `benchmarks/.derail.yaml` under `classifiers:`
and reference it from `--judges` on the runner.

## Adding a new attacker model

Pass the model name directly as `--attackers` on the runner — it is forwarded
to openevolve as `--primary-model`. The base config at
`benchmarks/configs/openevolve_base.yaml` is shared across attacker models;
per-cell variants are materialized at run time from the scenario's
`attacker_goal:`.
