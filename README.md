# CalBench

CalBench is a controlled benchmark for multi-agent calendar scheduling. Each
task gives agents private calendars with pre-existing errands and meetings, then
asks them to schedule a stream of incoming meetings through decentralized
communication. Agents see only their own calendar, so successful scheduling
requires coordinating across private information boundaries.

The public harness includes:

- configurable calendar task generation with CP-SAT oracle solutions
- a YAML experiment runner for scripted, baseline, and LLM agents
- DSM, private DSM, scheduling-difficulty, incremental-MAP, scripted, and LLM agents
- JSON trace output with coordination, cost, communication, fairness, and privacy events
- dataset/evaluation utilities for turning traces into pandas tables or CSVs

Public result traces are hosted at http://35.91.104.98:8000/leaderboard

View agent traces at http://35.91.104.98:8000/viewer.html

## Repository Layout

```text
a2a-engine/                    Game-agnostic agent-to-agent experiment framework
expt-runner/                   YAML experiment runner
games/calendar/                CalBench task engine, agents, fixtures, and analysis
games/calendar/tasks/          Checked-in benchmark task JSONL files
games/calendar/taskgen_configs/ YAML configs for generating new task suites
games/calendar/experiments/    Runnable experiment YAMLs
games/calendar/analysis/       Reusable analysis scripts used for paper figures/RQs
```

## Install

From the repository root:

```bash
uv pip install -e a2a-engine -e expt-runner -e games/calendar
```

For package-local development:

```bash
cd games/calendar
uv sync
```

## Fast Smoke Test

This uses deterministic scripted clients and does not require API keys:

```bash
cd games/calendar
uv run python run.py experiments/example.yaml --dry-run --max-parallelism 1
```

Real runs write traces to:

```text
games/calendar/results/<experiment_name>/<game_id>.json
```

## Generate Tasks

The checked-in public fixtures are:

- `games/calendar/tasks/uniform_full.jsonl`
- `games/calendar/tasks/varied_full.jsonl`

Regenerate those fixtures:

```bash
cd games/calendar
uv run python -m calendar_game.taskgen --suite uniform-full
uv run python -m calendar_game.taskgen --suite varied-full
```

Generate the 72-task 5-agent/3-participant configuration used for the main
5a3p benchmark slice:

```bash
uv run python -m calendar_game.taskgen --config taskgen_configs/calbench_72_uniform.yaml
uv run python -m calendar_game.taskgen --config taskgen_configs/calbench_72_varied.yaml
```

Create your own suite by copying `taskgen_configs/example_custom.yaml` and
editing:

- `total_agents`, `subset_size` / `subset_sizes`
- `num_meetings`, `num_slots`
- `densities`
- `pref_level` / `pref_levels`
- `meeting_cost_level`, `errand_cost_level`, or explicit `errand_cost_values`
- `candidates_per_config` and `selected_per_bucket`

The generator samples candidates per configuration cell, solves each with
OR-Tools CP-SAT, buckets candidates by oracle difficulty, and writes selected
easy/medium/hard tasks plus a summary JSON.

## Run Baselines

DSM baseline on the checked-in fixtures:

```bash
cd games/calendar
uv run python run.py experiments/uniform_full_dsm.yaml --max-parallelism 8 --resume
uv run python run.py experiments/varied_full_dsm.yaml --max-parallelism 8 --resume
```

Other implemented baseline/client types can be used in experiment YAML:

- `scripted`: deterministic smoke-test client
- `dsm`: distributed score-based multi-round baseline
- `paper_dsm`: privacy-preserving DSM preset
- `private_dsm`: high-privacy DSM preset
- `sd`: scheduling-difficulty baseline
- `imap` / `incremental_map`: complete-information incremental MAP baseline
- `llm` / `dspy`: model-backed agents

## Run LLM Agents

Set the provider key for the models in your experiment YAML, then run:

```bash
export OPENAI_API_KEY=...
# or ANTHROPIC_API_KEY=...
# or GOOGLE_API_KEY=...
# or OPENROUTER_API_KEY=...

cd games/calendar
uv run python run.py experiments/uniform_full_gemini31pro.yaml --max-parallelism 8 --resume
```

Provider detection is model-name based. You can also set `api_base`,
`api_format`, and `api_key` directly in an agent YAML entry for an
OpenAI-compatible endpoint:

```yaml
agents:
  - type: llm
    model: my-model
    api_format: openai
    api_base: https://my-provider.example/v1
    api_key: ${MY_PROVIDER_API_KEY}
```

Vertex AI models use Google ADC instead of API-key env vars. Authenticate with
`gcloud auth application-default login` and set `GOOGLE_CLOUD_PROJECT` /
`GOOGLE_CLOUD_LOCATION` as needed.

## Run Adversarial/Redteam & Blocked Calendar Version Experiments

Privacy/adversarial red-team runs:

```bash
cd games/calendar
uv run python run.py experiments/privacy_probe_nosy_agent0_sample.yaml --max-parallelism 2
uv run python run.py experiments/redteam_c006_uniform_5a3p_c020.yaml --max-parallelism 8 --resume
uv run python run.py experiments/redteam_c006_varied_5a3p_c020.yaml --max-parallelism 8 --resume
```

Blocked-calendar tasks are generated from an existing fixture:

```bash
uv run python -m calendar_game.taskgen --suite uniform-full-blocked --blocked-errands-per-agent 6
uv run python -m calendar_game.taskgen --suite varied-full-blocked --blocked-errands-per-agent 6
```

Then point an experiment YAML at the generated `task_path`.

## Evaluate Traces

Print grouped metrics:

```bash
cd games/calendar
uv run python -m calendar_game.evaluate results/uniform_full_dsm
```

Write CSVs for downstream analysis:

```bash
uv run python -m calendar_game.evaluate results/uniform_full_dsm \
  --summary-csv results/uniform_full_dsm/summary.csv \
  --game-csv results/uniform_full_dsm/games.csv \
  --round-csv results/uniform_full_dsm/rounds.csv \
  --agent-csv results/uniform_full_dsm/agents.csv \
  --message-csv results/uniform_full_dsm/messages.csv
```

Main metrics:

- `coordination_rate`: fraction of meetings scheduled consistently
- `realized_cost`: displacement cost actually incurred
- `optimal_cost`: CP-SAT oracle cost for the same task
- `excess_cost`: `realized_cost - optimal_cost`
- `cost_ratio`: `realized_cost / optimal_cost`
- `msgs_per_meeting` and `dm_chars_per_meeting`: communication load
- `cost_gini` and `fairness_metric`: cost-distribution fairness

## Tests

```bash
cd games/calendar
uv run pytest calendar_game/tests/
```

## Citation

If you use CalBench, please cite https://arxiv.org/abs/2605.09823.
