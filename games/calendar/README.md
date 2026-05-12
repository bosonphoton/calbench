# Calendar Game

This package contains the CalBench calendar scheduling environment. It registers
the `calendar` game with `a2a-engine`, provides task generation and oracle
solving, implements scripted/DSM/LLM agents, and writes JSON traces with
evaluation metrics.

## Public Fixtures

Checked-in fixtures:

| Dataset | Path | Setting |
| --- | --- | --- |
| Uniform full | `tasks/uniform_full.jsonl` | errands and meetings cost `1` |
| Varied full | `tasks/varied_full.jsonl` | errands cost `1`, `100`, or `1000`; meetings cost `1` |

Each task row includes:

- `params`: agents, participants, density, slots, meetings, and cost settings
- `calendars`: private per-agent initial calendars
- `meetings`: incoming meetings and speaker order
- `witness_solution`: generated feasible solution
- `optimal`: CP-SAT oracle schedule and cost
- `greedy`: deterministic greedy schedule and cost
- `difficulty`, `difficulty_score`, and normalized difficulty metadata

## Task Generation

Regenerate the full public fixtures:

```bash
uv run python -m calendar_game.taskgen --suite uniform-full
uv run python -m calendar_game.taskgen --suite varied-full
```

Generate the main 72-task 5-agent/3-participant slice:

```bash
uv run python -m calendar_game.taskgen --config taskgen_configs/calbench_72_uniform.yaml
uv run python -m calendar_game.taskgen --config taskgen_configs/calbench_72_varied.yaml
```

For a custom suite, copy `taskgen_configs/example_custom.yaml`:

```yaml
setting_name: my_calendar_suite
output: tasks/my_calendar_suite.jsonl
summary_output: tasks/my_calendar_suite_summary.json
seed_base: 900000
candidates_per_config: 12
selected_per_bucket: 1
difficulty_scorer: optimal_cost_per_participant_slot
configs:
  - total_agents: [3, 4]
    subset_sizes: [2, 3]
    num_slots: 16
    num_meetings: 3
    densities: [0.4, 0.7]
    pref_levels: [1, 3]
    meeting_cost_level: 1
    errand_cost_level: 3
```

The generator expands each config cell, samples `candidates_per_config` tasks,
solves each with CP-SAT, sorts by difficulty, and selects
`selected_per_bucket` tasks from each easy/medium/hard tertile.

Important knobs:

| Field | Meaning |
| --- | --- |
| `total_agents` | Number of agents with private calendars |
| `subset_size` / `subset_sizes` | Participants per incoming meeting |
| `num_meetings` | Incoming meeting stream length |
| `num_slots` | Calendar horizon; public tasks use `16` |
| `densities` | Fraction of each calendar initially occupied by errands |
| `pref_level` / `pref_levels` | Upper bound on randomly sampled errand costs |
| `meeting_cost_level` | Upper bound on meeting-move costs |
| `errand_cost_level` | Upper bound on errand costs when using uniform random costs |
| `errand_cost_values` | Explicit balanced errand-cost multiset, e.g. `[1, 100, 1000]` |
| `candidates_per_config` | Candidate pool size before difficulty selection |
| `selected_per_bucket` | Tasks retained from each difficulty bucket |

Use `--skip-optimal` only for quick one-off fixtures. Balanced suites require
oracle costs for difficulty bucketing.

Derived suites:

```bash
# Mark errands as immovable/blocked, then recompute feasibility and oracle costs.
uv run python -m calendar_game.taskgen --suite uniform-full-blocked --blocked-errands-per-agent 6
uv run python -m calendar_game.taskgen --suite varied-full-blocked --blocked-errands-per-agent 6

# Cost-ratio and prior-meeting repair probes.
uv run python -m calendar_game.taskgen --suite minimal-cost-ratio-v1
uv run python -m calendar_game.taskgen --suite initial-prior-meeting-move-v1
```

## Running Experiments

Dry-run smoke test with scripted clients and no API keys:

```bash
uv run python run.py experiments/example.yaml --dry-run --max-parallelism 1
```

DSM baseline:

```bash
uv run python run.py experiments/uniform_full_dsm.yaml --max-parallelism 8 --resume
uv run python run.py experiments/varied_full_dsm.yaml --max-parallelism 8 --resume
```

LLM example:

```bash
uv run python run.py experiments/uniform_full_gemini31pro.yaml --max-parallelism 8 --resume
uv run python run.py experiments/varied_full_gemini31pro.yaml --max-parallelism 8 --resume
```

Useful runner flags:

| Flag | Effect |
| --- | --- |
| `--results-dir DIR` | Write traces under `DIR/<experiment_name>/` |
| `--max-parallelism N` | Number of games to run concurrently |
| `--resume` | Skip completed `experiment_run_id`s |
| `--shard-index I --shard-count N` | Partition expanded runs across machines |
| `--dry-run` | Replace agents with deterministic scripted clients |

## Experiment YAML

Minimal real-model experiment:

```yaml
name: my_openai_run
defaults:
  game_name: calendar
  task_path: tasks/calbench_72_uniform.jsonl
  num_agents: 5
  num_slots: 16
  num_meetings: 5
  num_participants: 3
  max_turns_per_round: 15
  decision_retries: 3
  enable_fallback: false
  agents:
    - {type: llm, model: gpt-4o-mini}
    - {type: llm, model: gpt-4o-mini}
    - {type: llm, model: gpt-4o-mini}
    - {type: llm, model: gpt-4o-mini}
    - {type: llm, model: gpt-4o-mini}
batches:
  - label: b037_easy_5a_3p_d0p6_c1_1
    count: 1
    config:
      seed: 323032
      task_id: b037_easy_5a_3p_d0p6_c1_1
```

Agent types:

| Type | Purpose |
| --- | --- |
| `scripted` | deterministic smoke-test behavior; automatically used with `--dry-run` |
| `dsm` | distributed score-based multi-round baseline |
| `paper_dsm` | paper-style privacy-preserving DSM |
| `private_dsm` | high-privacy DSM preset |
| `sd` | scheduling-difficulty baseline |
| `imap` / `incremental_map` | complete-information incremental MAP baseline |
| `llm` | standard prompt with provider-backed LLM calls |
| `dspy` | prompt-variant-backed LLM client |

Provider credentials:

| Provider | Model examples | Credential |
| --- | --- | --- |
| OpenAI | `gpt-4o-mini`, `o3-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Google Gemini API | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Vertex AI | `publishers/google/models/gemini-...` | Google ADC |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct` | `OPENROUTER_API_KEY` |
| Ollama | `llama3`, `qwen...` | local Ollama server |

For custom OpenAI-compatible endpoints, set `api_format`, `api_base`, and
`api_key` directly on each agent entry.

## Extra Experiment Families

Privacy probe:

```bash
uv run python run.py experiments/privacy_probe_nosy_agent0_sample.yaml --max-parallelism 2
```

Adversarial red-team prompts:

```bash
uv run python run.py experiments/redteam_c006_uniform_5a3p_c020.yaml --max-parallelism 8 --resume
uv run python run.py experiments/redteam_c006_varied_5a3p_c020.yaml --max-parallelism 8 --resume
```

Blocked-calendar scheduling:

1. Generate blocked tasks with `calendar_game.taskgen`.
2. Copy an experiment YAML.
3. Set `task_path` to the blocked JSONL.
4. Run the same way as standard fixtures.

## Evaluation

Print grouped metrics:

```bash
uv run python -m calendar_game.evaluate results/uniform_full_dsm
```

Export analysis tables:

```bash
uv run python -m calendar_game.evaluate results/uniform_full_dsm \
  --summary-csv results/uniform_full_dsm/summary.csv \
  --game-csv results/uniform_full_dsm/games.csv \
  --round-csv results/uniform_full_dsm/rounds.csv \
  --agent-csv results/uniform_full_dsm/agents.csv \
  --message-csv results/uniform_full_dsm/messages.csv
```

Programmatic loading:

```python
from calendar_game.dataset import CalendarGameDataset

ds = CalendarGameDataset.from_dir("results/uniform_full_dsm")
game_df = ds.to_game_df()
round_df = ds.to_round_df()
agent_df = ds.to_agent_df()
message_df = ds.to_message_df()
```

Key evaluation columns:

- `coordination_rate`: scheduled meetings divided by attempted meetings
- `realized_cost`, `optimal_cost`, `excess_cost`, `cost_ratio`
- `total_dms`, `msgs_per_meeting`, `dm_chars_per_meeting`
- `cost_gini`, `fairness_metric`, and per-agent `cost_share`

Reusable paper-analysis scripts live in `analysis/scripts/`.

## Tests

```bash
uv run pytest calendar_game/tests/
```
