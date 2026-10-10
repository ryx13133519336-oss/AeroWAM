# AeroWAM-Duo data contract

These three JSON Schemas are the first frozen contract for the evaluator. They are intentionally simulator-agnostic: an AirSim/TravelUAV adapter and a real-flight adapter must emit the same episode fields, while only the environment metadata changes.

| File | Owner | Purpose |
| --- | --- | --- |
| `task_schema.json` | task generator | Defines language goal, two-agent observation/action budgets, communication conditions, hidden events, and source/target split membership. |
| `episode_schema.json` | environment runner | Defines synchronized per-step observations, actions, messages, event labels, and hidden truth written only to logs. |
| `episode_schema_v0_2.json` | deterministic handoff fixture | Extends episode logs with task specifications, sent/delivered message traces, per-agent visibility truth, and replay-complete handoff events. |
| `metrics_schema.json` | evaluator | Defines per-task and aggregate results, confidence intervals, transfer curves, WAM diagnostics, and reproducibility hashes. |

## Contract rules

1. The policy receives only the fields listed in `observation_contract`; `hidden_truth` is evaluator-side data and must never be returned by `reset` or `step`.
2. Both UAVs use the same episode clock and seed. A method is evaluated on identical task fixtures and initial states across methods.
3. Event labels are intervals, not single timestamps. This makes longest interruption, handoff latency, and overlapping verification events recomputable from raw logs.
4. The aggregate score is not a replacement for the metric vector. Leaderboards should expose task success, cooperation events, safety/cost, transfer, and WAM metrics separately.
5. Schema version changes are explicit. A result must record `evaluator_version`, `config_hash`, and `environment_hash` before it can enter a public leaderboard.

## Minimal validation

The files are valid JSON and can be parsed with PowerShell's `ConvertFrom-Json`. The next evaluator milestone should add a JSON-Schema validator in CI and a fixture test that checks:

- a valid six-template task and one three-seed episode;
- rejection of a task exposing target coordinates or teammate ground-truth pose;
- rejection of an episode with only one UAV or non-monotonic timestamps;
- deterministic recomputation of the metrics summary from the episode log.
