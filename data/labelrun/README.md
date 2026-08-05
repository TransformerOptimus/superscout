# Calibration label run (fresh 100-task set)

The 100-task calibration set used for the router label run. Tasks are drawn from
SWE-rebench's public 2026 pipeline (the corresponding docker images are public);
this release is distributed under CC-BY-4.0. The router feature arrays (hidden
states and embeddings) are published separately at
https://huggingface.co/datasets/SuperAGI/superscout-router-features and are not
included here.

## Files

- `labelrun_manifest.jsonl` — 100 task definitions: instance id, repo,
  base commit, docker image reference, FAIL_TO_PASS test list and
  PASS_TO_PASS count, difficulty and patch-size metadata, install/test
  configuration.
- `task_inputs.jsonl` — the problem statement given to each fixer, one record
  per task (verbatim upstream issue text).
- `gold_patches.jsonl` — gold patch and test patch per task, plus the
  FAIL_TO_PASS / PASS_TO_PASS test lists.
- `handoffs_raw.jsonl` — the searcher's handoff documents as originally
  generated (99 records; one task has no handoff arm, see below).
- `handoffs_stripped_v2.jsonl` — the canonical stripped handoffs. v2 supersedes
  `handoffs_stripped.jsonl` (v1): after replaying the searcher's in-episode
  files into the base sandbox, reproduction claims that ran clean (exit 0) were
  stripped as false. v2 was the input to the final handoff arm.
- `outcomes.jsonl` — per-episode outcome labels, 796 records:
  400 solo (100 tasks x 4 fixers) + 396 handoff (99 tasks x 4 fixers).
  Fields: `task_id`, `fixer` (GPT-5.2, Claude Opus 4.6, Gemini 3 Flash,
  Kimi K2.5), `arm` (`solo` | `handoff`), `solved` (bool),
  `cost_usd_pinned` (measured episode cost in USD at pinned prices).

## Denominators: 100 vs 99

The solo arm covers all 100 tasks. One task
(`koxudaxi__datamodel-code-generator-2998`) lacks a handoff arm, so the
handoff arm and every paired solo-vs-handoff comparison use N=99.

## Provenance of `outcomes.jsonl`

Derived (not copied) from the frozen per-episode result files of the label run.
Source, relative to the label-run directory this release was staged from:
`phase2_artifacts/episodes/*.json` (solo arm) and
`phase3_artifacts/episodes/*handoff.json` (handoff arm). Transform, the same
extraction the analysis code uses: for each episode JSON, take
`instance_id`, `fixer`, `arm`, `resolved`, and `attempt.cost_usd_pinned`,
keeping only the four fixer models above (one scripted-fixer episode is
excluded, as in the analysis). Internal fixer identifiers were mapped to the
public model names. No values were altered; per-fixer solved counts were
verified against the run's frozen summary (solo solved 61/48/55/52 for
GPT-5.2 / Claude Opus 4.6 / Gemini 3 Flash / Kimi K2.5).
