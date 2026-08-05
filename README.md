
# SuperScout

SuperScout routes repository-level software issues to frontier fixer models
after scouting the repository. A 7B searcher, SuperScout-7B, first explores the
repository and produces a structured handoff whose reproduction claims are
sandbox-verified, with false claims stripped before delivery; the searcher's
hidden states, together with the task text, then feed a resume-based router that
dispatches the task to one of four frontier fixers (GPT-5.2, Claude Opus 4.6,
Gemini 3 Flash, Kimi K2.5). Adding a new fixer requires no retraining. On the
full Python slice of SWE-bench Pro (266 tasks, official capped budget tier),
SuperScout matches the best single model's solve rate (159 of 266 for
SuperScout, 158 for the best model) at about a fifth of the total cost per
solve at a matched solve rate.

This repository contains the paper source and the released artifacts that back
its claims.

## Layout

```
paper/               LaTeX source of the paper (builds to main.pdf)
code/
  gate/              verify-then-strip gate: module, unit tests, handoff JSON schema
  router/            frozen router: spec, serialized head (npz), numpy-only inference
data/
  blocklist.json     23-repo contamination blocklist (11 SWE-bench Pro + 12 Verified)
  labelrun/          100-task calibration label run (tasks, handoffs, outcomes)
  receipts/          SWE-bench Pro Python-266 evidence layer (per-task receipts)
LICENSE              Apache-2.0 (code)
data/LICENSE         CC-BY-4.0 (data)
hf/                  upload-ready HuggingFace model and dataset cards
```

## Building the paper

```
cd paper && make
```

Output: `paper/main.pdf`.

## Claims-to-artifacts manifest

Every main-paper table and figure over the SWE-bench Pro benchmark is
recomputable from the staged files, without access to raw trajectories. See
`data/receipts/README.md` for per-file field documentation, the internal-name
mapping, and integrity notes, and `data/labelrun/README.md` for the calibration
files.

| Main float | Backing artifact(s) in this repo |
|---|---|
| Table: main results (Pro Python-266) | `data/receipts/paired_outcomes.json` and `data/receipts/outcomes_per_task.json` (solo rows); `data/receipts/routing_scenarios.json` joined with `outcomes_per_task.json` (system rows); `data/receipts/handoffs_stripped.jsonl` (the exact handoffs injected into the system runs) |
| Figure: cost per solve vs solve rate | same files as the main results table (same five points) |
| Table: file localization (Pro Python-266) | `data/receipts/localization.json` |
| Figure: gate-probability histograms | `data/receipts/routing_scenarios.json` (per-task, per-fixer probability vectors and thresholds) |
| Table: router recipe | `code/router/router_spec.json`, `code/router/router_head.npz`, `code/router/infer.py` (frozen deployed configuration, bit-for-bit) |
| Table: training-data composition | `data/blocklist.json` (contamination blocklist rows); the corpus itself is on HuggingFace (see below) |
| Figure: handoff redistribution (calibration) | `data/labelrun/outcomes.jsonl` (796 per-episode labels: 100 tasks x 4 fixers solo, 99 x 4 with handoff) |
| Figure: router design space (calibration) | `data/labelrun/` outcome and cost labels; the router feature arrays (hidden states and embeddings) are on HuggingFace (see below) |
| Figure: decoding (greedy vs sampled) | not backed here: the 450-task held-out exam and 100-task dial-set runs are not part of this release |
| Figure: language transfer | not backed here: the SWE-bench Multilingual handoff run is not part of this release |
| Figures: pipeline, solve-sets, onboarding | schematics, no measured quantities |
| Table: capability matrix | literature comparison, no data artifact |

In-text claims with staged receipts:

| Claim | Backing artifact(s) |
|---|---|
| Verify-then-strip rates (249 of 266 handoffs claiming a true reproduction; 50 genuine, 174 stripped; 206 spontaneous / 60 forced) | `data/receipts/verify_records/` (266 per-task records), `data/receipts/handoffs_stripped.jsonl`; mechanism in `code/gate/` |
| Searcher decode configuration (temperature 0.9, single draw, pinned sampling) | `data/receipts/sampling_pin_check.json` (6/6 PASS audit) |
| Router weights and resumes frozen before evaluation; threshold calibrated on lab outcomes | `code/router/router_spec.json` (`"firewall"` block and `theta_note`) |
| Zero blocklist hits in the frozen training file | `data/blocklist.json` (list and enforcement note) |

## Licenses

- Code (`code/`, `paper/`): Apache-2.0, see `LICENSE`.
- Data (`data/`): CC-BY-4.0, see `data/LICENSE`.

## Artifacts on HuggingFace

Private until the paper release; the links are permanent.

- SuperScout-7B weights:
  [`SuperAGI/SuperScout-7B`](https://huggingface.co/SuperAGI/SuperScout-7B)
  (model card mirrored at `hf/model-card.md`).
- The 19,911-example SFT search corpus:
  [`SuperAGI/superscout-sft-search`](https://huggingface.co/datasets/SuperAGI/superscout-sft-search)
  (dataset card mirrored at `hf/dataset-card.md`).
- The router feature arrays (hidden states and embeddings for the 266-task
  evaluation and the 100-task calibration set), not included in `data/`:
  [`SuperAGI/superscout-router-features`](https://huggingface.co/datasets/SuperAGI/superscout-router-features).

## Citation

```bibtex
@misc{superscout2026,
  title  = {Scrouting: Cost-Aware Routing of Coding Agents by Scouting the Repository First},
  author = {Ishaan Bhola and Adithyan Krishnan and Mukunda NS},
  year   = {2026},
  note   = {Preprint. Citation will be updated on release.}
}
```
