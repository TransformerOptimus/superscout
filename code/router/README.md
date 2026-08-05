# SuperScout frozen router

This directory ships the exact frozen SuperScout routing head: a resume-based head
that blends an issue-text embedding with the searcher's hidden state and outputs, per
candidate fixer, a probability that the cheap fixer solves the task. The task is
routed to the first fixer in measured-cheapest-first order whose probability clears
the threshold, else to the anchor.

The head weights, resume centroids, and feature recipe were frozen on 2026-07-29,
before any evaluation-set data existed (see `"firewall"` in `router_spec.json`, which
covers those fitted parameters, not the operating threshold). `router_head.npz` is a
byte-identical copy of the frozen weights file. Recomputing this head over the
released per-task feature arrays reproduces the published per-task probabilities and
routing decisions in `data/receipts/routing_scenarios.json` exactly.

## Files

| file | contents |
|---|---|
| `router_spec.json` | frozen configuration: pool, threshold, route rule, feature recipe, embedder pin, fit scope |
| `router_head.npz` | serialized head: per-fixer resume centroids (`s` solved, `f` failed, `p` base rate) and StandardScaler + logistic-regression coefficients, for both feature spaces |
| `infer.py` | minimal numpy-only inference: load spec + head, route one task |

The resume centroids the deployed head uses are serialized inside `router_head.npz`
(arrays `text_s`, `text_f`, `text_p`, `state_s`, `state_f`, `state_p`); no separate
resume file is consumed at inference time.

## Input contract

Two vectors per task:

1. **Text embedding** (1024-d): the task's problem statement embedded with
   `Qwen/Qwen3-Embedding-0.6B` at pinned revision
   `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, `max_seq_length=4096`, L2-normalized,
   plain text with no instruction prefix (see `"embedder"` in the spec).
2. **Hidden state** (3584-d): the SuperScout-7B final-token pre-decode hidden state
   from layer -4 (spec key `state_pre_-4`), L2-normalized.

From each vector `x` and each fixer's resume `(s, f, p)` the head computes the
six-feature vector `[cos(x,s), cos(x,f), cos(x,s)-cos(x,f), p, cos(x,s)*p,
(cos(x,s)-cos(x,f))*p]`, applies the frozen scaler + logistic regression, and the two
spaces' probabilities are averaged (the `"blend"` recipe).

## Operating point

- Threshold: theta = 0.30, calibrated on the 99-task label-run with-handoff outcomes
  (matched-point rule; `"theta_note"` in the spec).
- Route rule: first fixer in `cheap_order` with P(solve) >= theta, else the anchor
  (`gpt-5-2-high`). `cheap_order` is fixed by measured mean solo cost per task
  (`mean_solo_cost_labelrun` in the spec, kept verbatim).
- On the 266-task evaluation set, theta = 0.30 routes 263 tasks to Kimi K2.5 and 3
  to Gemini 3 Flash, with zero anchor fallbacks (see
  `data/receipts/routing_scenarios.json`, which also documents how the router row is
  scored).
- If no hidden state is available the route degrades to the text head alone at the
  same theta and tags the record degraded; a state is never synthesized. A text-only
  head at theta = 0.425 (`"text_only"` in the spec) is retained as a
  non-claim-bearing reference.

The fixer pool strings in the JSON are the internal model ids used throughout the
artifact and are kept untouched: `gpt-5-2-high` is GPT-5.2, `claude-4-6-opus` is
Claude Opus 4.6, `gemini-3-flash` is Gemini 3 Flash, `kimi-k2-5` is Kimi K2.5.

## Running

Requires only Python 3 + numpy.

```bash
# smoke test with synthetic unit vectors of the correct dimensionality
python infer.py --smoke

# route a real task from precomputed vectors
python infer.py --text-npy text_embedding.npy --state-npy hidden_state.npy
```

Or from Python:

```python
from infer import load_artifact, route
art = load_artifact()
rec = route(text_embedding, hidden_state, art)   # rec["pick"], rec["p_solve_blend"]
```

## Renames vs. the internal artifact

Public vocabulary only; no weights, costs, or measured values changed. The edits:

- the internal working names of the two recipe variants were replaced: the deployed
  text + hidden-state average is now `blend`, and the text-only reference head is now
  `text_only` (spec keys and values renamed consistently)
- the weights file was renamed to `router_head.npz` (byte-identical copy of the
  frozen original)
- absolute fit-time input paths in `"inputs"` replaced with descriptive notes; those
  fit-time data files are not part of this release
- one internal review annotation about the fit was removed from `fit_scope`; the fit
  scope numbers themselves are unchanged

`SHA256_MANIFEST.json` is regenerated over the files as staged here.
