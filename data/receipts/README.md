# SWE-bench Pro evidence layer (receipts)

Frozen evidence for the SuperScout evaluation on the 266-task Python slice of
SWE-bench Pro (repo families: ansible 96, openlibrary 91, qutebrowser 79).
Arms: the SuperScout system (SuperScout-7B searcher + routed fixers) and three
solo fixer arms. Model id strings inside the data are the raw run identifiers:
`gpt-5-2-high` (GPT-5.2), `claude-4-6-opus` (Claude Opus 4.6), `kimi-k2-5`
(Kimi K2.5), `gemini-3-flash` (Gemini 3 Flash, routed cheap fixer only).

Every main-paper table and figure over this benchmark is recomputable from the
files below without access to raw trajectories. No measured number was altered
in staging; the edits are naming scrubs, the key renames documented in the
mapping table at the end, and the per-file derivations documented below.

## Files

### verify_records/ (266 files, one JSON per task)
Per-task record of the reconstruct-then-verify-then-strip pass over the
SuperScout-7B search-phase handoffs. Fields:

- `task_id`, `handoff_type` (`spontaneous` | `forced`)
- `handoff_text`: the raw searcher handoff (model-generated text, verbatim;
  paths inside it such as `/app` are sandbox-container paths)
- `handoff_text_final`: the stripped, injection-ready handoff (repro claims
  removed when they failed verification)
- `verify`: `has_repro`, `command`, `claim`, `ran`, `returncode`, `verified`,
  `stripped`, `reason`, `output_tail`
- `failure_class`: one of `passed_stripped`, `real_failure`, `claim_not_true`,
  `import_error`, `nonzero_other`, `missing_file`, `no_repro`, `no_command`
- `replay` (repro-file replay op counts), `hidden_test_collision`

Backs: the verify-strip failure-class table (appendix) and the in-text
verify-strip numbers (249/266 claiming true, 50 genuine, 174 stripped;
spontaneous vs forced split 206/60).

### handoffs_stripped.jsonl (266 lines)
The canonical injection-ready stripped handoff set, one JSON object per task
(`task_id`, `handoff_type`, `handoff_text_final`, `verify`, `failure_class`).
This is exactly the text injected into the fixer stage of the system runs.

Canonical-version decision: the source tree contained two candidates; the
verify-strip run report pins its output artifact as the 678346-byte file with
sha256 `70466df0d2e1793697764e278396201637e6c77a4c80ee6a192d443ed32cd876`
(also pinned in the frozen verify-strip summary receipt). That file is shipped
here, byte-identical. The other candidate was an earlier intermediate of the
same run and is not shipped.

Backs: handoff-cap statistics and, as the injected input, all system-arm
results.

### paired_outcomes.json
Aggregate paired comparison of the two anchor solo fixers over all 266 tasks:
resolve counts and rates with Wilson 95% intervals (GPT-5.2 139/266 = 52.26%,
Claude Opus 4.6 158/266 = 59.40%), discordant-pair counts, exact McNemar
p = 0.02949, per-family splits, and per-task `rows` (solved flag, cost, steps
per fixer). Copied byte-identical from the frozen receipt.

Backs: the solo-anchor rows of the main results table and the paired
anchor-comparison numbers in the text.

### outcomes_per_task.json (derived; sources and transform below)
Per-task graded outcomes for every arm.

- `solo`: for each of the 266 tasks, `family` plus, per solo arm
  (`gpt`, `opus`, `kimi`): `resolved`, `cost_usd_pinned`,
  `cost_usd_litellm`, `steps`.
- `system_union_pairs`: for each (task, fixer) pair actually executed with
  the SuperScout system (searcher handoff + fixer): `resolved`,
  `cost_usd_pinned`, `cost_usd_litellm`, `steps`. 364 pairs: the 331 pairs
  of the frozen union run plus 33 post-hoc episodes marked `post_hoc: true`
  (30 Kimi K2.5 + handoff, 3 Gemini 3 Flash + handoff) run after the
  2026-07-30 evaluation freeze under the same official capped protocol (same
  frozen stripped handoffs, same frozen harness image, same grader). With
  these, Kimi K2.5 + handoff covers all 266 tasks and every task routed in
  `routing_scenarios.json` has a measured episode for its routed fixer.

Derivation: fields are copied unmodified from the frozen graded episode
records of each arm (grader: `upstream_verbatim`; sources: the per-task
episode JSONs of the three solo runs, the union system run, and the post-hoc
system runs in the frozen evaluation tree; field paths: top-level `resolved`,
`attempt.cost_usd_pinned`, `attempt.cost_usd_litellm`, `attempt.n_steps`).
No value was invented or recomputed. Staging cross-checks (all passed):

- solo resolve totals equal the published 139 / 158 / 149;
- `resolved`, `steps` and litellm costs match `paired_outcomes.json` rows
  exactly (costs to that receipt's 4-decimal rounding);
- joining `routing_scenarios.json` decisions against `system_union_pairs`
  reproduces the router cell exactly (159/266 solved, fixer cost to $0.01;
  $0.2299/solve with the flat add below), and the 266 Kimi K2.5 + handoff
  entries reproduce the no-router reference cell exactly (159/266 solved,
  $0.2272/solve).

Cost conventions in the frozen receipts: `paired_outcomes.json` per-row costs
and the GPT-5.2 / Opus 4.6 arm totals ($151.64 / $201.25) are litellm-priced;
the Kimi K2.5 arm total ($28.23) and the router-cell fixer costs are
pinned-priced (provider-pinned pricing). Both measured fields are included per
task so either aggregate can be recomputed.

All-in system costs in the paper add a flat $5.13 of measured searcher-side
spend once (not per task) on top of the per-task fixer costs in these
receipts: the searcher's entire GPU bill for the 266 search episodes ($1.13
total) plus approximately $4 of verify-then-strip sandbox infrastructure.
Both come from the paper's measured-spend disclosure (the spend table in the
appendix); neither is a per-task field in these files.

Backs: the main results table (solo rows directly; the router row and the
no-router ablation row via the joins above) and the cost-per-task /
cost-per-solve numbers.

### routing_scenarios.json
The router receipt at the calibrated operating point `theta` = 0.30:
threshold, cheap-fixer preference order, the routing split (Kimi K2.5 263 /
Gemini 3 Flash 3 / others 0, zero anchor fallbacks), the router cell and the
no-router blanket-Kimi reference cell, per-task routing `decisions`, and
per-task, per-fixer `probability_vectors` with the three router heads:
`p_text` (text-only head), `p_state` (hidden-state head), `p_blend` (uniform
average, the routing head).

Every probability and decision recomputes deterministically from the released
frozen head (`code/router/router_head.npz` + `router_spec.json`, evaluated by
`code/router/infer.py`) over the released per-task feature arrays; staging
re-derived all 266 decisions that way with zero mismatches.

theta = 0.30 was calibrated on the 99-task label-run with-handoff outcomes,
with no evaluation-benchmark data in the selection. The router cell is
evaluated by scoring each task's routed pick against that fixer's measured
with-handoff episode in `outcomes_per_task.json` under the official capped
protocol; all 266 picks have real measured episodes.

Backs: the router row and the no-router ablation row of the main results
table (via the join described above), the gate-probability histogram figure
(distributions of the three heads), and the routed-composition numbers.

### localization.json
File-localization quality of the 266 stripped handoffs against gold-patch
files: overall and by handoff type (spontaneous/forced), by repo family, a
solved/unsolved crosstab (`outcome_crosstab_router`, partitioned by joining
`routing_scenarios.json` decisions with `outcomes_per_task.json`), a sanity
anchor vs the label-run baseline, and per-task `rows` (n_pred, n_gold,
recall, precision, f1, all_hit). The `convention` block documents the
gold-file and predicted-file extraction conventions verbatim.

Backs: the localization table (overall / spontaneous / forced) and the
appendix per-repo localization table.

### sampling_pin_check.json
The frozen 6/6 PASS sampling-pin audit for the SuperScout-7B serving stack:
pinned sampling tail (top_p 1.0, top_k -1, penalties 0), resolved decode spec
(temperature 0.9, max_tokens 2048, stop `<|im_end|>`), wire-payload pin,
cache-salt acceptance, no-server-override, and driver sampling stamps. Copied
byte-identical from the frozen receipt.

Backs: the decoding/sampling claims in the training and calibration sections
(reproducibility of the searcher decode configuration).

### SHA256_MANIFEST.json
Fresh manifest (byte counts + sha256) over every staged file in this
directory, generated after all scrubs. Hashes therefore cover the shipped
bytes, not the internal pre-scrub originals.

## Vocabulary mapping (internal grid names to public names)

The internal receipts used a grid vocabulary for the two router heads and for
system-run configuration labels. All keys and in-string mentions were renamed
as follows; values were left untouched.

| internal | public | meaning |
|---|---|---|
| `Row A` / `rowA` | `text_only` | text-only router head / variant |
| `Row C` / `rowC` | `blend` | blend router head (the router) |
| `p_rowC_blend` | `p_blend` | blend head probability |

The localization crosstab (`outcome_crosstab_router`) is computed from the
routing decisions and per-task outcomes above rather than carrying its own
outcome flags, so no configuration-label keys remain in `localization.json`.

## Integrity

- Every JSON file parses; the JSONL parses line by line.
- Verify records and `handoffs_stripped.jsonl`, `paired_outcomes.json`,
  `sampling_pin_check.json` are byte-identical to the frozen originals.
- `localization.json` differs from its original only by the renames above
  (deep-compared for value identity in staging).
- `routing_scenarios.json` is fully recomputable from the released router
  artifacts and feature arrays (zero-mismatch re-derivation in staging; see
  its section above).
- `outcomes_per_task.json` is derived; its sources, field paths, and
  cross-checks are listed in its own `derivation_note` and above.
