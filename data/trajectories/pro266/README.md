# SuperScout-7B search episodes (SWE-bench Pro, 266 tasks)

`episodes.jsonl` holds the 266 SuperScout-7B search episodes that produced the
handoffs in `data/receipts/handoffs_stripped.jsonl`. There is one JSON object
per line and one episode per task on the 266-task Python slice of SWE-bench Pro
(ansible 96, openlibrary 91, qutebrowser 79). The join key is `instance_id`,
which equals `task_id` in `data/receipts/`.

## How the episodes were produced

The harness is in `code/scout/`. Each task got one episode, drawn once with the
pinned sampling settings recorded in `sampling` (temperature 0.9, top_p 1.0,
top_k -1, no penalties, max_tokens 2048, stop `<|im_end|>`). Each episode ran
inside the task's public SWE-bench Pro image. On every turn the model wrote
its reasoning and then made at most one tool call (`bash` or
`str_replace_editor`). The tool output came back as an `OBSERVATION:` user
message, capped at 8000 characters (head and tail kept, middle elided with a
marker). An episode ended when the model emitted a handoff (`<files>`,
`<repro>`, `<notes>`, then `STOP`). If the 40-turn budget ran out first, one
more user message asked for the handoff immediately, and the model's reply was
recorded as a `forced` handoff. Of the 266 episodes, 206 handed off
spontaneously and 60 were forced.

## Schema

| field | description |
|---|---|
| `instance_id` | SWE-bench Pro instance id (join key) |
| `repo` | upstream repository, e.g. `ansible/ansible` |
| `repo_family` | `ansible` \| `openlibrary` \| `qutebrowser` |
| `image_ref` | public SWE-bench Pro Docker image tag the episode ran in |
| `workdir` | repository path inside the image (`/app`) |
| `handoff_type` | `spontaneous` \| `forced` |
| `n_turns` | number of entries in `turns` |
| `context_trimmed` | `true` if older turns were dropped from `messages` to fit the context window (see below) |
| `max_turns` | turn budget (40) |
| `sampling` | the pinned decoding parameters |
| `messages` | the full conversation as the model saw it on its final call, followed by the model's final message: system prompt, task message, alternating assistant turns and `OBSERVATION:` user messages, the forced-handoff request (forced episodes only), and the handoff. Roles and contents are verbatim. |
| `turns` | one entry per assistant turn, in order (see below) |
| `handoff` | the handoff text exactly as emitted, before verify-then-strip (equal to the last message in `messages`) |

Each entry in `turns` has these fields:

- `step`: turn index, starting at 0.
- `kind`: `action` (tool call), `think` (reasoning with no tool call),
  `malformed` (the turn could not be parsed), `handoff` (spontaneous handoff)
  or `handoff_forced` (the reply to the forced-handoff request).
- `raw`: the assistant text for the turn, verbatim. For `handoff_forced`, this
  is the forced handoff text.
- `action`: `{name, args}` for `action` turns only. Each argument value is
  truncated to 200 characters. The complete call is in `raw`.
- `obs_head` / `obs_tail`: present only when `context_trimmed` is true. These
  hold the first 200 and last 300 characters of the capped observation for the
  turn (see below).

## Context trimming caveat

When a conversation grew past the model's 32k-token window, the harness dropped
the oldest exploration turns and put a single marker message in their place:
`[... earlier exploration turns were truncated to fit the context window ...]`.
The system prompt and the task message were always kept. This happened in 9
episodes (`context_trimmed: true`). For those episodes, `messages` is exactly
what the model saw last, but it does not contain the dropped early turns. The
`raw` text of every assistant turn is still in `turns`. The full observations
for the dropped turns are not available, which is why `obs_head` and
`obs_tail` are kept for these episodes.

In the other 257 episodes, `messages` is the complete conversation. Every
`turns[i].raw` appears as an assistant message in the same order, and every
observation is present in full, up to the 8000-character cap.

## Handoffs: raw and stripped

`handoff` is the raw handoff. Before injection into the fixer stage, each
handoff's repro claim was re-run and removed if it did not verify. The
stripped, injection-ready text is `handoff_text_final` in
`data/receipts/handoffs_stripped.jsonl`. Per-task verification details are in
`data/receipts/verify_records/`, and the `handoff_text` field there is
identical to `handoff` here. Localization scores against the gold-patch files
are in `data/receipts/localization.json`.

## Not included

Internal run bookkeeping is omitted: timestamps, wall-clock and per-turn
latency, a random per-episode cache-isolation salt, and sandbox setup
metadata. So are gold-derived evaluation labels. Localization scoring is
released separately in `data/receipts/localization.json`.

## License

CC-BY-4.0, covered by `data/LICENSE`. The observations contain excerpts of
the upstream repositories (ansible, openlibrary, qutebrowser), which remain
under their own licenses.
