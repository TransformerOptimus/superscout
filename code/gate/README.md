# The verify-then-strip gate

SuperScout-7B ends its search episode by emitting a structured handoff: a ranked
`<files>` list, a `<repro>` block (the reproduction claim), and free-prose `<notes>`,
terminated by the token `STOP`. The `<repro>` block claims that a command demonstrates
the bug before any fix (`reproduced: true`).

The gate (`verify_strip.py`) replays that claim in the task's sandbox at the base
commit, before any fixer sees the handoff:

- claim `reproduced: true` and the command actually fails: **verified**, the block is kept
  and the handoff is forwarded unchanged;
- claim `reproduced: true` but the command passes (exit code 0): the claim is **false**,
  the entire `<repro>` block is **stripped** from the forwarded handoff (a misleading
  "I reproduced it" would poison the fixer's prompt);
- claim false/unknown, no `<repro>` block, or no command: nothing to verify, the handoff
  passes through unchanged.

Every outcome is logged as a verification record (see the schema below), so the
claimed-versus-verified rate is measured as a side effect of gating.

## Decision rule

A reproduction that demonstrates a bug exits non-zero (a failing test, a raised
exception, an assertion). So: exit code != 0 means reproduced; exit code 0 means not
reproduced. An optional `failure_hints` flag (off by default) upgrades an exit-0 run to
"reproduced" when the output contains a failure substring (`Traceback`, `FAILED`,
`Error`, `AssertionError`, `Exception`, `FAIL `), for scripts that swallow their exit
code. Only a clean exit-0 pass strips.

## The injected `exec_fn` contract

The gate does not own a sandbox. Command execution is injected:

```python
cleaned_handoff, record = verify_strip.verify_and_strip(handoff_text, exec_fn)
```

- `exec_fn(command) -> (output, returncode)`: runs `command` in the task's sandbox at
  the base commit (pre-fix) and returns the captured output and the exit code.
- `exec_fn` is called at most once per handoff, and only when the block claims
  `reproduced: true` and contains a command.
- Time limits are the caller's responsibility: bind them inside `exec_fn`. The
  `timeout` keyword accepted by `verify_and_strip` is not used by the module itself.
- The record keeps the last 1000 characters of the command output (`output_tail`).

In deployment `exec_fn` wraps the task sandbox's exec (with the task's tool environment
activated); in the unit tests it is a stub.

## Handoff schema

`handoff.schema.json` (JSON Schema draft 2020-12) describes one released gated-handoff
record (one object per JSONL line): the task id, the handoff type
(`spontaneous`/`forced`), the post-gate handoff document, the gate's verification
record, and a diagnostic failure class. The handoff document itself is raw tagged text,
not JSON; the schema documents its `<files>`/`<repro>`/`<notes>`/`STOP` structure and
the format's 4KB size target (enforced upstream where handoffs are produced; the gate
does not reject oversized documents). The schema validates against the released
records.

## Running the tests

From the repository root:

```
python -m pytest code/gate/tests
```

The tests are self-contained (stub `exec_fn`, no sandbox, no network) and cover the
parser, all keep/strip/pass-through branches, the multi-line repro-command parsing, the
`failure_hints` upgrade, and the output-tail cap.
