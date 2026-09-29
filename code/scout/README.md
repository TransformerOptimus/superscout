# Running the searcher

This directory holds the harness that runs SuperScout-7B search episodes on SWE-bench
Pro tasks: the frozen tool-call dialect the model was trained on, the episode loop, a
Docker sandbox with the same tools and output formats, and a small CLI. It reproduces the
search episodes behind the paper's SWE-bench Pro results (266 Python tasks).

Contents:

| file | what it is |
|---|---|
| `dialect.py` | frozen dialect: system prompt, tool-call format, parser, context-overflow policy |
| `prompts/` | every model-visible harness string, as plain text |
| `engine.py` | stdlib client for a vLLM OpenAI-compatible server, sampling pinned |
| `sandbox.py` | Docker sandbox and tool executors (`bash`, `str_replace_editor`) |
| `episode.py` | the episode loop |
| `tasks.py` | builds a task from a public SWE-bench Pro row |
| `score.py` | parses a handoff (`<files>`, `<repro>`) and scores localization |
| `run_local.py` | CLI: one or more tasks, end to end |
| `serving_config.json` | model pin, server flags, sampling, loop constants |

## Requirements

- A GPU server for the model. The reported runs used one 48 GB GPU serving 20 concurrent
  episodes. The weights are about 15 GB in bf16.
- Docker. The SWE-bench Pro images are `linux/amd64`; on an arm64 host they run under
  emulation (`--platform linux/amd64` is passed for you). Each task has its own image;
  the ones we checked take 1.4 to 3.0 GB of disk each.
- Python 3.9+ with `datasets` (task rows) and `transformers` (the tokenizer used for
  context fitting, `Qwen/Qwen2.5-Coder-7B-Instruct`, fetched on first use).

## 1. Serve the model

```
pip install vllm==0.10.2 transformers==4.56.2
python -m vllm.entrypoints.openai.api_server \
  --model SuperAGI/SuperScout-7B --revision 8249e92f2328d8a23f5daf5da08872a0c0e7377b \
  --served-model-name SuperScout-7B \
  --dtype bfloat16 --max-model-len 32768 --enforce-eager --generation-config vllm \
  --host 127.0.0.1 --port 8000
```

These are the flags the reported runs used. The original served a local download of the
same revision, so `--revision` is the only addition (in this vLLM version it also pins the
tokenizer and chat template). Prefix caching is on by default in this vLLM version; each
episode sends its own `cache_salt` (`<instance_id>-<8 hex>`) so cached prefixes are never
shared across episodes.

**Sampling is pinned client-side.** Every request carries temperature 0.9, max_tokens
2048, stop `<|im_end|>`, top_p 1.0, top_k -1, repetition_penalty 1.0, and zero presence
and frequency penalties (`engine.py`). Do not rely on the checkpoint's
`generation_config.json`: it holds the base model's chat defaults (temperature 0.7,
top_p 0.8, top_k 20, repetition_penalty 1.1), which is not the evaluated policy.
`--generation-config vllm` keeps the server from applying them.

## 2. Run one task

```
python code/scout/run_local.py --endpoint http://127.0.0.1:8000 \
  --instance-id instance_ansible__ansible-e64c6c1ca50d7d26a8e7747d8eb87642e767cd74-v0f01c69f1e2528b935359cfe578530722bca2c59 \
  --out out/
```

For each `--instance-id` (repeatable) the runner:

1. loads the task row from `ScaleAI/SWE-bench_Pro` at revision `7ab5114912ba...`, the
   original 731-task release the paper used. The dataset's current default config is a
   revised 642-task set that rewrites most task texts and drops 29 of the 266 tasks, so
   the revision matters;
2. composes the task text (problem statement, requirements, new interfaces; see `tasks.py`);
3. pulls `jefzda/sweap-images:<dockerhub_tag>` and starts it with its entrypoint cleared
   and `sleep infinity` as the command;
4. runs one episode (up to 40 turns) and writes `out/<instance_id>.json`;
5. prints the handoff.

Every tool command runs as `bash -c` under a guarded environment prefix
(`sandbox.TOOL_ENV`) followed by `cd /app`. The prefix was part of the model's tool shell
in the reported runs; `--no-tool-env` removes it, which breaks reproduction on ansible
tasks (a stale installed copy of `ansible` shadows the repo).

## 3. What comes out

The episode record holds the per-turn trace (`turns`), `final_request` (the exact message
list of the last request), `final_response`, `outcome`, `handoff_type`, `sampling`,
`cache_salt`, and localization scores against the gold patch's files (`find_rate`).

`handoff_type` is `spontaneous` when the model emitted the handoff on its own and
`forced` when it did not within 40 turns and answered the one extra forced-handoff
request (`prompts/forced_handoff_nudge.txt`). Report the two separately.

A real handoff from the evaluation (`data/receipts/handoffs_stripped.jsonl`):

```
<files>
lib/ansible/modules/unarchive.py
</files>
<repro>
command: cd /app && python test_timestamp.py
reproduced: true
observed: OBSERVATION:
Error: time data '19800000.000000' does not match format '%Y%m%d.%H%M%S'
</repro>
<notes>
Search highlighted lib/ansible/modules/unarchive.py, showing sections around Windows zip warnings, zipinfoflag handling, and a line where the timestamp string “19800000.000000” is parsed with the format “%Y%m%d.%H%M%S”. Repeated view‑replace/editor steps yielded no additional relevant code. A temporary test script /app/test_timestamp.py was created and run; execution fails with the same “time data ... does not match format …” error. No build or environment issues observed.
</notes>
STOP
```

`score.extract_files` and `score.extract_repro` parse it.

## 4. Gate the handoff before a fixer sees it

Repro claims are often wrong (249 of 266 handoffs claimed a reproduction; 50 were
genuine). Pass each handoff through the verify-then-strip gate in `code/gate/`, with
`exec_fn` running in a fresh container of the same task image:

```python
import json, sys
sys.path[:0] = ["code/scout", "code/gate"]
import sandbox, verify_strip

rec = json.load(open("out/<instance_id>.json"))
sb = sandbox.DockerSandbox(rec["instance_id"], rec["image_ref"])
try:
    handoff, record = verify_strip.verify_and_strip(
        rec["final_response"], lambda cmd: sb.exec(cmd, timeout=300))
finally:
    sb.close()
```

Many repro commands run a script the searcher wrote during its episode (like
`test_timestamp.py` above), which does not exist in a fresh container. The evaluation
first replayed the episode's file-writing actions (editor `create` / `str_replace` and
file-writing `bash` commands, recovered from `turns[i].raw` with `dialect.parse_assistant`)
into the fresh container, then ran the gate. That replay step is not included here.

## Decode sampled, keep the format

Greedy decoding and a mismatched tool format both produce exploration without
commitment: the model searches but rarely emits a handoff. On the 450-task held-out exam
(Section 5.2 of the paper), greedy decoding finds the right files at a rate of 0.110
against 0.306 for one sampled draw at temperature 0.9, and almost all of the gain is
commitment (the handoff emission rate rises 3.4x). Keep the pinned sampling and the
prompts and tool format in this directory unchanged.

## Fidelity to the reported runs

The original episodes ran on hosted sandboxes, not Docker. We replayed three recorded
episodes (6, 9 and 22 turns; openlibrary and qutebrowser) through this harness on the
public images, feeding the recorded model turns back in place of the model. The system
prompt and task message matched exactly in all three, and 67 of 74 messages of the final
request matched byte for byte. Every mismatch came from the filesystem, not the harness:
the order in which `grep -r`, `find` and pytest enumerate directory entries (which also
changes what a following `head` keeps), directory sizes and mode bits in `ls -la`, and
file modification times seen by `find -newer`. Expect these to differ between container
runtimes. Other known differences:

- A command that outlives its client-side read bound (timeout plus 30 s) returns
  `[read timeout]`; the original backend used a different string for this case, which
  never occurred in the reported runs. Commands killed by the in-container `timeout`
  (default 60 s, max 120 s) return the same `[command killed: exceeded Ns timeout]` as
  before.
- A file written with `create` travels as one base64 command argument. The original
  backend rejected arguments over 64 KB; Linux rejects single arguments over 128 KB.
- The episode wall-clock budget (1200 s) is enforced between turns only; the original
  sandbox was also destroyed at that point.
- CPU and memory are unlimited unless you pass `--cpus` / `--memory`; the original tasks
  had 1 to 4 CPUs and 5 to 30 GiB.

## Tests

```
python -m pytest code/scout/tests
```

Offline (no GPU, Docker, or network). They check parsing of 20 recorded assistant turns,
byte equality of every prompt string and the sampling payload with the original harness,
and that the episode loop, run on scripted turns against a fake sandbox, sends exactly
the requests the original loop sent (including observation capping, context truncation,
malformed-turn recovery and forced emission).
