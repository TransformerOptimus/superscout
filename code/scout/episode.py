# SPDX-License-Identifier: Apache-2.0
"""The SuperScout-7B search episode loop.

Per turn: send the message list, parse the reply with the frozen dialect, run the tool call in
the sandbox, append the (capped) observation as the next user turn. The episode ends on a
handoff (a turn containing <files> and ending in STOP) or after MAX_TURNS turns. If no handoff
was emitted, one extra forced-emission turn asks for it; such handoffs are tagged
handoff_type="forced" and must be reported separately from spontaneous ones.

The record written per episode includes `final_request` (the message list exactly as sent on the
last generate call) and `final_response` (the handoff text, or the last reply if none).
"""
from __future__ import annotations
import concurrent.futures as _cf
import json
import os
import time
import uuid

import dialect
import prompts
import score as scoring

MAX_TURNS = 40
MAX_OBS_CHARS = 8000                  # observation cap (head + tail kept)
CHAR_BUDGET = 90000                   # cheap gate before the token-exact context fit
MODEL_LEN = 32768
FIT_TARGET = MODEL_LEN - 2048 - 256   # leave room for the 2048-token completion
SETUP_TIMEOUT = 300                   # bound on container start + boot probes (seconds)
TOKENIZER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


def cap_obs(raw):
    if len(raw) <= MAX_OBS_CHARS:
        return raw
    h = MAX_OBS_CHARS // 2
    return raw[:h] + "\n...[%d chars truncated]...\n" % (len(raw) - MAX_OBS_CHARS) + raw[-h:]


def normalize_obs(raw):
    return raw if raw.startswith("OBSERVATION:") else "OBSERVATION:\n" + raw


_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return _TOKENIZER


def fit_context(messages, max_tokens=FIT_TARGET, tokenizer=None):
    """Token-exact truncation with the same policy used at training time (dialect.truncate_to_fit)."""
    flat = [{"role": m["role"], "content": m["content"], "loss": False} for m in messages]
    flat, _ = dialect.truncate_to_fit(flat, tokenizer or _tokenizer(), max_tokens=max_tokens)
    return [{"role": m["role"], "content": m["content"]} for m in flat]


def _bounded(make, timeout=SETUP_TIMEOUT):
    """Run make() under a wall-clock bound -> (sandbox, None) or (None, 'setup_timeout')."""
    ex = _cf.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(make)
    try:
        return fut.result(timeout=timeout), None
    except _cf.TimeoutError:
        return None, "setup_timeout"
    finally:
        ex.shutdown(wait=False)


def write_rec(out_dir, rec):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, rec["instance_id"] + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, default=str)
    os.replace(tmp, path)
    return path


def run_episode(task, engine, sandbox_factory, out_dir=None, max_turns=MAX_TURNS,
                tokenizer=None, cancel=None):
    """One search episode.

    task: dict from tasks.task_from_row. engine: object with generate(messages, cache_salt) -> str
    and a `sampling` dict. sandbox_factory(task) -> sandbox with ready/workdir/dispatch/close.
    Returns the episode record (also written to out_dir/<instance_id>.json when out_dir is set).
    """
    fit = lambda msgs, mt=FIT_TARGET: fit_context(msgs, max_tokens=mt, tokenizer=tokenizer)  # noqa: E731
    salt = task["instance_id"] + "-" + uuid.uuid4().hex[:8]          # per-episode prefix isolation
    rec = {"instance_id": task["instance_id"], "repo": task["repo"],
           "repo_key": task.get("repo_key", task["repo"]), "language": task.get("language", ""),
           "gold_files": task.get("gold_files", []), "image_ref": task.get("image_ref"),
           "workdir": task.get("repo_path_in_image", "/app"), "turns": [], "outcome": None,
           "t_start": time.time(), "cache_salt": salt,
           "sampling": engine.sampling,
           "final_request": None, "final_response": None}

    def _done(r):
        if out_dir:
            write_rec(out_dir, r)
        return r

    rs, setup_err = _bounded(lambda: sandbox_factory(task))
    if setup_err:
        rec["outcome"] = setup_err
        return _done(rec)
    if not getattr(rs, "ready", False):
        rec["outcome"] = getattr(rs, "scrub", {}).get("status", "setup_failed")
        rec["scrub"] = getattr(rs, "scrub", None)
        rs.close()
        return _done(rec)
    rec["workdir"] = rs.workdir
    rec["scrub"] = getattr(rs, "scrub", None)

    messages = [{"role": "system", "content": dialect.SYSTEM_PROMPT},
                {"role": "user", "content": prompts.issue_message(task["problem_statement"],
                                                                  rs.workdir,
                                                                  task.get("language", ""))}]
    final = None            # handoff text if one is emitted
    last_response = None    # whatever the model last emitted
    last_request = None     # snapshot of the message list as sent on the last generate call
    try:
        for step in range(max_turns):
            if cancel is not None and cancel.is_set():
                rec["outcome"] = rec["outcome"] or "cancelled"
                rec["cancelled_at_turn"] = step
                break
            # Episode wall-clock budget (the sandbox lifetime in the original runs).
            if time.time() - getattr(rs, "created", time.time()) >= getattr(rs, "timeout", 10 ** 9):
                rec["outcome"] = rec["outcome"] or "wall_timeout"
                break
            last_request = [dict(m) for m in messages]
            t0 = time.time()
            try:
                text = engine.generate(messages, cache_salt=salt)
            except Exception as e:
                # HTTP 400 = context overflow: one aggressive trim and a same-turn retry.
                if "400" in str(e):
                    messages = fit(messages, int(FIT_TARGET * 0.8))
                    rec["overflow_trims"] = rec.get("overflow_trims", 0) + 1
                    last_request = [dict(m) for m in messages]
                    try:
                        text = engine.generate(messages, cache_salt=salt)
                    except Exception as e2:
                        rec["outcome"] = "model_error"; rec["error"] = str(e2)[:200]; break
                else:
                    rec["outcome"] = "model_error"; rec["error"] = str(e)[:200]; break
            last_response = text
            try:
                parsed = dialect.parse_assistant(text)
            except Exception as e:
                turn = {"step": step, "kind": "malformed", "latency": round(time.time() - t0, 2),
                        "raw": text}
                obs = prompts.malformed_message(e)
                turn["obs_head"] = obs[:200]; rec["turns"].append(turn)
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": normalize_obs(obs)})
                continue
            turn = {"step": step, "kind": parsed["kind"], "latency": round(time.time() - t0, 2),
                    "raw": text}
            if parsed["kind"] == "handoff":
                rec["turns"].append(turn); final = text; rec["outcome"] = "handoff"
                break
            if parsed["kind"] == "think":
                obs = prompts.THINK_OBSERVATION
            else:
                turn["action"] = {"name": parsed["name"],
                                  "args": {k: str(v)[:200] for k, v in parsed.get("args", {}).items()}}
                try:
                    obs = rs.dispatch(parsed["name"], parsed.get("args", {}))
                except Exception as e:
                    obs = "ERROR:\n" + str(e)[:200]
            obs = cap_obs(obs)
            turn["obs_head"] = obs[:200]
            if len(obs) > 200:
                turn["obs_tail"] = obs[-300:]
            rec["turns"].append(turn)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": normalize_obs(obs)})
            if sum(len(m["content"]) for m in messages) > CHAR_BUDGET:
                messages = fit(messages)
        if rec["outcome"] is None:
            rec["outcome"] = "no_handoff"

        # Forced emission: turn budget spent (or model error) with no handoff -> one last request
        # for the handoff. Tagged "forced"; never pool with spontaneous handoffs.
        rec["handoff_type"] = "spontaneous" if final else None
        if final is None and rec["outcome"] == "model_error":
            messages = fit(messages, int(FIT_TARGET * 0.8))
        if final is None and rec["outcome"] in ("no_handoff", "model_error") and not (cancel and cancel.is_set()):
            messages.append({"role": "user", "content": prompts.FORCED_HANDOFF_NUDGE})
            last_request = [dict(m) for m in messages]
            try:
                text = engine.generate(messages, cache_salt=salt)
                messages.append({"role": "assistant", "content": text})
                parsed = dialect.parse_assistant(text)
                if parsed["kind"] == "handoff":
                    final, last_response = text, text
                    rec["outcome"], rec["handoff_type"] = "handoff", "forced"
                    rec["turns"].append({"step": len(rec["turns"]), "kind": "handoff_forced"})
                else:
                    rec["forced_emission_refused"] = parsed["kind"]
            except Exception as e:
                rec["forced_emission_error"] = str(e)[:200]

        rec["final_request"] = last_request
        rec["final_response"] = final if final is not None else last_response
        rec["gold_at_base"] = getattr(rs, "gold_at_base", [])
        if final:
            preds = scoring.extract_files(final)
            rec["find_rate"] = scoring.score_files(preds, rs.gold_at_base, rs.workdir)
            rec["repro"] = scoring.extract_repro(final)
        rec["n_turns"] = len(rec["turns"])
        rec["wall"] = round(time.time() - rec["t_start"], 1)
    finally:
        rs.close()
    return _done(rec)
