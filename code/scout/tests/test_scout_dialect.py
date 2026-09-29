# SPDX-License-Identifier: Apache-2.0
"""Dialect parsing of recorded turns, and the model-visible strings against the original harness."""
import hashlib
import json
from pathlib import Path

import pytest

import dialect
import prompts

FIX = Path(__file__).resolve().parent / "fixtures"
TURNS = json.loads((FIX / "recorded_turns.json").read_text(encoding="utf-8"))["turns"]
PARITY = json.loads((FIX / "loop_parity.json").read_text(encoding="utf-8"))
STRINGS = PARITY["strings"]


@pytest.mark.parametrize("t", TURNS, ids=[t["category"] + "-" + str(i) for i, t in enumerate(TURNS)])
def test_recorded_turn_parses_as_recorded(t):
    if t["recorded_kind"] == "malformed":
        with pytest.raises(Exception):
            dialect.parse_assistant(t["raw"])
        return
    p = dialect.parse_assistant(t["raw"])
    assert p["kind"] == t["recorded_kind"]
    if p["kind"] == "action":
        rec = t["recorded_action"]
        assert p["name"] == rec["name"]
        assert {k: str(v)[:200] for k, v in p["args"].items()} == rec["args"]


def test_fixture_covers_every_kind():
    cats = {t["category"] for t in TURNS}
    assert {"handoff", "think", "sre_view", "sre_view_range", "sre_create", "sre_str_replace",
            "bash", "bash_thought", "unknown_tool", "malformed_synthetic"} <= cats


def test_system_prompt_byte_identical():
    assert dialect.SYSTEM_PROMPT == STRINGS["system_prompt"]
    assert hashlib.sha256(dialect.SYSTEM_PROMPT.encode()).hexdigest() == STRINGS["system_prompt_sha256"]
    assert prompts.SYSTEM_PROMPT == dialect.SYSTEM_PROMPT


def test_issue_message_identical():
    ex = STRINGS["issue_message_example"]
    got = prompts.issue_message(ex["problem_statement"], ex["workdir"], ex["language"])
    assert got == ex["expected"]


def test_loop_strings_identical():
    # The original loop's malformed / think / nudge messages, as they appear in its final request.
    fr = PARITY["scripts"]["malformed_think"]["final_request"]
    contents = [m["content"] for m in fr]
    assert "OBSERVATION:\n" + prompts.THINK_OBSERVATION in contents
    assert prompts.FORCED_HANDOFF_NUDGE == contents[-1]
    malformed = [c for c in contents if c.startswith("OBSERVATION:\nERROR:\nYour last turn")]
    assert len(malformed) == 1
    try:
        dialect.parse_assistant(PARITY_MALFORMED)
    except Exception as e:  # noqa: BLE001
        assert "OBSERVATION:\n" + prompts.malformed_message(e) == malformed[0]
    else:
        raise AssertionError("expected a parse error")


PARITY_MALFORMED = ("Look.\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n"
                    "<parameter=path>/app/x.py</parameter>\n<parameter=view_range>[1, </parameter>\n"
                    "</function>")


def test_truncation_marker_and_constants():
    import episode
    assert dialect.TRUNCATION_MARKER == STRINGS["truncation_marker"]
    c = STRINGS["constants"]
    assert (episode.MAX_TURNS, episode.MAX_OBS_CHARS, episode.CHAR_BUDGET, episode.FIT_TARGET) == \
        (c["MAX_TURNS"], c["MAX_OBS_CHARS"], c["CHAR_BUDGET"], c["FIT_TARGET"])


def test_render_parse_roundtrip():
    args = {"command": "str_replace", "path": "/app/a.py", "old_str": "x = 1\n", "new_str": "x = 2\n"}
    text = "Fix it.\n" + dialect.render_action("str_replace_editor", args)
    p = dialect.parse_assistant(text)
    assert p == {"kind": "action", "thought": "Fix it.", "name": "str_replace_editor", "args": args}


def test_stray_function_tag_in_reasoning():
    text = ("I could call <function=bash> here but first view.\n<function=str_replace_editor>\n"
            "<parameter=command>view</parameter>\n<parameter=path>/app</parameter>\n</function>")
    p = dialect.parse_assistant(text)
    assert p["name"] == "str_replace_editor" and p["args"] == {"command": "view", "path": "/app"}
