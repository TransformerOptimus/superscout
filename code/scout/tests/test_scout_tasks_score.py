# SPDX-License-Identifier: Apache-2.0
"""Task-text composition from dataset rows, and handoff parsing."""
import json
from pathlib import Path

import score
import tasks

TURNS = json.loads((Path(__file__).resolve().parent / "fixtures" / "recorded_turns.json")
                   .read_text(encoding="utf-8"))["turns"]


def test_json_decode_field_round_trip_rule():
    assert tasks.json_decode_field('"line one\\nline two"') == ("line one\nline two", True)
    assert tasks.json_decode_field("plain text") == ("plain text", False)
    assert tasks.json_decode_field('["a"]') == ('["a"]', False)          # not a string literal
    assert tasks.json_decode_field('"a\\u0041"') == ('"a\\u0041"', False)  # re-encode differs


def test_task_from_row():
    row = {"instance_id": "instance_x__y-abc", "repo": "x/y", "repo_language": "python",
           "dockerhub_tag": "x.y-x__y-abc", "base_commit": "deadbeef",
           "problem_statement": '"Bug:\\nit breaks"', "requirements": "- must work",
           "interface": "No new interfaces are introduced.",
           "patch": "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n"
                    "diff --git a/pkg/new.py b/pkg/new.py\n"}
    t = tasks.task_from_row(row)
    assert t["image_ref"] == "jefzda/sweap-images:x.y-x__y-abc"
    assert t["problem_statement"] == ("Bug:\nit breaks\n\nRequirements:\n- must work"
                                      "\n\nNew interfaces introduced:\nNo new interfaces are introduced.")
    assert t["gold_files"] == ["pkg/a.py", "pkg/new.py"]
    assert t["language"] == "python" and t["repo_path_in_image"] == "/app"


def test_extract_from_recorded_handoff():
    h = next(t["raw"] for t in TURNS if t["category"] == "handoff")
    files = score.extract_files(h)
    assert files and all(not f.startswith("<") for f in files)
    rep = score.extract_repro(h)
    assert rep["present"] and rep["command"] and rep["reproduced_claim"] == "true"


def test_score_files():
    s = score.score_files(["/app/pkg/a.py", "pkg/b.py"], ["pkg/a.py"], "/app")
    assert s["recall"] == 1.0 and s["precision"] == 0.5 and s["all_hit"]
    assert score.extract_files("no tags") == [] and score.extract_repro("x") == {"present": False}
