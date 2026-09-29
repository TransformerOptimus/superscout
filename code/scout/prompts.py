# SPDX-License-Identifier: Apache-2.0
"""Model-visible harness strings, loaded byte-exact from prompts/*.txt.

Files are read without newline translation. Templates use {{name}} slots, filled in a single
pass so that text substituted into one slot is never re-scanned for other slots.
"""
from __future__ import annotations
import os
import re

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_SLOT = re.compile(r"\{\{(\w+)\}\}")

# Language names used in the issue message (unknown keys pass through unchanged).
LANGNAME = {"go": "Go", "python": "Python", "typescript": "TypeScript", "javascript": "JavaScript"}


def load(name: str) -> str:
    with open(os.path.join(_DIR, name), encoding="utf-8", newline="") as f:
        return f.read()


def fill(template: str, **values) -> str:
    return _SLOT.sub(lambda m: values[m.group(1)], template)


SYSTEM_PROMPT = load("system_prompt.txt")
ISSUE_TEMPLATE = load("issue_template.txt")
MALFORMED_TEMPLATE = load("malformed_turn.txt")
THINK_OBSERVATION = load("think_observation.txt")
FORCED_HANDOFF_NUDGE = load("forced_handoff_nudge.txt")


def issue_message(problem_statement: str, workdir: str, language: str) -> str:
    """First user message: the task text wrapped in the trained issue template."""
    return fill(ISSUE_TEMPLATE, workdir=workdir, language=LANGNAME.get(language, language),
                problem_statement=problem_statement)


def malformed_message(error: Exception) -> str:
    """Observation sent back when an assistant turn cannot be parsed."""
    return fill(MALFORMED_TEMPLATE, error=str(error)[:120])
