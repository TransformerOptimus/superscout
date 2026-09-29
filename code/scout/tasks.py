# SPDX-License-Identifier: Apache-2.0
"""Build an episode task from a public SWE-bench Pro row.

Task text is the three-field composite used by the SWE-bench Pro reference harness:

    problem_statement + "\\n\\nRequirements:\\n" + requirements
                      + "\\n\\nNew interfaces introduced:\\n" + interface

Before composing, each field gets one mechanical cleanup: some rows store a field as a
JSON-encoded string (wrapped in quotes, with literal backslash-n escapes). Such a field is decoded
only if json.loads yields a string AND re-encoding reproduces the stored text byte-for-byte.
Anything else is left untouched.

Dataset pin: the reported runs used the original 731-task release of ScaleAI/SWE-bench_Pro.
The dataset's default config was later replaced by a revised 642-task set whose task text differs,
so rows are loaded at the pinned revision below.
"""
from __future__ import annotations
import json
import re

HF_DATASET = "ScaleAI/SWE-bench_Pro"
HF_SPLIT = "test"
HF_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"   # original 731-task release
IMAGE_PREFIX = "jefzda/sweap-images:"
WORKDIR = "/app"
LANGUAGE = {"python": "python", "go": "go", "js": "javascript", "ts": "typescript"}

DECODABLE_FIELDS = ("problem_statement", "requirements", "interface")
_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.M)


def json_decode_field(value):
    """(cleaned_value, was_decoded) under the round-trip rule in the module docstring."""
    if not isinstance(value, str):
        return value, False
    s = value.strip()
    if not s:
        return value, False
    try:
        decoded = json.loads(s)
    except Exception:
        return value, False
    if not isinstance(decoded, str):
        return value, False
    if json.dumps(decoded, ensure_ascii=False) == s or json.dumps(decoded, ensure_ascii=True) == s:
        return decoded, True
    return value, False


def compose_task_text(problem_statement, requirements, interface):
    return (f"{problem_statement}\n\nRequirements:\n{requirements}"
            f"\n\nNew interfaces introduced:\n{interface}")


def task_text(row):
    cleaned = {f: json_decode_field(row.get(f) or "")[0] for f in DECODABLE_FIELDS}
    return compose_task_text(cleaned["problem_statement"], cleaned["requirements"],
                             cleaned["interface"])


def gold_files_from_patch(patch):
    """Files changed by the gold patch. Used only to score localization, never shown to the model."""
    files = []
    for _a, b in _DIFF_GIT.findall(patch or ""):
        if b and b != "/dev/null" and b not in files:
            files.append(b)
    return files


def load_row(instance_id, revision=HF_REVISION):
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, revision=revision)
    for r in ds:
        if r["instance_id"] == instance_id:
            return dict(r)
    raise KeyError("instance_id not found in %s@%s: %s" % (HF_DATASET, revision[:12], instance_id))


def task_from_row(row):
    """The task dict consumed by episode.run_episode."""
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "repo_key": row["repo"],
        "language": LANGUAGE.get(row.get("repo_language", "python"), row.get("repo_language", "")),
        "image_ref": IMAGE_PREFIX + row["dockerhub_tag"],    # the column verbatim, never re-derived
        "problem_statement": task_text(row),
        "gold_files": gold_files_from_patch(row.get("patch", "")),
        "base_commit": row["base_commit"],
        "repo_path_in_image": WORKDIR,
    }
