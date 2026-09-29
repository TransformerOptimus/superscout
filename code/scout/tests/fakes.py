# SPDX-License-Identifier: Apache-2.0
"""Deterministic fake sandbox, tokenizer, and scripted model turns for the offline loop tests.

fixtures/loop_parity.json holds what the original episode loop produced for these same scripts."""
import time

class FakeTokenizer:
    """Counts one token per two characters. Deterministic stand-in for the real tokenizer."""
    def __call__(self, text, add_special_tokens=False, **_):
        return {"input_ids": [0] * ((len(text) + 1) // 2)}

class FakeWorld:
    def __init__(self):
        self.ready, self.workdir, self.gold_at_base = True, "/app", ["pkg/mod.py"]
        self.scrub = {"status": "ok"}
        self.created, self.timeout = time.time(), 10 ** 9
        self.calls = []
    def dispatch(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "bash":
            cmd = args.get("command", "")
            if cmd.startswith("emit "):
                n = int(cmd.split()[1]); out, i = [], 0
                while sum(len(x) for x in out) < n:
                    out.append("line %05d of emitted output\n" % i); i += 1
                return "".join(out)[:n]
            if cmd == "raise":
                raise RuntimeError("sandbox went away")
            return "ran: " + cmd + "\n"
        if name == "str_replace_editor":
            return "Here's the files and directories up to 2 levels deep in %s, excluding hidden items:\n%s\n%s/pkg" % (args.get("path"), args.get("path"), args.get("path"))
        return "ERROR:\nUnknown tool " + name
    def close(self):
        pass

def act(cmd, thought="Checking."):
    return thought + "\n<function=bash>\n<parameter=command>" + cmd + "</parameter>\n</function>"

HANDOFF = ("<files>\npkg/mod.py\n</files>\n<repro>\ncommand: python -c 'import pkg'\nreproduced: false\n"
           "observed: nothing\n</repro>\n<notes>\nfake\n</notes>\nSTOP")

SCRIPTS = {
  # no spontaneous handoff: malformed, think, view, a dispatch exception, large observations that
  # trigger capping and context truncation, then the forced-emission turn
  "forced": {"max_turns": 18, "turns":
      [act("emit 100"),
       "Look.\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/app/x.py</parameter>\n<parameter=view_range>[1, </parameter>\n</function>",
       "I should think about where the bug lives before acting.",
       "<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/app</parameter>\n</function>",
       act("raise")]
      + [act("emit 12000", "Big output %d." % i) for i in range(13)]
      + [HANDOFF]},
  "spontaneous": {"max_turns": 40, "turns": [act("ls"), act("emit 250"), HANDOFF]},
  "forced_refused": {"max_turns": 2, "turns": [act("ls"), act("pwd"), "I am not done yet."]},
  # malformed turn, then a think turn, then the forced-emission turn (no truncation)
  "malformed_think": {"max_turns": 2, "turns":
      ["Look.\n<function=str_replace_editor>\n<parameter=command>view</parameter>\n<parameter=path>/app/x.py</parameter>\n<parameter=view_range>[1, </parameter>\n</function>",
       "I should think about where the bug lives before acting.", HANDOFF]},
}
TASK = {"instance_id": "fake__task-1", "repo": "fake/repo", "language": "python",
        "gold_files": ["pkg/mod.py"], "image_ref": "none", "repo_path_in_image": "/app",
        "problem_statement": "Calling {{workdir}} f() with {braces} fails.\n\nRequirements:\n- x\n\nNew interfaces introduced:\nNone"}
