"""verify_strip.py — L15 / SPECS §3: verify a handoff's <repro> claim, strip it if it doesn't hold.

A frozen-dialect handoff has three tag blocks:
    <files> … </files>     ranked candidate files, one per line
    <repro> … </repro>     command: <cmd> / reproduced: <bool> / observed: <output>
    <notes> … </notes>     key findings

The handoff CLAIMS the repro command demonstrates the bug (reproduced: true) BEFORE any fix. This
module runs that command in the task sandbox at the base commit (pre-injection) and checks the claim:
  - claim reproduced:true AND the command actually fails  -> VERIFIED, keep the block.
  - claim reproduced:true BUT the command PASSES (rc 0)   -> claim is FALSE -> STRIP the <repro>
    block (a misleading "I reproduced it" would poison the fixer's prompt).
  - claim reproduced:false / no repro block               -> nothing to verify, pass through.
The claim-vs-verified outcome is logged (it doubles as the first real repro-validity measurement).

"Fails as claimed" heuristic: a repro that reproduces a bug exits NON-ZERO (a failing test / a
raised exception / an assertion). rc != 0 => reproduced. rc == 0 => not reproduced. Optional
failure-substring hints (traceback, FAILED, Error) can upgrade an rc==0 to "reproduced" for scripts
that swallow their exit code; off by default, exposed via `failure_hints`.

exec_fn is injected: `exec_fn(command) -> (output, returncode)`. In phase 3 this is
LabelRunSandbox.exec (activate=True) at base_commit. Unit-tested now (phase 1) with a stub — the
real-sandbox path gets its shakedown at phase-3 start (L16).
"""
from __future__ import annotations
import re

_REPRO_RE = re.compile(r"<repro>(.*?)</repro>\s*", re.S)
# command: value spans until the next repro field (reproduced:/observed:) or end of block — a repro
# command can be a MULTI-LINE `python -c "..."`; a single-line regex truncated it to `python -c "`
# (unterminated quote -> bash EOF error). DOTALL, non-greedy, stop at the next field.
_CMD_RE = re.compile(r"command:[ \t]*(.*?)(?:\n[ \t]*(?:reproduced|observed)[ \t]*:|\Z)", re.S)
_REPRODUCED_RE = re.compile(r"^\s*reproduced:\s*(\w+)", re.M)
_FAIL_HINTS = ("Traceback", "FAILED", "Error", "AssertionError", "Exception", "FAIL ")


def parse_handoff(text: str) -> dict:
    """Extract the repro block + its command/claim. Missing pieces come back as None."""
    m = _REPRO_RE.search(text)
    if not m:
        return {"has_repro": False, "repro_block": None, "command": None, "claim": None}
    body = m.group(1)
    cmd_m = _CMD_RE.search(body)
    claim_m = _REPRODUCED_RE.search(body)
    claim = None
    if claim_m:
        claim = claim_m.group(1).strip().lower() in ("true", "yes")
    return {
        "has_repro": True,
        "repro_block": m.group(0),      # includes the tags + trailing ws (what strip removes)
        "repro_span": m.span(),
        "command": cmd_m.group(1).strip() if cmd_m else None,
        "claim": claim,
    }


def _fails_as_claimed(output: str, rc: int, failure_hints: bool) -> bool:
    if rc != 0:
        return True
    if failure_hints and any(h in (output or "") for h in _FAIL_HINTS):
        return True
    return False


def verify_and_strip(handoff_text: str, exec_fn, *, timeout: int = 300,
                     failure_hints: bool = False) -> tuple[str, dict]:
    """Verify the handoff's repro; return (cleaned_handoff, record).
    exec_fn(command) -> (output, returncode). If claim holds, handoff is returned unchanged.
    If the claim fails, the <repro> block is stripped from the returned handoff."""
    p = parse_handoff(handoff_text)
    rec = {"has_repro": p["has_repro"], "command": p["command"], "claim": p["claim"],
           "ran": False, "returncode": None, "verified": None, "stripped": False, "reason": None}

    if not p["has_repro"]:
        rec["reason"] = "no_repro_block"
        return handoff_text, rec
    if p["claim"] is not True:
        # claim is false/unknown -> nothing to falsify; keep as-is but log.
        rec["reason"] = "claim_not_true"
        return handoff_text, rec
    if not p["command"]:
        rec["reason"] = "no_command_in_repro"
        return handoff_text, rec

    out, rc = exec_fn(p["command"])
    rec["ran"] = True
    rec["returncode"] = rc
    rec["output_tail"] = (out or "")[-1000:]
    verified = _fails_as_claimed(out, rc, failure_hints)
    rec["verified"] = verified
    if verified:
        rec["reason"] = "verified_reproduces"
        return handoff_text, rec

    # claim false -> strip the repro block
    a, b = p["repro_span"]
    cleaned = handoff_text[:a] + handoff_text[b:]
    rec["stripped"] = True
    rec["reason"] = "claim_false_repro_passes"
    return cleaned, rec
