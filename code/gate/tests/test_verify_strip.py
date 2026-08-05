"""Unit tests for the verify-then-strip gate (verify_strip.py).

The gate is exercised exactly as deployed: `exec_fn` is injected, so a stub
standing in for the sandbox lets every branch be covered without any sandbox.
Contract under test (module docstring + SPECS §3):
  - claim reproduced:true AND the command fails (rc != 0) -> VERIFIED, keep.
  - claim reproduced:true BUT the command passes (rc == 0) -> STRIP the <repro> block.
  - claim false/unknown, or no <repro> block, or no command -> pass through, never run.
"""
import verify_strip as vs

HANDOFF = (
    "<files>\n"
    "src/pkg/mod.py\n"
    "src/pkg/util.py\n"
    "</files>\n"
    "<repro>\n"
    "command: python -m pytest tests/test_mod.py -x\n"
    "reproduced: true\n"
    "observed: AssertionError: expected 2 got 3\n"
    "</repro>\n"
    "<notes>\n"
    "The bug lives in mod.py; util.py was a dead end.\n"
    "</notes>\n"
    "STOP"
)

NO_REPRO = "<files>\nsrc/pkg/mod.py\n</files>\n<notes>\nno repro found\n</notes>\nSTOP"


def exec_fail(cmd):
    return "Traceback (most recent call last):\nAssertionError", 1


def exec_pass(cmd):
    return "1 passed", 0


def never_called(cmd):
    raise AssertionError("exec_fn must not run for this handoff")


# ---------------------------------------------------------------- parse_handoff
def test_parse_extracts_command_and_claim():
    p = vs.parse_handoff(HANDOFF)
    assert p["has_repro"] is True
    assert p["command"] == "python -m pytest tests/test_mod.py -x"
    assert p["claim"] is True
    assert p["repro_block"].startswith("<repro>")


def test_parse_no_repro_block():
    p = vs.parse_handoff(NO_REPRO)
    assert p == {"has_repro": False, "repro_block": None, "command": None, "claim": None}


def test_parse_claim_false_and_yes():
    p = vs.parse_handoff("<repro>\ncommand: x\nreproduced: false\n</repro>")
    assert p["claim"] is False
    p = vs.parse_handoff("<repro>\ncommand: x\nreproduced: Yes\n</repro>")
    assert p["claim"] is True


def test_parse_multiline_command():
    # Regression guard for the documented parser fix: a multi-line `python -c "..."`
    # command must be captured whole, not truncated at the first newline.
    body = ('<repro>\ncommand: python -c "\nimport pkg\nassert pkg.f() == 1\n"\n'
            "reproduced: true\nobserved: AssertionError\n</repro>")
    p = vs.parse_handoff(body)
    assert p["command"].startswith('python -c "')
    assert p["command"].endswith('"')
    assert "assert pkg.f() == 1" in p["command"]


def test_parse_missing_command():
    p = vs.parse_handoff("<repro>\nreproduced: true\nobserved: boom\n</repro>")
    assert p["has_repro"] is True and p["command"] is None and p["claim"] is True


# ---------------------------------------------------------------- verify_and_strip
def test_true_claim_command_fails_kept():
    cleaned, rec = vs.verify_and_strip(HANDOFF, exec_fail)
    assert cleaned == HANDOFF
    assert rec["ran"] is True and rec["returncode"] == 1
    assert rec["verified"] is True and rec["stripped"] is False
    assert rec["reason"] == "verified_reproduces"


def test_true_claim_command_passes_stripped():
    cleaned, rec = vs.verify_and_strip(HANDOFF, exec_pass)
    assert rec["stripped"] is True and rec["verified"] is False
    assert rec["reason"] == "claim_false_repro_passes"
    assert "<repro>" not in cleaned and "reproduced:" not in cleaned
    # the other blocks survive untouched
    assert "<files>" in cleaned and "<notes>" in cleaned and cleaned.rstrip().endswith("STOP")
    assert "src/pkg/mod.py" in cleaned and "dead end" in cleaned


def test_no_repro_block_passthrough():
    cleaned, rec = vs.verify_and_strip(NO_REPRO, never_called)
    assert cleaned == NO_REPRO
    assert rec["ran"] is False and rec["stripped"] is False
    assert rec["reason"] == "no_repro_block"


def test_false_claim_never_run():
    h = HANDOFF.replace("reproduced: true", "reproduced: false")
    cleaned, rec = vs.verify_and_strip(h, never_called)
    assert cleaned == h
    assert rec["ran"] is False and rec["reason"] == "claim_not_true"


def test_no_command_never_run():
    h = ("<repro>\nreproduced: true\nobserved: boom\n</repro>\nSTOP")
    cleaned, rec = vs.verify_and_strip(h, never_called)
    assert cleaned == h
    assert rec["reason"] == "no_command_in_repro"


def test_failure_hints_off_by_default():
    # rc==0 with a scary-looking output still strips unless failure_hints=True.
    def swallowed(cmd):
        return "Traceback (most recent call last):\nAssertionError", 0
    _, rec = vs.verify_and_strip(HANDOFF, swallowed)
    assert rec["stripped"] is True
    _, rec = vs.verify_and_strip(HANDOFF, swallowed, failure_hints=True)
    assert rec["verified"] is True and rec["stripped"] is False


def test_output_tail_capped_at_1000_chars():
    def chatty(cmd):
        return "x" * 5000 + "END", 1
    _, rec = vs.verify_and_strip(HANDOFF, chatty)
    assert len(rec["output_tail"]) == 1000 and rec["output_tail"].endswith("END")
