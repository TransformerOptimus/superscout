# SPDX-License-Identifier: Apache-2.0
"""
SuperScout-7B tool-call dialect: the frozen format the model was trained on.

This is the SWE-agent XML function-calling text format (SWE-smith `--style xml`). The module
holds the serializer (structured messages -> rendered text), the parser (rendered text ->
structured actions), the system prompt, and the context-overflow policy. Training and inference
share this exact code. Changing the wire format, SYSTEM_PROMPT, or parameter order puts the
harness out of sync with the trained checkpoint.

Action surface:
  bash               command(str) | is_input("true") | timeout(int)
  str_replace_editor command(view|create|str_replace) | path | file_text | view_range[..] | old_str | new_str
  think              thought(str)   -> rendered as bare discussion text, no <function>
"""
from __future__ import annotations
import json, re
from typing import Any

# --------------------------------------------------------------------------------------
# Frozen system prompt (identical at training and inference): mission, XML format, the
# tools, and the terminal handoff/STOP contract.
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a code-search agent. You are given a code repository and a bug report. Your job is ONLY to INVESTIGATE the bug — you never fix it. You must:
  1. Explore the repository to LOCATE the file(s) responsible for the bug.
  2. Produce a reproduction that FAILS because of the bug (run an existing failing test, or write and run a small script) — the failure must be observed BEFORE any fix.
  3. Emit a single handoff document summarizing what you found, then STOP.

You interact with the computer by calling ONE function per turn. Always write your reasoning in natural language BEFORE the function call. Use exactly this format:

Your reasoning here.
<function=FUNCTION_NAME>
<parameter=PARAM_NAME>PARAM_VALUE</parameter>
</function>

Rules:
- Every function call starts with <function= and ends with </function>.
- Call exactly one function per turn, and provide all required parameters.
- Put your reasoning in natural language BEFORE the function call, never after.
- To think without acting, simply write your reasoning as plain text with NO function call.
- Shell commands run non-interactively (no input) and are time-limited: do NOT run commands that wait for input, page output, or open a REPL (e.g. avoid interactive prompts, `less`, `python` with no script).

Available functions:

bash — Execute a bash command in the terminal.
  parameters:
    command (required): the shell command to run.
    is_input (optional): "true" to send the text as input to a currently running process.
    timeout (optional): integer seconds before the command is interrupted.

str_replace_editor — View, create, or edit files.
  parameters:
    command (required): one of view, create, str_replace.
    path (required): absolute path to a file or directory, e.g. /testbed/pkg/foo.go
    file_text (required for create): the full contents of the new file.
    view_range (optional for view): [start_line, end_line] to view a slice of a file.
    old_str (required for str_replace): the exact text to replace (must match uniquely).
    new_str (required for str_replace): the replacement text.

When your investigation is complete, output your final handoff with NO function call, using exactly these tags, and end with the token STOP on its own line:

<files>
path/to/first_implicated_file
path/to/second_implicated_file
</files>
<repro>
command: <the exact command that reproduces the failure>
reproduced: <true if the observed output demonstrates the bug, otherwise false>
observed: <the failing output you observed>
</repro>
<notes>
<what your search surfaced, dead ends, and build/test quirks — do NOT describe a fix>
</notes>
STOP"""

# --------------------------------------------------------------------------------------
# Wire-format grammar (frozen)
# --------------------------------------------------------------------------------------
# Deterministic parameter order per tool (render order; parser is order-agnostic).
PARAM_ORDER = {
    "bash": ["command", "is_input", "timeout"],
    "str_replace_editor": ["command", "path", "file_text", "view_range", "old_str", "new_str"],
    "submit": [],
    "fetch": ["url"],
}
# Params whose value is newline-wrapped inside the tag (matches SWE-smith XML_STR_REPLACES).
BLOCK_PARAMS = {"file_text", "old_str", "new_str"}
# Params parsed as JSON (everything else is a raw string).
JSON_PARAMS = {"view_range"}
# Params coerced to int on parse.
INT_PARAMS = {"timeout"}

STOP = "STOP"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)

# --- training-data load-time handling of a few off-spec stored rows ---
VALID_SRE_COMMANDS = {"view", "create", "str_replace", "insert", "undo_edit"}
IGNORE_ARG_KEYS = {"duplicate_str_replace_editor"}   # junk key in one stored row, never rendered
DROP_TOOLS = {"submit", "fetch"}                     # tools outside the vocabulary: drop the trace


class DialectError(ValueError):
    """Raised when a structure cannot be rendered, or text cannot be parsed, losslessly."""


# ---- rendering (structured -> text) --------------------------------------------------
def _render_value(key: str, value: Any) -> str:
    if key in JSON_PARAMS:
        return json.dumps(value)               # [20, 60] -> "[20, 60]"
    if key in INT_PARAMS:
        return str(value)
    s = value if isinstance(value, str) else json.dumps(value)
    if key in BLOCK_PARAMS:
        return "\n" + s + "\n"                 # newline-wrapped block
    return s


def render_action(name: str, args: dict) -> str:
    """One tool call -> a single <function=...>...</function> block (no trailing newline)."""
    order = PARAM_ORDER.get(name)
    keys = order + [k for k in args if k not in order] if order is not None else list(args)
    lines = [f"<function={name}>"]
    for k in keys:
        if k not in args or k in IGNORE_ARG_KEYS:
            continue
        rendered = _render_value(k, args[k])
        # collision guard: a value containing any structural token would break round-trip.
        if any(tok in rendered for tok in ("<function=", "</function>", "<parameter=", "</parameter>")):
            raise DialectError(f"value for '{k}' contains a reserved dialect token")
        lines.append(f"<parameter={k}>{rendered}</parameter>")
    lines.append("</function>")
    return "\n".join(lines)


def fold_think(content: str, thought_arg: str) -> str:
    """think folds (message content, thought arg) into one discussion string, order-preserving."""
    parts = [p for p in ((content or "").strip(), (thought_arg or "").strip()) if p]
    return "\n".join(parts)


def render_assistant(msg: dict) -> str:
    """
    Render one stored assistant turn to its dialect text.
      - terminal handoff (no tool_calls): content verbatim (already the <files>..STOP handoff).
      - think tool call: bare discussion text (folded), no <function>.
      - any other tool call: thought (message content, stripped) then the <function> block.
    """
    tcs = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    if not tcs:
        return content                                  # terminal handoff, verbatim
    tc = tcs[0]                                          # exactly one action per turn (verified)
    name = tc["function"]["name"]
    args = tc["function"]["arguments"]
    args = json.loads(args) if isinstance(args, str) else args
    if name == "think":
        return fold_think(content, args.get("thought", ""))
    block = render_action(name, args)
    thought = content.strip()
    return (thought + "\n" + block) if thought else block


# ---- parsing (text -> structured) ----------------------------------------------------
def _parse_value(key: str, raw: str) -> Any:
    if key in JSON_PARAMS:
        return json.loads(raw)
    if key in INT_PARAMS:
        return int(raw)
    if key in BLOCK_PARAMS:
        if raw.startswith("\n"):
            raw = raw[1:]
        if raw.endswith("\n"):
            raw = raw[:-1]
        return raw
    return raw


def parse_action(block: str) -> tuple[str, dict]:
    """A <function=...>...</function> block -> (name, args dict). Inverse of render_action."""
    m = _FUNC_RE.search(block)
    if not m:
        raise DialectError("no <function=...> block found")
    name, body = m.group(1), m.group(2)
    args: dict[str, Any] = {}
    for pm in _PARAM_RE.finditer(body):
        args[pm.group(1)] = _parse_value(pm.group(1), pm.group(2))
    if not args:
        # Tolerant fallback (inference only; the <parameter=> form above is what was trained).
        # Some models, including the untrained base, emit `key=value` lines.
        for ln in body.strip().splitlines():
            ln = ln.strip()
            if "=" in ln:
                k, v = ln.split("=", 1)
                k = k.strip()
                if k.isidentifier():
                    args[k] = _parse_value(k, v.strip())
    return name, args


def is_handoff(text: str) -> bool:
    return "<files>" in text and text.rstrip().endswith(STOP)


def parse_assistant(text: str) -> dict:
    """
    Classify + parse one rendered assistant turn.
      -> {"kind":"handoff", "text":...}
      -> {"kind":"action", "thought":..., "name":..., "args":...}
      -> {"kind":"think",  "thought":...}
    """
    if is_handoff(text):
        return {"kind": "handoff", "text": text}
    # The action is always the LAST function block (reasoning precedes it, one call per turn).
    # Pair the last </function> with the last <function= before it, so a stray "<function=...>"
    # inside the reasoning text cannot absorb the real call.
    close = text.rfind("</function>")
    if close == -1:
        return {"kind": "think", "thought": text.strip()}
    open_ = text.rfind("<function=", 0, close)
    if open_ == -1:
        return {"kind": "think", "thought": text.strip()}
    block = text[open_:close + len("</function>")]
    name, args = parse_action(block)
    return {"kind": "action", "thought": text[:open_].strip(), "name": name, "args": args}


# ---- training-time flattening --------------------------------------------------------
def render_for_training(messages: list[dict]) -> list[dict]:
    """
    Stored structured messages -> rendered chat messages carrying the loss flag.
      system  -> SYSTEM_PROMPT (frozen), loss=False
      user    -> verbatim issue, loss=False
      tool    -> "OBSERVATION:\\n{content}" as a user turn, loss=False
      assistant -> render_assistant(...), loss=True
    The masker/collator downstream uses the per-message loss flag.
    """
    out = []
    for m in messages:
        role = m["role"]
        if role == "system":
            out.append({"role": "system", "content": SYSTEM_PROMPT, "loss": False})
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or "", "loss": False})
        elif role == "tool":
            c = m.get("content") or ""
            if not c.startswith("OBSERVATION:"):
                c = "OBSERVATION:\n" + c
            out.append({"role": "user", "content": c, "loss": False})
        elif role == "assistant":
            out.append({"role": "assistant", "content": render_assistant(m), "loss": True})
        else:
            raise DialectError(f"unexpected role {role!r}")
    return out


# ---- load-time disposition + context-overflow policy (frozen; inference uses the same code) ----
def drop_reason(example: dict) -> str | None:
    """Return why a stored trace is dropped at load (unknown tool / invalid editor cmd), else None."""
    for m in example.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            name = tc["function"]["name"]
            if name in DROP_TOOLS:
                return f"phantom-tool:{name}"
            if name == "str_replace_editor":
                a = tc["function"]["arguments"]
                a = json.loads(a) if isinstance(a, str) else a
                if a.get("command") not in VALID_SRE_COMMANDS:
                    return "invalid-editor-command"
    return None


TRUNCATION_MARKER = "[... earlier exploration turns were truncated to fit the context window ...]"


def _chatml(m: dict) -> str:
    return f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"


def _ntok(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_to_fit(flat: list[dict], tokenizer, max_tokens: int = 32768, margin: int = 16):
    """
    Frozen context-overflow policy (training == live episodes). `flat` = render_for_training output:
    [system, user(issue), ...exploration turns..., assistant(handoff)]. If it exceeds the window,
    KEEP system + issue + the FULL final handoff + as many of the MOST RECENT turns as fit, and
    drop the oldest middle turns, inserting one visible TRUNCATION_MARKER line at the cut.
    Returns (possibly-truncated flat, was_truncated: bool).
    """
    lens = [_ntok(_chatml(m), tokenizer) for m in flat]
    if sum(lens) <= max_tokens:
        return flat, False
    marker = {"role": "user", "content": TRUNCATION_MARKER, "loss": False}
    fixed = lens[0] + lens[1] + lens[-1] + _ntok(_chatml(marker), tokenizer)  # system+issue+handoff+marker
    budget, kept, used = max_tokens - margin - fixed, [], 0
    for m, l in zip(reversed(flat[2:-1]), reversed(lens[2:-1])):
        if used + l > budget:
            break
        kept.insert(0, m)
        used += l
    result = flat[:2] + [marker] + kept + [flat[-1]]
    # exact re-verify (per-turn sums aren't perfectly additive across BPE boundaries)
    while kept and _ntok("".join(_chatml(m) for m in result), tokenizer) > max_tokens:
        kept.pop(0)
        result = flat[:2] + [marker] + kept + [flat[-1]]
    return result, True


def build_labeled(flat: list[dict], tokenizer):
    """
    Canonical loss masker used at training time. Render `flat` (render_for_training output)
    to Qwen ChatML, tokenize once with offsets, and return (input_ids, labels) where labels = the
    token id on assistant content+<|im_end|> spans (loss=True) and -100 everywhere else. The
    <|im_start|>role\\n header is masked (it is forced at generation).
    """
    spans, parts, pos = [], [], 0
    for m in flat:
        hdr = f"{IM_START}{m['role']}\n"
        parts.append(hdr); pos += len(hdr)
        start = pos
        body = m["content"] + IM_END
        parts.append(body); pos += len(body)
        if m["loss"]:
            spans.append((start, pos))
        parts.append("\n"); pos += 1
    s = "".join(parts)
    enc = tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels = [tid if (a != b and any(a >= ls and b <= le for ls, le in spans)) else -100
              for tid, (a, b) in zip(ids, offs)]
    return ids, labels
