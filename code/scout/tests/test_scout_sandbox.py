# SPDX-License-Identifier: Apache-2.0
"""Docker sandbox command construction and tool output formats (docker is stubbed out)."""
import json
import shlex
import subprocess
from pathlib import Path

import pytest

import sandbox

STRINGS = json.loads((Path(__file__).resolve().parent / "fixtures" / "loop_parity.json")
                     .read_text(encoding="utf-8"))["strings"]


def inner(argv):
    """The tool-shell payload inside `timeout N bash -c <payload> < /dev/null`."""
    return shlex.split(argv[5])[4]


class FakeDocker:
    """Stands in for subprocess.run. `responder(payload) -> (stdout, stderr, rc)`."""

    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[1] == "run":
            return subprocess.CompletedProcess(argv, 0, "cid123\n", "")
        if argv[1] == "exec":
            out, err, rc = self.responder(inner(argv))
            return subprocess.CompletedProcess(argv, rc, out.encode(), err.encode())
        return subprocess.CompletedProcess(argv, 0, b"", b"")


def boot(monkeypatch, responder):
    fd = FakeDocker(responder)
    monkeypatch.setattr(sandbox.subprocess, "run", fd)
    sb = sandbox.DockerSandbox("iid", "img:tag", gold_files=["a.py"], workdir="/app")
    return sb, fd


def default_responder(cmd):
    if "echo SMOKE_OK" in cmd:
        return "SMOKE_OK\n", "", 0
    if "ls -A" in cmd:
        return "x\n", "", 0
    if "test -e" in cmd and "echo Y" in cmd:
        return "Y\n", "", 0
    return "", "", 0


def test_tool_env_identical():
    assert sandbox.TOOL_ENV == STRINGS["tool_env"]


def test_boot_and_command_shape(monkeypatch):
    sb, fd = boot(monkeypatch, default_responder)
    run = fd.calls[0]
    assert run[:3] == ["docker", "run", "-d"]
    i = run.index("--entrypoint")
    assert run[i + 1] == ""
    assert run[-3:] == ["img:tag", "sleep", "infinity"]
    assert run[run.index("--platform") + 1] == "linux/amd64"
    assert sb.ready and sb.workdir == "/app" and sb.gold_at_base == ["a.py"]

    fd.calls.clear()
    ex = STRINGS["bash_payload_example"]
    sb.bash(ex["command"])
    argv = fd.calls[0]
    assert argv[:5] == ["docker", "exec", "cid123", "bash", "-c"]
    assert argv[5] == ex["expected_inner"]


def test_bash_timeout_clamp_and_kill_message(monkeypatch):
    sb, fd = boot(monkeypatch, default_responder)
    fd.responder = lambda inner: ("partial\n", "", 124)
    obs = sb.bash("sleep 999", timeout="500")
    assert fd.calls[-1][5].startswith("timeout 120 bash -c ")
    assert obs == "partial\n[command killed: exceeded 120s timeout]"
    sb.bash("x", timeout="abc")
    assert fd.calls[-1][5].startswith("timeout 60 bash -c ")


def test_client_side_read_timeout(monkeypatch):
    sb, fd = boot(monkeypatch, default_responder)

    def hang(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

    monkeypatch.setattr(sandbox.subprocess, "run", hang)
    assert sb.bash("x") == "[read timeout]"


def test_view_formats(monkeypatch):
    def resp(inner):
        if "echo D || echo F" in inner:
            return ("D\n" if "test -d /app/pkg " in inner else "F\n"), "", 0
        if "find " in inner:
            return "/app/pkg\n/app/pkg/a.py\n", "", 0
        if "cat -n" in inner and "missing.py" in inner:
            return "", "cat: /app/missing.py: No such file or directory\n", 1
        if "cat -n" in inner:
            return "     1\tx = 1\n", "", 0
        return default_responder(inner)

    sb, fd = boot(monkeypatch, resp)
    assert sb.dispatch("str_replace_editor", {"command": "view", "path": "/app/pkg"}) == (
        "Here's the files and directories up to 2 levels deep in /app/pkg, excluding hidden "
        "items:\n/app/pkg\n/app/pkg/a.py")
    assert sb.dispatch("str_replace_editor", {"command": "view", "path": "/app/a.py"}) == (
        "Here's the result of running `cat -n` on /app/a.py:\n     1\tx = 1")
    sb.dispatch("str_replace_editor", {"command": "view", "path": "/app/a.py", "view_range": [3, 9]})
    assert inner(fd.calls[-1]).endswith("cat -n /app/a.py | sed -n '3,9p'")
    assert sb.dispatch("str_replace_editor", {"command": "view", "path": "/app/missing.py"}) == (
        "ERROR:\ncat: /app/missing.py: No such file or directory")


def test_dispatch_errors(monkeypatch):
    sb, _ = boot(monkeypatch, default_responder)
    assert sb.dispatch("bash", {}) == "ERROR:\nMissing required parameter(s) for `bash`: command."
    assert sb.dispatch("str_replace_editor", {"command": "insert", "path": "/a"}) == (
        "ERROR:\nstr_replace_editor requires `command` to be one of view, create, str_replace "
        "(got 'insert').")
    assert sb.dispatch("str_replace_editor", {"command": "create", "path": "/a"}) == (
        "ERROR:\nMissing required parameter(s) for `create`: file_text.")
    assert sb.dispatch("introspect", {"command": "x"}) == "ERROR:\nUnknown tool introspect"
    assert sb.dispatch("bash", {"command": "ls", "bogus": 1}) == ""   # extra kwargs are ignored
    assert sb.dispatch("str_replace_editor", {"command": "view", "path": "/a", "x": 1}).startswith("Here's")


def test_create_and_str_replace(monkeypatch):
    files = {"/app/f.py": "a = 1\nb = 2\n"}

    def resp(inner):
        if "test -e /app/new.py && echo Y" in inner:
            return "N\n", "", 0
        if "test -e /app/f.py && echo Y || echo N" in inner:
            return "Y\n", "", 0
        if inner.split("2>/dev/null; ")[-1].startswith("cat /app/f.py"):
            return files["/app/f.py"], "", 0
        if "cat -n /app/f.py | sed -n" in inner:
            return "     1\ta = 1\n     2\tb = 3\n", "", 0
        return default_responder(inner)

    sb, fd = boot(monkeypatch, resp)
    assert sb.dispatch("str_replace_editor", {"command": "create", "path": "/app/f.py",
                                              "file_text": "x"}) == (
        "ERROR:\nFile already exists at: /app/f.py. Cannot overwrite with `create`.")
    assert sb.dispatch("str_replace_editor", {"command": "create", "path": "/app/new.py",
                                              "file_text": "x"}) == "File created successfully at: /app/new.py"
    assert sb.dispatch("str_replace_editor", {"command": "str_replace", "path": "/app/f.py",
                                              "old_str": "zzz", "new_str": "q"}) == (
        "ERROR:\nNo replacement performed, old_str did not appear verbatim in /app/f.py.")
    assert sb.dispatch("str_replace_editor", {"command": "str_replace", "path": "/app/f.py",
                                              "old_str": "b = 2", "new_str": "b = 3"}) == (
        "The file /app/f.py has been edited. Here's the result of running `cat -n` on a snippet "
        "of /app/f.py:\n     1\ta = 1\n     2\tb = 3")
    assert inner(fd.calls[-1]).endswith("cat -n /app/f.py | sed -n '1,6p'")


def test_close_removes_container(monkeypatch):
    sb, fd = boot(monkeypatch, default_responder)
    sb.close()
    assert fd.calls[-1][:3] == ["docker", "rm", "-f"] and sb.cid is None
    sb.close()      # idempotent


@pytest.mark.parametrize("activate,prefix", [(True, "if [ -d /opt/conda"), (False, "true ; cd ")])
def test_activate_flag(monkeypatch, activate, prefix):
    fd = FakeDocker(default_responder)
    monkeypatch.setattr(sandbox.subprocess, "run", fd)
    sb = sandbox.DockerSandbox("iid", "img", activate=activate)
    sb.bash("ls")
    assert fd.calls[-1][5].startswith("timeout 60 bash -c '" + prefix)


def test_gate_exec(monkeypatch):
    sb, fd = boot(monkeypatch, default_responder)
    fd.responder = lambda cmd: ("out", "Traceback\n", 1)
    assert sb.exec("python t.py") == ("out\nTraceback\n", 1)
    assert inner(fd.calls[-1]).endswith(" ; cd /app && python t.py")
    assert fd.calls[-1][5].startswith("timeout 300 bash -c ")
