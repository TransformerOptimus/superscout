# SPDX-License-Identifier: Apache-2.0
"""Docker sandbox and tool executors for SuperScout-7B search episodes on SWE-bench Pro.

One container per task, started from the task's public SWE-bench Pro image with the image
ENTRYPOINT cleared and `sleep infinity` as the command. Pro images end with
ENTRYPOINT ["/bin/bash"]; without clearing it the keep-alive command is handed to bash as a
script name and the container exits at once.

Every tool command runs as

    bash -c "timeout <t> bash -c '<TOOL_ENV> ; cd <workdir> 2>/dev/null; <cmd>' < /dev/null"

TOOL_ENV is a guarded per-repo environment prefix. It was part of the model's tool shell in the
reported runs, so it must be kept to reproduce them: without the ansible clause, `import ansible`
resolves to a stale installed copy instead of the repo under /app. Each clause is `test`-guarded
and is a no-op where it does not apply.

The executors (bash / view / create / str_replace / dispatch) and their output formats are the
ones the model was trained against. Observations are returned raw; the episode loop adds the
"OBSERVATION:" prefix.
"""
from __future__ import annotations
import base64
import shlex
import subprocess
import time
import uuid

_CONDA_ENV = "/opt/conda/envs/testbed"
TOOL_ENV = (
    f'if [ -d {_CONDA_ENV} ]; then '
    f'. /opt/conda/etc/profile.d/conda.sh 2>/dev/null; conda activate testbed 2>/dev/null; fi ; '
    '[ -d /app/lib/ansible ] && export PYTHONPATH="/app/lib:${PYTHONPATH}" ; '
    '[ -f /app/conf/openlibrary.yml ] && export OL_CONFIG=/app/conf/openlibrary.yml ; '
    'if [ -d /app/qutebrowser ]; then '
    'export QT_QPA_PLATFORM=offscreen; export DISPLAY=:99; export PYTEST_QT_API=pyqt5; fi'
)

# Repo-root candidates probed when the preferred workdir is missing or empty.
ROOT_CANDIDATES = ["/testbed", "/app", "/repo", "/home/user", "/workspace", "/src", "/code"]

IMAGE_PREFIX = "jefzda/sweap-images:"


def image_present(image_ref, docker="docker"):
    return subprocess.run([docker, "image", "inspect", image_ref], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def pull_image(image_ref, platform="linux/amd64", docker="docker"):
    """Pull the image if it is not present locally. Raises on failure."""
    if image_present(image_ref, docker):
        return
    subprocess.run([docker, "pull", "--platform", platform, image_ref], check=True)


class DockerSandbox:
    """One task's world. `ready` is True once the repo root was located and is non-empty."""

    CMD_TIMEOUT = 60      # default per-command limit (seconds)
    READ_SLACK = 30       # client-side bound = command timeout + this slack

    def __init__(self, instance_id, image_ref, gold_files=None, workdir="/app", timeout=1200,
                 platform="linux/amd64", docker="docker", activate=True, cpus=None, memory=None):
        self.iid = instance_id
        self.image_ref = image_ref
        self.gold_files = gold_files or []
        self.workdir = workdir
        self.docker = docker
        self.tool_env = TOOL_ENV if activate else "true"
        self.cid = None
        argv = [docker, "run", "-d", "--rm", "--platform", platform, "--entrypoint", "",
                "--name", "superscout-" + uuid.uuid4().hex[:12]]
        if cpus:
            argv += ["--cpus", str(cpus)]
        if memory:
            argv += ["--memory", str(memory)]
        argv += [image_ref, "sleep", "infinity"]
        p = subprocess.run(argv, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError("docker run failed: " + p.stderr.strip()[:500])
        self.cid = p.stdout.strip()
        # Episode wall-clock budget: the loop stops taking turns once this has elapsed.
        self.timeout, self.created = timeout, time.time()
        try:
            out, _err, _rc = self._raw("echo SMOKE_OK")
            self.boot_smoke_ok = "SMOKE_OK" in (out or "")
            self.workdir, self.root_fallback = self._resolve_root(workdir)
            if self.workdir is None:
                self.gold_at_base = []
                self.scrub = {"status": "setup_failed", "reason": "no_repo_root"}
                self.ready = False
                return
            present = [f for f in self.gold_files
                       if self._exec("test -e " + shlex.quote(f) + " && echo Y")[0].strip() == "Y"]
            self.gold_at_base = present
            self.scrub = {"status": "ok", "gold_at_base": present,
                          "gold_measurable": len(present) >= 1, "workdir": self.workdir,
                          "root_fallback": self.root_fallback,
                          "boot_smoke_ok": self.boot_smoke_ok,
                          "tool_env": "activated" if activate else "bare"}
            self.ready = True
        except BaseException:
            self.close()
            raise

    # ---- command execution ----
    def _raw(self, cmd, timeout=None):
        t = timeout or self.CMD_TIMEOUT
        # `timeout` bounds runtime; `</dev/null` gives interactive prompts EOF so they abort.
        argv = [self.docker, "exec", self.cid, "bash", "-c",
                f"timeout {t} bash -c {shlex.quote(cmd)} < /dev/null"]
        try:
            p = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=t + self.READ_SLACK)
        except subprocess.TimeoutExpired:
            return "", "[read timeout]", 124
        out = p.stdout.decode("utf-8", errors="replace")
        err = p.stderr.decode("utf-8", errors="replace")
        rc = p.returncode
        if rc == 124:
            err = (err + f"\n[command killed: exceeded {t}s timeout]").strip()
        return out, err, rc

    def _exec(self, cmd, workdir=None, timeout=None):
        wd = workdir or self.workdir
        payload = self.tool_env + " ; cd " + shlex.quote(wd) + " 2>/dev/null; " + cmd
        return self._raw(payload, timeout=timeout)

    def exec(self, cmd, timeout=300):
        """(output, returncode) for the verify-then-strip gate's exec_fn: TOOL_ENV, then
        `cd <workdir> && cmd`, stdout and stderr joined on a newline."""
        payload = self.tool_env + " ; cd " + shlex.quote(self.workdir) + " && " + cmd
        out, err, rc = self._raw(payload, timeout=timeout)
        combined = out if not err else (out + ("\n" if out and not out.endswith("\n") else "") + err)
        return combined, rc

    def _resolve_root(self, preferred):
        """(root, fallback_flag): `preferred` if non-empty, else the first non-empty candidate."""
        def nonempty(d):
            out, _, rc = self._exec("test -d " + shlex.quote(d) +
                                    " && ls -A " + shlex.quote(d) + " | head -1")
            return rc == 0 and out.strip() != ""
        if nonempty(preferred):
            return preferred, False
        for c in ROOT_CANDIDATES:
            if c != preferred and nonempty(c):
                return c, True
        return None, False

    # ---- tool executors (raw observations) ----
    def bash(self, command, timeout=None, **_):
        try:
            t = min(int(timeout), 120) if timeout else self.CMD_TIMEOUT
        except (ValueError, TypeError):
            t = self.CMD_TIMEOUT
        out, err, _ = self._exec(command, timeout=t)
        return out + err

    def _is_dir(self, path):
        out, _, rc = self._exec(f"test -d {shlex.quote(path)} && echo D || echo F")
        return out.strip() == "D"

    def view(self, path, view_range=None, **_):
        if self._is_dir(path):
            out, _, _ = self._exec(
                f"find {shlex.quote(path)} -maxdepth 2 -not -path '*/.*' | sort")
            return (f"Here's the files and directories up to 2 levels deep in {path}, "
                    f"excluding hidden items:\n{out.rstrip()}")
        cmd = f"cat -n {shlex.quote(path)}"
        if view_range and isinstance(view_range, list) and len(view_range) == 2:
            a, b = view_range
            cmd = f"cat -n {shlex.quote(path)} | sed -n '{a},{b}p'"
        out, err, rc = self._exec(cmd)
        if rc != 0:
            return f"ERROR:\n{err.strip() or 'could not view '+path}"
        return f"Here's the result of running `cat -n` on {path}:\n{out.rstrip()}"

    def create(self, path, file_text, **_):
        # The first probe's result is unused; it is kept so the command sequence matches the
        # original executor exactly.
        _, _, rc = self._exec(f"test -e {shlex.quote(path)} && echo EXISTS || true")
        exists, _, _ = self._exec(f"test -e {shlex.quote(path)} && echo Y || echo N")
        if exists.strip() == "Y":
            return f"ERROR:\nFile already exists at: {path}. Cannot overwrite with `create`."
        b64 = base64.b64encode(file_text.encode()).decode()
        _, err, rc = self._exec(f"mkdir -p $(dirname {shlex.quote(path)}) && echo {b64} | base64 -d > {shlex.quote(path)}")
        if rc != 0:
            return f"ERROR:\n{err.strip()}"
        return f"File created successfully at: {path}"

    def str_replace(self, path, old_str, new_str="", **_):
        content, err, rc = self._exec(f"cat {shlex.quote(path)}")
        if rc != 0:
            return f"ERROR:\nThe path {path} does not exist."
        n = content.count(old_str)
        if n == 0:
            return f"ERROR:\nNo replacement performed, old_str did not appear verbatim in {path}."
        if n > 1:
            return f"ERROR:\nNo replacement performed. old_str appeared {n} times in {path}; it must be unique."
        new_content = content.replace(old_str, new_str, 1)
        b64 = base64.b64encode(new_content.encode()).decode()
        self._exec(f"echo {b64} | base64 -d > {shlex.quote(path)}")
        line = new_content[:new_content.find(new_str)].count("\n") + 1 if new_str else content[:content.find(old_str)].count("\n") + 1
        a, b = max(1, line - 4), line + new_str.count("\n") + 4
        snip, _, _ = self._exec(f"cat -n {shlex.quote(path)} | sed -n '{a},{b}p'")
        return (f"The file {path} has been edited. Here's the result of running `cat -n` on a "
                f"snippet of {path}:\n{snip.rstrip()}")

    # Required params per tool/command. A mis-formatted call gets a clean ERROR naming the
    # missing parameter instead of a Python exception.
    _REQUIRED = {"bash": ["command"], "view": ["path"],
                 "create": ["path", "file_text"], "str_replace": ["path", "old_str"]}

    def _missing(self, key, args):
        need = [p for p in self._REQUIRED.get(key, []) if p not in args or args.get(p) in (None, "")]
        return f"ERROR:\nMissing required parameter(s) for `{key}`: {', '.join(need)}." if need else None

    def dispatch(self, name, args):
        """Route a parsed action to its executor. Returns raw observation text. Malformed or
        incomplete calls come back as an ERROR observation the model can recover from."""
        try:
            if name == "bash":
                return self._missing("bash", args) or self.bash(**args)
            if name == "str_replace_editor":
                cmd = args.get("command")
                if cmd not in ("view", "create", "str_replace"):
                    return (f"ERROR:\nstr_replace_editor requires `command` to be one of "
                            f"view, create, str_replace (got {cmd!r}).")
                rest = {k: v for k, v in args.items() if k != "command"}
                return self._missing(cmd, rest) or \
                    {"view": self.view, "create": self.create, "str_replace": self.str_replace}[cmd](**rest)
            if name == "think":
                return "Your thought has been logged."
            return f"ERROR:\nUnknown tool {name}"
        except TypeError as e:
            return f"ERROR:\nMalformed call to `{name}` ({str(e)[:120]}). Check the parameter names."

    def close(self):
        """Remove the container. Never raises."""
        if not self.cid:
            return
        try:
            subprocess.run([self.docker, "rm", "-f", self.cid], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)
        except Exception as e:  # noqa: BLE001
            print("[sandbox] WARN could not remove container %s: %s" % (self.cid[:12], e),
                  flush=True)
        self.cid = None
