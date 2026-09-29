# SPDX-License-Identifier: Apache-2.0
"""Sampling is pinned client-side: the exact dict on the wire, including the cache_salt."""
import json
import urllib.error
from pathlib import Path

import pytest

import engine

STRINGS = json.loads((Path(__file__).resolve().parent / "fixtures" / "loop_parity.json")
                     .read_text(encoding="utf-8"))["strings"]

FROZEN = {"temperature": 0.9, "max_tokens": 2048, "stop": ["<|im_end|>"], "top_p": 1.0,
          "top_k": -1, "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def test_resolved_sampling():
    assert engine.resolved_sampling() == FROZEN == STRINGS["sampling"]


def test_payload_exact():
    eng = engine.HttpEngine("http://localhost:8000/", "SuperScout-7B")
    msgs = [{"role": "user", "content": "x"}]
    p = eng._payload(msgs, "salt-abc")
    assert p == {"model": "SuperScout-7B", "messages": msgs, **FROZEN, "cache_salt": "salt-abc"}
    assert p == STRINGS["payload_example"]
    assert "cache_salt" not in eng._payload(msgs, None)
    assert eng.sampling == FROZEN
    assert eng.endpoint == "http://localhost:8000"


def test_4xx_is_raised_not_retried(monkeypatch):
    eng = engine.HttpEngine("http://localhost:1", "m", max_retries=3)
    calls = []

    def boom(messages, cache_salt):
        calls.append(1)
        raise urllib.error.HTTPError("u", 400, "Bad Request", {}, None)

    monkeypatch.setattr(eng, "_once", boom)
    with pytest.raises(urllib.error.HTTPError) as ei:
        eng.generate([], "s")
    assert "400" in str(ei.value) and len(calls) == 1


def test_replay_engine():
    r = engine.ReplayEngine(["a", "b"])
    assert r.generate([{"role": "user", "content": "q"}], "s") == "a"
    assert r.generate([], "s") == "b"
    with pytest.raises(RuntimeError):
        r.generate([], "s")
    assert r.sampling == FROZEN
