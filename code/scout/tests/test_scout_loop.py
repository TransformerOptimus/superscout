# SPDX-License-Identifier: Apache-2.0
"""Episode loop parity: the released loop, run on scripted turns against a fake sandbox, must send
exactly the requests the original loop sent (fixtures/loop_parity.json)."""
import hashlib
import json
from pathlib import Path

import pytest

import engine
import episode
import fakes

PARITY = json.loads((Path(__file__).resolve().parent / "fixtures" / "loop_parity.json")
                    .read_text(encoding="utf-8"))


class Scripted(engine.ReplayEngine):
    def generate(self, messages, cache_salt=None):
        self.salts = getattr(self, "salts", []) + [cache_salt]
        self.hashes = getattr(self, "hashes", []) + [
            hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()]
        return super().generate(messages, cache_salt)


@pytest.mark.parametrize("name", sorted(PARITY["scripts"]))
def test_loop_matches_original(name):
    exp = PARITY["scripts"][name]
    world = fakes.FakeWorld()
    eng = Scripted(fakes.SCRIPTS[name]["turns"])
    rec = episode.run_episode(dict(fakes.TASK), eng, lambda t: world,
                              max_turns=exp["max_turns"], tokenizer=fakes.FakeTokenizer())

    assert rec["final_request"] == exp["final_request"]
    assert eng.hashes == exp["request_sha256"]           # every request, not just the last
    assert [list(c) for c in world.calls] == [list(c) for c in exp["world_calls"]]
    got_turns = [{k: v for k, v in t.items() if k != "latency"} for t in rec["turns"]]
    assert got_turns == exp["turns"]
    for k, v in exp["record"].items():
        assert rec.get(k) == v, k

    salt = rec["cache_salt"]
    assert salt.startswith(fakes.TASK["instance_id"] + "-") and len(salt) == len(fakes.TASK["instance_id"]) + 9
    assert set(eng.salts) == {salt}


def test_forced_script_exercises_truncation_and_capping():
    fr = PARITY["scripts"]["forced"]["final_request"]
    import dialect
    assert any(m["content"] == dialect.TRUNCATION_MARKER for m in fr)
    assert any("chars truncated]..." in m["content"] for m in fr)
    assert PARITY["scripts"]["forced"]["record"]["handoff_type"] == "forced"


def test_cap_obs():
    raw = "a" * 5000 + "b" * 5000
    capped = episode.cap_obs(raw)
    assert capped == "a" * 4000 + "\n...[2000 chars truncated]...\n" + "b" * 4000
    assert episode.cap_obs("short") == "short"
    assert episode.normalize_obs("x") == "OBSERVATION:\nx"
    assert episode.normalize_obs("OBSERVATION:\nx") == "OBSERVATION:\nx"
