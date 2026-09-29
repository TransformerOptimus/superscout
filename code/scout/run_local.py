# SPDX-License-Identifier: Apache-2.0
"""Run SuperScout-7B search episodes on SWE-bench Pro tasks with a local Docker sandbox.

    python code/scout/run_local.py --endpoint http://localhost:8000 \\
        --instance-id <instance_id> [--instance-id ...] [--out out/]

Loads each task row from the public dataset (pinned revision), pulls its public image, runs one
episode against the vLLM endpoint, writes <out>/<instance_id>.json, and prints the handoff.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine as engine_mod    # noqa: E402
import episode                 # noqa: E402
import sandbox as sandbox_mod  # noqa: E402
import tasks                   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", required=True, help="vLLM OpenAI-compatible base URL")
    ap.add_argument("--model", default="SuperScout-7B", help="served model name")
    ap.add_argument("--instance-id", action="append", required=True,
                    help="SWE-bench Pro instance_id (repeatable)")
    ap.add_argument("--out", default="out", help="output directory for episode records")
    ap.add_argument("--dataset-revision", default=tasks.HF_REVISION)
    ap.add_argument("--max-turns", type=int, default=episode.MAX_TURNS)
    ap.add_argument("--episode-timeout", type=int, default=1200,
                    help="episode wall-clock budget in seconds (1200 in the reported runs)")
    ap.add_argument("--platform", default="linux/amd64")
    ap.add_argument("--docker", default="docker", help="docker CLI binary")
    ap.add_argument("--cpus", default=None, help="optional docker --cpus limit")
    ap.add_argument("--memory", default=None, help="optional docker --memory limit, e.g. 30g")
    ap.add_argument("--no-tool-env", dest="activate", action="store_false",
                    help="run tools in the bare shell (not what the reported runs used)")
    a = ap.parse_args(argv)

    eng = engine_mod.HttpEngine(a.endpoint, a.model)
    print("[scout] sampling=" + json.dumps(eng.sampling), flush=True)

    def factory(task):
        return sandbox_mod.DockerSandbox(task["instance_id"], task["image_ref"],
                                         gold_files=task.get("gold_files"),
                                         workdir=task.get("repo_path_in_image", "/app"),
                                         timeout=a.episode_timeout, platform=a.platform,
                                         docker=a.docker, activate=a.activate,
                                         cpus=a.cpus, memory=a.memory)

    for iid in a.instance_id:
        task = tasks.task_from_row(tasks.load_row(iid, revision=a.dataset_revision))
        print("[scout] %s  image=%s" % (iid, task["image_ref"]), flush=True)
        sandbox_mod.pull_image(task["image_ref"], platform=a.platform, docker=a.docker)
        rec = episode.run_episode(task, eng, factory, out_dir=a.out, max_turns=a.max_turns)
        print("[scout] outcome=%s handoff_type=%s turns=%s wall=%ss -> %s"
              % (rec["outcome"], rec.get("handoff_type"), rec.get("n_turns"), rec.get("wall"),
                 os.path.join(a.out, iid + ".json")), flush=True)
        if rec.get("final_response"):
            print(rec["final_response"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
