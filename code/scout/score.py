# SPDX-License-Identifier: Apache-2.0
"""Parse a handoff (<files>, <repro>) and score predicted files against gold files (recall, path-normalized)."""
import re


def norm_path(p, workdir):
    p = p.strip()
    for pre in (workdir.rstrip("/") + "/", "/testbed/", "/workspace/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.strip().strip("/")


def extract_files(handoff_text):
    m = re.search(r"<files>\n?(.*?)</files>", handoff_text, re.S)
    if not m:
        return []
    return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]


def extract_repro(handoff_text):
    m = re.search(r"<repro>\n?(.*?)</repro>", handoff_text, re.S)
    if not m:
        return {"present": False}
    body = m.group(1)
    cmd = re.search(r"command:\s*(.*)", body)
    rep = re.search(r"reproduced:\s*(.*)", body)
    return {"present": True,
            "command": cmd.group(1).strip() if cmd else None,
            "reproduced_claim": rep.group(1).strip() if rep else None}


def score_files(predicted, gold, workdir="/testbed"):
    P = {norm_path(p, workdir) for p in predicted if p.strip()}
    G = {norm_path(g, workdir) for g in gold if g.strip()}
    if not G:
        return {"recall": 0.0, "precision": 0.0, "all_hit": False, "n_gold": 0}
    hits = P & G
    return {"recall": len(hits) / len(G),
            "precision": len(hits) / len(P) if P else 0.0,
            "all_hit": G.issubset(P),
            "hits": sorted(hits), "missed": sorted(G - P),
            "predicted": sorted(P), "gold": sorted(G),
            "n_gold": len(G), "n_pred": len(P)}
