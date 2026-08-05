"""infer.py — minimal inference for the frozen SuperScout router head.

Pure numpy. This module never fits, refits, calibrates or updates anything: it loads
the serialized per-fixer resume centroids + logistic-regression coefficients from
router_head.npz (the scaler/LR are just an affine map + sigmoid) and the frozen
configuration from router_spec.json, and evaluates them.

Inputs per task:
    text_embedding : 1024-d issue-text embedding of the problem statement, produced by
                     the pinned embedder in router_spec.json -> "embedder"
                     (Qwen/Qwen3-Embedding-0.6B, L2-normalized, plain text).
    hidden_state   : 3584-d SuperScout-7B final-token pre-decode hidden state,
                     layer -4 (spec key "state_pre_-4").

Output:
    P_text[m]  = head(text resumes,  text_embedding)          per fixer m
    P_state[m] = head(state resumes, hidden_state)            per fixer m
    P_blend[m] = mean(P_text[m], P_state[m])                  the deployed blend
    pick       = first fixer in cheap_order with P_blend >= theta (spec "theta"), else anchor

If hidden_state is None the route degrades to the text head alone at the blend's
theta and the record is tagged degraded=True (the deployed system never synthesizes
a state).

Smoke test (synthetic unit vectors of the correct dimensionality):
    python infer.py --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_artifact(root=None) -> dict:
    root = Path(root or HERE)
    spec = json.loads((root / "router_spec.json").read_text())
    z = np.load(root / "router_head.npz")
    art = {"spec": spec, "pool": [str(x) for x in z["pool"]]}
    for space in ("text", "state"):
        art[space] = {
            "s": z[f"{space}_s"], "f": z[f"{space}_f"], "p": z[f"{space}_p"],
            "mean": z[f"{space}_scaler_mean"], "scale": z[f"{space}_scaler_scale"],
            "coef": z[f"{space}_coef"], "intercept": float(z[f"{space}_intercept"][0]),
        }
    return art


def _l2(v):
    v = np.asarray(v, np.float32)
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v


def _cos(x, c):
    return float(x @ c / (np.linalg.norm(c) + 1e-12))


def _feats(x, s, f, p):
    cs, cf = _cos(x, s), _cos(x, f)
    return [cs, cf, cs - cf, float(p), cs * float(p), (cs - cf) * float(p)]


def _head_p(space: dict, x, mi: int) -> float:
    fv = _feats(x, space["s"][mi], space["f"][mi], space["p"][mi])
    z = (np.asarray(fv, float) - space["mean"]) / space["scale"]
    logit = float(z @ space["coef"] + space["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def route(text_embedding, hidden_state=None, art=None) -> dict:
    """Route one task. Returns probabilities per fixer and the decision."""
    art = art or load_artifact()
    spec, pool = art["spec"], art["pool"]
    theta = float(spec["theta"])
    anchor = spec["anchor"]
    cheap = list(spec["cheap_order"])

    x_text = _l2(text_embedding)
    if x_text.shape[0] != art["text"]["s"].shape[1]:
        raise ValueError(f"text embedding dim {x_text.shape[0]} != frozen "
                         f"{art['text']['s'].shape[1]}")
    p_text = {m: _head_p(art["text"], x_text, mi) for mi, m in enumerate(pool)}

    degraded = hidden_state is None
    if not degraded:
        x_st = _l2(hidden_state)
        if x_st.shape[0] != art["state"]["s"].shape[1]:
            raise ValueError(f"hidden state dim {x_st.shape[0]} != frozen "
                             f"{art['state']['s'].shape[1]}")
        p_state = {m: _head_p(art["state"], x_st, mi) for mi, m in enumerate(pool)}
        p_blend = {m: float(np.mean([p_text[m], p_state[m]])) for m in pool}
    else:
        p_state = None
        p_blend = dict(p_text)

    pick = next((m for m in cheap if p_blend[m] >= theta), anchor)
    return {
        "recipe": "blend_text_plus_state" if not degraded else "blend_DEGRADED_text_only",
        "degraded": degraded,
        "theta": theta,
        "anchor": anchor,
        "cheap_order": cheap,
        "pick": pick,
        "p_solve_blend": {m: round(p_blend[m], 6) for m in pool},
        "p_solve_text": {m: round(p_text[m], 6) for m in pool},
        "p_solve_state": ({m: round(p_state[m], 6) for m in pool} if not degraded else None),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="route one task from precomputed vectors")
    ap.add_argument("--smoke", action="store_true",
                    help="run with synthetic unit vectors (deterministic, seed 1234)")
    ap.add_argument("--text-npy", help="path to .npy with the 1024-d text embedding")
    ap.add_argument("--state-npy", help="path to .npy with the 3584-d hidden state")
    a = ap.parse_args()
    art = load_artifact()
    if a.smoke:
        rng = np.random.default_rng(1234)
        x_text = _l2(rng.normal(size=art["text"]["s"].shape[1]))
        x_state = _l2(rng.normal(size=art["state"]["s"].shape[1]))
    else:
        if not a.text_npy:
            ap.error("--text-npy is required unless --smoke")
        x_text = np.load(a.text_npy)
        x_state = np.load(a.state_npy) if a.state_npy else None
    print(json.dumps(route(x_text, x_state, art), indent=2))
