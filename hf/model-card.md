---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
pipeline_tag: text-generation
tags:
  - code
  - software-engineering
  - agent
  - routing
---


# SuperScout-7B

SuperScout-7B is a 7B searcher model for repository-level software issues. Given an
issue and a repository, it explores the codebase, localizes the implicated files,
attempts to write a failing reproduction, and emits a structured handoff document
(`<files>` / `<repro>` / `<notes>`, terminated by `STOP`, kept to about a page of
text). It is a searcher only: it does not write the fix.

The model is the front end of the SuperScout system, which sandbox-verifies the
handoff's reproduction claims (stripping false ones) and then routes each task to
one of four frontier fixer models (GPT-5.2, Claude Opus 4.6, Gemini 3 Flash,
Kimi K2.5) using a resume-based router fed by the task text and the searcher's own
hidden states. Adding a new fixer requires no retraining.

Companion artifacts: the training corpus at
[SuperAGI/superscout-sft-search](https://huggingface.co/datasets/SuperAGI/superscout-sft-search),
the router feature arrays at
[SuperAGI/superscout-router-features](https://huggingface.co/datasets/SuperAGI/superscout-router-features),
and the paper, code, and evaluation receipts at
[TransformerOptimus/superscout](https://github.com/TransformerOptimus/superscout).

## Training

SuperScout-7B is a LoRA fine-tune (r=64, alpha=128, all linear projections, no
embedding or output head, bf16, 2 epochs at 32k-token packed context) of
Qwen2.5-Coder-7B-Instruct. It was trained on 19,905 search-phase demonstrations
sliced from openly licensed agent trajectories produced by other systems. Each
example is one complete search episode with a synthesized handoff-emission turn
as the final supervised target; within that turn the file list is extracted
deterministically from the trace and the reproduction record is copied verbatim,
with only the free-form notes written by a model (gpt-oss-120b). Loss is taken on
assistant turns only.

The realized language mix is Python 37.3%, Go 36.7%, TypeScript 19.7%, and
JavaScript 6.3%. A 23-repository contamination blocklist covering all 11
SWE-bench Pro repositories and 12 SWE-bench Verified repositories was excluded
from every training and calibration set; a programmatic gate verified zero
blocklist hits in the frozen file. Before training, the base model's localization
rate is near zero; the search behavior is entirely learned.

## Serve it sampled: temperature 0.9, single draw

Decoding matters more than expected. On a 450-task held-out exam, greedy decoding
finds the right files at a rate of 0.1104, while a single sampled draw at
temperature 0.9 reaches 0.3058, a 2.65x gain after matching for infrastructure
timeouts. The exam's decomposition shows the trade: the emission rate rises
3.4x (0.213 to 0.718) while recall per emitted handoff falls from 0.517 to
0.426, an 18% quality cost, a strongly net-positive exchange. A separate
100-task dial set shows the same mechanism with commitment up 3.03x and
per-handoff quality flat (0.96x). Greedy decoding makes the searcher reluctant
to declare a result. The shipped and evaluated configuration is therefore a
single sampled draw at temperature 0.9 (pinned sampling tail: top_p 1.0,
top_k -1, penalties 0). Do not serve this model greedily.

## Language transfer

Although trained on four languages, the searcher's localization transfers. On the
nine-language SWE-bench Multilingual evaluation (which contains no Python),
spontaneous-handoff localization quality is higher on the six never-trained
languages (file-level F1 = 0.630) than on the three trained ones (F1 = 0.455),
and the inversion survives a difficulty-matched control restricted to
single-gold-file tasks (recall 0.791 never-trained vs 0.524 trained). The
TypeScript and C++ cells rest on 12 assigned tasks each (7 and 8 spontaneous
handoffs analyzed) and are indicative only. The contrast carries a confound:
JavaScript and TypeScript are at once the weakest cells and the smallest
training slices, and may be structurally harder to localize in independently
of training exposure, so the pattern is read as a hypothesis (a learned search
method rather than a language-specific vocabulary), not a causal claim.

## Evaluation

On the full Python slice of SWE-bench Pro (266 tasks, official capped budget
tier), the SuperScout system solves 159 of 266 tasks (59.77%) at a total cost
per solve of $0.230. The pool's best single model, Claude Opus 4.6 run solo,
solves 158 of 266 (59.40%) at $1.274 per solve. SuperScout matches the best
single model's solve rate (159 of 266 for SuperScout, 158 for the best model)
at about a fifth of the total cost per solve at a matched solve rate; the
one-task gap at n=266 is a match, not a win. The calibrated router
concentrates 263 of 266 tasks on the cheapest fixer (Kimi K2.5); a no-router
ablation, always the cheapest fixer with the handoff, also solves 159, so on
this benchmark the handoff carries the result and routing is a cost-allocation
decision. System costs are all-in and include the searcher's entire GPU bill
($1.13 across all 266 search episodes).

As a component, the searcher's file localization on the same 266 tasks reaches
per-task mean recall 0.566 and precision 0.821 from a single sampled draw, with
every gold file named in 24.8% of tasks.

## Intended use and limitations

- **Searcher, not a fixer.** The model emits a handoff for a downstream fixer
  model; it is not trained to produce patches and is not evaluated as a fixer.
- **Verify its reproduction claims.** On the 266-task benchmark, 249 handoffs
  claimed a verified reproduction but only 50 of those claims were genuine; the
  system replays every claim in a sandbox and strips false ones before any fixer
  sees the handoff. Deployments should do the same rather than trust the
  `<repro>` block.
- **Evidence base is one benchmark's Python slice.** The headline system result
  rests on 266 Python tasks from a single benchmark; the language-transfer
  evidence covers the searcher alone, not the full routed system, and
  cross-benchmark generality is untested.
- **Moderate localization from a single pass.** Overall recall is 0.566 from a
  single sampled inference pass whose variance is uncharacterized; the system
  result shows this suffices for routing, not that the model is a
  state-of-the-art localizer.
- **Absolute rates are protocol-specific.** All solve rates are specific to the
  benchmark's capped, silent-budget, auto-submit tier; other tiers would change
  the numbers.
- **Hidden-state routing is checkpoint-tied.** The router's hidden-state
  features are tied to this exact checkpoint; retraining the searcher requires
  re-extracting hidden states and refitting the router heads.

## Citation

```bibtex
@misc{superscout2026,
  title         = {Scrouting: Cost-Aware Routing of Coding Agents by Scouting the Repository First},
  author        = {Ishaan Bhola and Adithyan Krishnan and Mukunda NS},
  year          = {2026},
  eprint        = {2608.04804},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE}
}
```
