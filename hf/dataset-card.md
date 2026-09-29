---
license: cc-by-4.0
task_categories:
  - text-generation
tags:
  - code
  - software-engineering
  - agent
  - sft
  - trajectories
size_categories:
  - 10K<n<100K
---


# SuperScout search corpus (SFT)

The supervised fine-tuning corpus behind SuperScout-7B, a 7B searcher that
explores a repository, localizes the fault, writes a failing reproduction, and
emits a structured handoff. The dataset contains 19,911 examples as built and
frozen; six rows carrying malformed tool-call wrappers are dropped at load time,
giving the 19,905 examples actually trained on. A further 9,478 examples (the
third-best trace per issue) were held back as a shelf and never trained on.

The trained model is at
[SuperAGI/SuperScout-7B](https://huggingface.co/SuperAGI/SuperScout-7B); the
paper, code, and evaluation receipts are at
[TransformerOptimus/superscout](https://github.com/TransformerOptimus/superscout).

## What one example is

Each example is one complete search episode sliced from an agent trajectory: the
agent explores a repository, localizes the fault, and writes a failing
reproduction, cut at the point the search is complete, with a synthesized
handoff-emission turn appended as the final supervised target. Within that
synthesized turn:

- the file list is extracted deterministically from the trace;
- the reproduction record is copied verbatim;
- only the free-form notes are model-written, by the open-weights gpt-oss-120b.

Loss is taken on assistant turns only, so environment output never contributes a
gradient.

## Sources and filtering

Traces are sliced from three public, openly licensed (CC-BY-4.0) trajectory
sets:

| Source | Role |
|---|---|
| Open-SWE-Traces | Go / TypeScript / JavaScript + Python |
| SWE-rebench-openhands | Python (OpenHands scaffold trajectories) |
| SWE-Hero | Python (OpenHands scaffold trajectories) |

Trajectories are success-filtered against the gold patch, keeping only traces
that actually found the right files, then deduplicated to the two highest
quality-ranked traces per issue so the corpus counts distinct bugs rather than
retellings of the same fix. 97.7% of examples carry a verified reproduction.

## Composition

| Language | Examples | Share |
|---|---|---|
| Python | 7,417 | 37.3% |
| Go | 7,304 | 36.7% |
| TypeScript | 3,932 | 19.7% |
| JavaScript | 1,258 | 6.3% |

Percentages are of the 19,911-example built set; the 6 load-time drops are not
attributed to a language. One epoch is 382.4M tokens, packed into 12,723 blocks
of 32k tokens at 91.7% fill.

## Contamination control

A 23-repository blocklist, decided before any training data was built, excludes
every repository appearing in SWE-bench Pro (11 repositories) or SWE-bench
Verified (12 repositories) from all training and calibration data. In addition,
all 450 issues of a held-out evaluation vault are excluded. Both exclusions were
verified by a programmatic gate on the frozen file (zero hits), not assumed. The
blocklist ships with the paper repository as `data/blocklist.json`.

All 266-task results use `ScaleAI/SWE-bench_Pro` at revision
`7ab5114912baf22bb098818e604c02fe7ad2c11f` (equivalently `config="v1"`). The
dataset's default config changed on 2026-09-22 to a newer 642-task set whose tasks
and task texts differ, so loading the default will not reproduce these rows.

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
