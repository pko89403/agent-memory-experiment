# memlab — Agent Memory Experiment Harness & Study Guide

*English · [한국어](README.ko.md)*

A project to **reimplement, verify, and compare** agent memory papers on the
LoCoMo benchmark. We keep adding methods — MemoryOS (arXiv:2506.06326), Zep
(arXiv:2501.13956, temporal knowledge graph), and Nemori (arXiv:2508.03341,
adaptive memory distillation) are reimplemented so far — with the end goal of
testing a cause-aware forgetting variant under the same conditions.

This repository is **both code and guide**:

> **Logic in the package, story in the notebooks.**

- `src/memlab/` — the code, where there has to be exactly one right answer (loader, metrics, methods, runner). Experiments run reproducibly from the CLI.
- `notebooks/` — the learning narrative (dataset exploration, metrics by hand, architecture dissection). Only the cheap, fast parts actually run.

## Directory Layout

```
├── src/memlab/
│   ├── config.py                 # all paths + reference SHA pins (the only committed trace of the references)
│   ├── data/                     # LoCoMo loader
│   ├── methods/                  # common interface: ingest(turn) / answer(question)
│   │   ├── memoryos/             # MemoryOS reimplementation (method #1, baseline)
│   │   ├── zep/                  # Zep reimplementation (temporal knowledge graph)
│   │   └── nemori/               # Nemori reimplementation (adaptive memory distillation)
│   ├── evaluation/               # set-F1 / standard F1 / BLEU-1, per-category report
│   └── run.py                    # experiment runner CLI (checkpoints, error isolation)
├── notebooks/                    # guide chapters 01–05
├── scripts/                      # fetch_data.py (data), fetch_reference.py (references)
├── external/                     # upstream repos (gitignored — committed only as SHA pins)
├── runs/                         # experiment results + config snapshots (gitignored)
└── tests/                        # loader integrity & scoring-function tests
```

## Design Decisions and Why

**1. Upstream repos aren't vendored into this repository.**
Committing someone else's repo wholesale muddies the licensing and the
history. Instead we commit just the SHA pins in `config.py`, and
`scripts/fetch_reference.py` clones and verifies at those SHAs. Anyone who
clones this gets exactly the same code every time — so we can always say which
code a given baseline number came from.

**1-1. Benchmark data comes from the original source.**
LoCoMo-10 originates from snap-research/locomo (Maharana et al., ACL 2024).
`scripts/fetch_data.py` downloads it at a pinned commit (`LOCOMO_SHA`) and
checks the contents against a SHA-256 hash (`LOCOMO_SHA256`) — pinning both
the revision and what's in it. The MemoryOS repo vendors a copy too, but
that's one paper's snapshot, not the benchmark. (Verified 2026-07-03: the two
files are byte-identical, so score comparability isn't affected.)

**2. Methods are reimplemented, not wrapped as adapters.**
Writing STM/MTM/LPM, heat, and eviction — or a knowledge graph, or a
prediction-error distillation pipeline — by hand is what makes it click, and
it means that when a cause-aware forgetting variant goes on later, we're
editing our own code, not someone else's research code. We read the originals
as textbooks, and as the source for the constants and prompts the papers
leave unspecified.

**3. The spec is the paper; the code is reference material.**
Upstream eval code diverges from its own paper (LFU deletion, whole-chain
migration, dead recency) and carries bugs (dropped utterances, the wrong
segments gaining heat). Since the LLM differs anyway (qwen3.5-9b-mlx vs.
gpt-4o-mini), there's no reason to reproduce the code's bugs — we implement
what the paper describes and borrow only the constants it omits (α·β·γ, and so
on) from the code. So our baseline is "the paper-spec method + qwen3.5-9b-mlx
(local LM Studio)," and we don't line it up against the paper's own table
numbers. It's only the reference point for the variant (forgetting)
experiments.

**4. Methods sit behind a common interface.**
Implement `ingest(turn)` / `answer(question)` and any memory system runs
through the same runner and scorer. Comparing baseline against variant under
**exactly the same conditions** is the whole point of this harness.

**5. A small one: `evaluation/`, not `eval/`.**
`eval` is a Python builtin, so a module by that name trips a shadowing
warning. The upstream repo uses `eval/`; we follow our own rule.

## Getting Started

```bash
uv sync                              # Python 3.12 + deps (pinned via uv.lock)
uv run scripts/fetch_data.py         # LoCoMo-10 from the original source (SHA-256 verified)
uv run scripts/fetch_reference.py    # prepare external/ references at pinned SHAs
```

The LLM is a local LM Studio instance (`localhost:1234`, `qwen3.5-9b-mlx`) —
no API key needed. Two things have to be set when you load the model: **context
length 16384** and **thinking turned off** (put
`{%- set enable_thinking = false %}` at the top of the Jinja prompt template).
The Groq free API is fallback-only — its 6K TPM limit makes it impractical in
practice (see the `config.py` comments). The original papers used gpt-4o-mini,
so rather than chase the paper's numbers we focus on self-baseline vs. variant
under one fixed model. Dataset exploration, metrics, and diff tests all run
with no LLM at all.

## Current Status — Full Baseline (LoCoMo 10 dialogues, 1,540 questions)

Per-method set_f1 over the full 10 dialogues on qwen3.5-9b-mlx, for Zep and
Nemori. MemoryOS is the conv-26 smoke value (full run still pending).
Artifacts live in `runs/` (not committed):

| category | n | MemoryOS* | Zep | Nemori |
|---|---|---|---|---|
| MULTI_HOP | 282 | 0.276 | 0.337 | **0.340** |
| TEMPORAL | 321 | 0.301 | 0.177 | **0.434** |
| OPEN_DOMAIN | 96 | 0.304 | 0.135 | 0.190 |
| SINGLE_HOP | 841 | 0.357 | **0.500** | 0.462 |
| ADVERSARIAL ↓ | 446 | 0.370 | 0.286 | **0.236** |
| **OVERALL (1–4)** | 1540 | 0.322 | 0.380 | **0.417** |

*MemoryOS is the conv-26 smoke value. ADVERSARIAL is scored against trap
answers, so lower is better (↓). Cost per dialogue: Zep runs ~10k calls / ~36h
(it processes message by message); Nemori runs ~370 calls (episode by episode)
— an order of magnitude apart, exactly as the paper's efficiency claim (§4.3)
predicts.

The standout: **Nemori's temporal 0.434 is 2.4× Zep's (0.177)**. Anchoring
relative references ("yesterday") to absolute dates inside the episode
narrative (paper §3.2.2) shows up straight in the score. Adversarial comes out
lowest (best), too — Nemori invents memories for trap questions less often
than the others. Single-hop runs the other way, favoring Zep (0.500): plain
fact lookup rewards a knowledge graph's precise search. The paper's temporal
edge and overall lead (Table 2) hold up here, at least in direction, on a
local 9B.

## Guide Roadmap (notebooks/)

| Chapter | Topic | What you learn |
|---|---|---|
| 01 | LoCoMo dataset | 10 samples / 5,882 turns / 1,986 QA. Verify the categories (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial) straight from the data via evidence counts and answer shapes |
| 02 | How to verify memory | The ingest → answer → score paradigm. repo-style set-F1 vs. standard F1 vs. BLEU-1, the batch scorer |
| 03 | Observing MemoryOS | Feed real dialogue fragments to MemoryOS and watch memory form, recall, promote, and forget (real LLM calls) |
| 04 | Observing Zep | Build a temporal knowledge graph — entity/fact extraction, bi-temporal stamps, communities, RRF search |
| 05 | Observing Nemori | Adaptive memory distillation — partition → narrative episode → merge, semantic distillation via predict-calibrate |

## Notes for Reproducing the Baseline (upstream code quirks)

Things we found reading the upstream `eval/` that differ from the paper or the
pypi package:

- `ShortTermMemory(max_capacity=1)` — the paper says 7, the pypi default is
  10. At 1, an evict fires every turn, which is what triggers a per-turn LLM call.
- The real LLM client is a single module-global `gpt_client` in `utils.py`.
  The `OpenAIClient` class's key/base_url config is dead code under openai 1.x.
- The model is hardcoded to `gpt-4o-mini` at **temperature=0.7** — so answers
  can shift from run to run. Any method comparison has to control for that variance.
- Upstream F1 is set-token based (it ignores token frequency), so it isn't
  standard F1, and BLEU-1 is missing from the repo entirely → our
  `evaluation/` computes all three.
- `get_embedding()` reloads SentenceTransformer on every call (a performance trap).
- 444 of the 446 cat5 (adversarial) questions have empty answers — you need a
  scoring-time exclusion option.
- **MTM `max_capacity=2000` against at most ~340 pages per dialogue → heat-based
  forgetting (deletion) never fires on this benchmark.** Only promotion
  (heat>5 → LPM) does. Recency is effectively constant during a run, too
  (γ=0.0001, keyed to real wall-clock time), so heat ≈ 0.8·N_visit +
  0.8·L_interaction. Forgetting experiments have to manufacture capacity
  pressure on their own, and that condition needs its own baseline.
- The pair-folding in `process_conversation` (turns → page grouping)
  mishandles sessions that open with speaker_b (124 of 272 sessions): **61
  turns get lost to overwrites** and **57 get mispaired across session
  boundaries** (their timestamps picking up the previous session's). Tracing
  the impact, **94 QA evidence items** point at lost (4) or polluted (90)
  turns — and temporal questions in particular depend on utterances that end
  up stored with the wrong timestamp. → Our reimplementation uses lossless
  pair folding (paper-first, no bug replication).

## Attribution & License

- **Code in this repository**: MIT ([LICENSE](LICENSE))
- **MemoryOS** ([BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS),
  Apache-2.0; paper arXiv:2506.06326): the reference implementation we
  reimplement from. The prompts in `prompt_templates.py` and the 90-dimension
  personality item list were borrowed and adapted from that repo (`eval/`,
  `memoryos-pypi/`). Apache-2.0 applies to the borrowed parts — full text at
  [licenses/MemoryOS-Apache-2.0.txt](licenses/MemoryOS-Apache-2.0.txt).
- **Zep** ([getzep/graphiti](https://github.com/getzep/graphiti), Apache-2.0;
  paper arXiv:2501.13956) and **Nemori**
  ([nemori-ai/nemori](https://github.com/nemori-ai/nemori), MIT; paper
  arXiv:2508.03341): reference implementations, pinned in `external/` and drawn
  on only where the papers leave constants or prompts unspecified.
- **LoCoMo** ([snap-research/locomo](https://github.com/snap-research/locomo),
  CC BY-NC 4.0; Maharana et al., ACL 2024): benchmark data.
  **The data file is not included in this repository** — `scripts/fetch_data.py`
  fetches it from the original source. Dialogue excerpts in notebook outputs
  are citations for non-commercial research purposes.
