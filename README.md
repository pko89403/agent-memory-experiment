# memlab — Agent Memory Experiment Harness & Study Guide

*English · [한국어](README.ko.md)*

A project to **reimplement, verify, and compare** agent memory papers on the
LoCoMo benchmark. Methods keep getting added — MemoryOS (arXiv:2506.06326),
Zep (arXiv:2501.13956, temporal knowledge graph), and Nemori
(arXiv:2508.03341, adaptive memory distillation) are reimplemented, and the
end goal is to experiment with a cause-aware forgetting variant under the same
conditions.

This repository is **both code and guide**:

> **Logic in the package, story in the notebooks.**

- `src/memlab/` — code that must have a single source of truth (loader, metrics, methods, runner). Experiments run via CLI, reproducibly.
- `notebooks/` — the learning narrative (dataset exploration, metrics by hand, architecture dissection). Only the cheap and fast parts execute.

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

**1. Upstream repos are not vendored into this repository.**
Committing someone else's repo wholesale pollutes licensing and history.
Instead we commit only the SHA pins in `config.py`, and
`scripts/fetch_reference.py` clones/verifies at those SHAs. Anyone, anytime,
gets exactly the same code — so we can always answer which code a baseline
number came from.

**1-1. Benchmark data comes from the original source.**
LoCoMo-10 originates from snap-research/locomo (Maharana et al., ACL 2024).
`scripts/fetch_data.py` downloads it at a pinned commit (`LOCOMO_SHA`) and
verifies the contents with a SHA-256 checksum (`LOCOMO_SHA256`) — a double
pin of both time and content. A copy is vendored in the MemoryOS repo too,
but that is "one paper's snapshot," not the benchmark. (Verified 2026-07-03:
the two files are byte-identical, so score comparability is unaffected.)

**2. Methods are reimplemented, not wrapped as adapters.**
Writing STM/MTM/LPM, heat computation, eviction — or a knowledge graph, or a
prediction-error distillation pipeline — by hand is what produces real
understanding, and it means that when a cause-aware forgetting variant gets
attached later, we modify *our* code, not someone else's research code. The
originals are read as textbooks, and as the source for constants and prompts
the papers stay silent on.

**3. The spec for a reimplementation is the paper. Code is reference material.**
Upstream eval code diverges from its paper (LFU deletion, whole-chain
migration, dead recency) and has bugs (dropped utterances, wrong segments
gaining heat). Since the LLM differs anyway (qwen3.5-9b-mlx vs gpt-4o-mini),
there is no reason to replicate the code bugs — we **implement as the paper
describes** and borrow only the constants the paper omits (α·β·γ, etc.) from
the code. So our baseline is "the paper-spec method + qwen3.5-9b-mlx (local
LM Studio)," and we do not compare directly against the paper's table
numbers. It serves only as the reference line for variant (forgetting)
experiments.

**4. Methods sit behind a common interface.**
Implement `ingest(turn)` / `answer(question)` and any memory system is
evaluated by the same runner and scorer. Baseline and variant being compared
under **exactly the same conditions** is the whole reason this harness exists.

**5. Minor but: `evaluation/`, not `eval/`.**
`eval` is a Python builtin, so a module of that name triggers a shadowing
warning. The upstream repo uses `eval/`; we follow our own rule.

## Getting Started

```bash
uv sync                              # Python 3.12 + deps (pinned via uv.lock)
uv run scripts/fetch_data.py         # LoCoMo-10 from the original source (SHA-256 verified)
uv run scripts/fetch_reference.py    # prepare external/ references at pinned SHAs
```

The LLM is a local LM Studio (`localhost:1234`, `qwen3.5-9b-mlx`) — no API
key needed. Two things are required when loading the model: **context length
16384** and **thinking disabled** (put `{%- set enable_thinking = false %}`
at the top of the Jinja Prompt Template). The Groq free API is fallback-only
(measured impractical at the 6K TPM limit — see `config.py` comments). The
original papers used gpt-4o-mini, so we give up direct comparison to paper
numbers and focus on **self-baseline vs variant under the same model**.
Dataset exploration, metrics, and diff tests all run without an LLM.

## Current Status — Full Baseline (LoCoMo 10 dialogues, 1,540 questions)

Per-method set_f1 from running the full 10 dialogues on qwen3.5-9b-mlx for
Zep and Nemori. MemoryOS is the conv-26 smoke value (full run pending).
Artifacts: `runs/` (not committed):

| category | n | MemoryOS* | Zep | Nemori |
|---|---|---|---|---|
| MULTI_HOP | 282 | 0.276 | 0.337 | **0.340** |
| TEMPORAL | 321 | 0.301 | 0.177 | **0.434** |
| OPEN_DOMAIN | 96 | 0.304 | 0.135 | 0.190 |
| SINGLE_HOP | 841 | 0.357 | **0.500** | 0.462 |
| ADVERSARIAL ↓ | 446 | 0.370 | 0.286 | **0.236** |
| **OVERALL (1–4)** | 1540 | 0.322 | 0.380 | **0.417** |

*MemoryOS is the conv-26 smoke value. ADVERSARIAL is scored against trap
answers, so lower is better (↓). Cost per dialogue: Zep ~10k calls/~36h
(message-wise processing), Nemori ~370 calls (episode-wise) — an
order-of-magnitude difference, as the efficiency claim in the paper (§4.3)
predicts.

Pattern: **Nemori's temporal 0.434 is 2.4× Zep's (0.177)** — the design of
anchoring relative references ("yesterday") to absolute dates in the episode
narrative (paper §3.2.2) turns directly into score. Adversarial is also
lowest (=best): it fabricates memories for trap questions least often.
Conversely, single-hop favors Zep (0.500) — simple fact retrieval benefits
from a knowledge graph's precise search. The paper's claimed temporal
advantage and overall lead (Table 2) reproduce, direction intact, on a
local 9B.

## Guide Roadmap (notebooks/)

| Chapter | Topic | What you learn |
|---|---|---|
| 01 | LoCoMo dataset | 10 samples / 5,882 turns / 1,986 QA. Verify the categories (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial) directly from the data via evidence counts and answer shapes |
| 02 | How to verify memory | The ingest → answer → score paradigm. repo-style set-F1 vs standard F1 vs BLEU-1, the batch scorer |
| 03 | Observing MemoryOS | Feed real dialogue fragments to MemoryOS and watch memory form, recall, promote, and forget (real LLM calls) |
| 04 | Observing Zep | Build a temporal knowledge graph — entity/fact extraction, bi-temporal stamps, communities, RRF search |
| 05 | Observing Nemori | Adaptive memory distillation — partition → narrative episode → merge, semantic distillation via predict-calibrate |

## Notes for Reproducing the Baseline (upstream code quirks)

Points found while reading the upstream `eval/` that differ from the paper /
pypi package:

- `ShortTermMemory(max_capacity=1)` — the paper says 7, the pypi default is
  10. This causes an evict every turn, which is why an LLM call happens per turn.
- The actual LLM client is a single module-global `gpt_client` in `utils.py`.
  The `OpenAIClient` class's key/base_url config is dead code under openai 1.x.
- The model is hardcoded `gpt-4o-mini` with **temperature=0.7** — meaning
  answers can differ run to run. Method-comparison experiments must control
  for this variance.
- The upstream F1 is set-token based (ignoring token frequency), so it
  differs from standard F1. BLEU-1 is absent from the upstream repo entirely
  → our `evaluation/` computes all three.
- `get_embedding()` reloads SentenceTransformer on every call (a performance trap).
- 444 of 446 cat5 (adversarial) questions have empty answers — a
  scoring-time exclusion option is needed.
- **MTM `max_capacity=2000` vs at most ~340 pages per dialogue → heat-based
  forgetting (deletion) never fires on the benchmark.** Only promotion
  (heat>5 → LPM) works. Also recency is effectively constant during a run
  (γ=0.0001, real wall-clock based), so heat ≈ 0.8·N_visit +
  0.8·L_interaction. Forgetting experiments must create capacity pressure
  separately, and that condition's baseline needs its own measurement.
- The pair-folding logic in `process_conversation` (turns → page grouping)
  mishandles sessions starting with speaker_b (124/272 sessions): **61 turns
  are lost to overwrites** and **57 turns are mispaired across session
  boundaries** (timestamps polluted with the previous session's). Impact
  analysis shows **94 QA evidence items** point at lost (4)/polluted (90)
  turns; temporal questions in particular require utterances stored with the
  wrong timestamp. → Our reimplementation uses lossless pair folding
  (paper-first strategy — no bug replication).

## Attribution & License

- **Code in this repository**: MIT ([LICENSE](LICENSE))
- **MemoryOS** ([BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS),
  Apache-2.0; paper arXiv:2506.06326): reference implementation for the
  reimplementation. The prompts in `prompt_templates.py` and the 90-dimension
  personality item list were borrowed and adapted from that repo (`eval/`,
  `memoryos-pypi/`). Apache-2.0 applies to the borrowed parts — full text at
  [licenses/MemoryOS-Apache-2.0.txt](licenses/MemoryOS-Apache-2.0.txt).
- **Zep** ([getzep/graphiti](https://github.com/getzep/graphiti), Apache-2.0;
  paper arXiv:2501.13956) and **Nemori**
  ([nemori-ai/nemori](https://github.com/nemori-ai/nemori), MIT; paper
  arXiv:2508.03341): reference implementations, pinned in `external/` and
  borrowed from only where the papers stay silent on constants and prompts.
- **LoCoMo** ([snap-research/locomo](https://github.com/snap-research/locomo),
  CC BY-NC 4.0; Maharana et al., ACL 2024): benchmark data.
  **The data file is not included in this repository** — `scripts/fetch_data.py`
  fetches it from the original source. Dialogue excerpts in notebook outputs
  are citations for non-commercial research purposes.
