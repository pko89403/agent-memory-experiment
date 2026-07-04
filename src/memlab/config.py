"""Central paths & pins for the memlab experiment harness.

The original MemoryOS repo is NOT part of this repository — only this pinned
commit SHA is. Run ``uv run scripts/fetch_reference.py`` to materialize it
under ``external/``. Freezing the reference at one SHA is what makes the
baseline reproducible: anyone cloning this repo evaluates against exactly
the same upstream code we studied.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- external reference (read-only study material + differential-test oracle) ---
EXTERNAL_DIR = PROJECT_ROOT / "external"
MEMORYOS_DIR = EXTERNAL_DIR / "MemoryOS"
MEMORYOS_REPO_URL = "https://github.com/BAI-LAB/MemoryOS.git"
MEMORYOS_SHA = "1d717060350931af33d1d0dc3d4e50a72c125a48"  # upstream main, 2026-04-28

# --- data (canonical source, NOT the MemoryOS vendored copy) ---
# LoCoMo-10 originates from snap-research/locomo (Maharana et al., ACL 2024).
# We fetch it from there at a pinned commit and verify its checksum.
# Note: MemoryOS's eval/locomo10.json is byte-identical (verified 2026-07-03),
# so scores stay comparable — but the benchmark's provenance is the original repo.
LOCOMO_REPO = "snap-research/locomo"
LOCOMO_SHA = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"  # upstream main, pinned
LOCOMO_URL = (
    f"https://raw.githubusercontent.com/{LOCOMO_REPO}/{LOCOMO_SHA}/data/locomo10.json"
)
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_PATH = EXTERNAL_DIR / "locomo10.json"

# --- experiment outputs ---
RUNS_DIR = PROJECT_ROOT / "runs"

# --- LLM (Groq free API, OpenAI 호환) ---
# 키는 프로젝트 루트의 .env 파일에 GROQ_API_KEY=... 로 둔다 (.gitignore 대상).
# 모델은 여기 고정한다: baseline과 변형 실험 내내 같은 모델이어야 비교가 성립한다.
load_dotenv(PROJECT_ROOT / ".env")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# llama-3.1-8b-instant: 무료 티어에서 전량 실행이 가능한 유일한 모델
# (일일 14.4K 요청 / 500K 토큰 — 나머지는 1K 요청 벽에 걸림)
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 로컬 (원본 eval과 동일 모델)
# 추론(thinking) 모델은 <think>가 답에 섞이고 토큰을 ~9배 쓰므로 꺼서
# 원본 실험(비추론 gpt-4o-mini)과 성격을 맞춘다. 비추론 모델엔 불필요.
LLM_EXTRA_BODY = (
    {"reasoning_effort": "none"} if "qwen3" in LLM_MODEL else {}
)


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY가 없습니다. 프로젝트 루트의 .env 파일에 "
            "GROQ_API_KEY=gsk_... 한 줄을 추가하세요."
        )
    return key
