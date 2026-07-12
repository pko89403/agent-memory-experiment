"""Central paths & pins for the memlab experiment harness.

The original reference repos (MemoryOS, graphiti) are NOT part of this
repository — only their pinned commit SHAs are. Run
``uv run scripts/fetch_reference.py`` to materialize them under
``external/``. Freezing each reference at one SHA is what makes the
baselines reproducible: anyone cloning this repo evaluates against exactly
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

GRAPHITI_DIR = EXTERNAL_DIR / "graphiti"
GRAPHITI_REPO_URL = "https://github.com/getzep/graphiti.git"
# tag v0.5.2 (2025-01-24) — arXiv 2501.13956v1 공개(01-23) 다음날의 paper-era 코드.
# 이후 HEAD는 멀티 DB 드라이버·프롬프트 변경 등으로 크게 드리프트해 참고자료로
# 부적합함을 확인하고 v0.5.2로 고정 (2026-07-09).
GRAPHITI_SHA = "0f50b74735c6936676a1448a6da7a820a21fa809"

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

# --- LLM (OpenAI 호환 — 기본은 로컬 LM Studio, Groq는 폴백) ---
# Groq 폴백용 키만 프로젝트 루트의 .env 파일에 GROQ_API_KEY=... 로 둔다 (.gitignore 대상).
# 모델은 여기 고정한다: baseline과 변형 실험 내내 같은 모델이어야 비교가 성립한다.
load_dotenv(PROJECT_ROOT / ".env")

# 프로바이더 선택: "lmstudio"(로컬 MLX, rate limit 없음) | "groq"(무료 API)
# - Groq 무료 티어: TPM 6K 벽 때문에 실용 불가 판정 (2026-07-04 실측)
# - 24GB M4에서 MLX 엔진(빠름)을 쓸 수 있는 로컬 런타임은 LM Studio뿐
#   (Ollama의 MLX 백엔드는 32GB 이상 전용)
LLM_PROVIDER = "lmstudio"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"  # 무료 티어에서 유일하게 RPD가 넉넉한 모델

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "qwen3.5-9b-mlx"  # MLX 4bit — 24GB에서 체급 대비 최상 품질

LLM_MODEL = LMSTUDIO_MODEL if LLM_PROVIDER == "lmstudio" else GROQ_MODEL
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 로컬 (원본 eval과 동일 모델)

# LLM 동시 호출 폭 — LM Studio를 `--parallel 4`로 로드한 것과 한 쌍.
# M4 Air 24GB 실측(2026-07-10): 현실적 프롬프트에서 4-way가 4.16x로 포화,
# 8-way는 이득 0. 파이프라인의 fan-out들이 이 값을 쓴다.
LLM_PARALLEL = 4

# --- Neo4j (zep 메소드의 그래프 저장소 — 논문의 "Neo4j's Lucene" 검색을 그대로 재현) ---
# 로컬 개발용 인스턴스 (brew, Community Edition 2026.06.0, 2026-07-09 설치).
# 비밀번호는 localhost 전용 개발값이라 비밀이 아님 — .env 대상 아님.
# 기동: `neo4j start` (러너 사전 조건 — LM Studio와 동급).
# 병렬 워커는 MEMLAB_NEO4J_URI로 자기 인스턴스를 지정한다 (CE는 DB가 하나라
# 워커 수만큼 인스턴스가 필요 — 대화 하나 = 그래프 하나의 실험 격리 유지).
NEO4J_URI = os.environ.get("MEMLAB_NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "memlab-local"
# qwen 계열의 thinking 비활성화는 LM Studio의 Prompt Template(Jinja) 최상단에
# {%- set enable_thinking = false %} 를 넣어 해결한다 — API 파라미터
# (chat_template_kwargs, /no_think 등)로는 불가함을 실측으로 확인 (2026-07-05).
# 원본 실험(비추론 gpt-4o-mini)과 성격을 맞추기 위해 반드시 꺼야 한다.


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY가 없습니다. 프로젝트 루트의 .env 파일에 "
            "GROQ_API_KEY=gsk_... 한 줄을 추가하세요."
        )
    return key
