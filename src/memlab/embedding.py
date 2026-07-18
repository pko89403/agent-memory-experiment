"""로컬 임베딩 + 벡터 검색 공용 수학 — 메소드들이 공유하는 범용 인프라
(LLMProvider와 같은 계층).

모델은 all-MiniLM-L6-v2 (원본 eval과 동일, API 비용 0).

원본 utils.get_embedding은 호출할 때마다 SentenceTransformer를 새로
로드했다 (호출당 ~1초 낭비). 여기서는 모델을 한 번만 로드해 재사용한다 —
임베딩 값은 동일하고 속도만 다르다.

cosine_top_k는 인메모리 벡터 검색의 공통 구현 (2026-07-17, nemori부터).
기존 메소드는 각자 방식 유지 — zep은 Neo4j vector index에 위임하고,
memoryos는 원본 재현 구조(normalize_vector + np.dot)라 그대로 둔다.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from einops import einsum  # einsum 표기는 사용자 선택 (2026-07-17, 검증 리뷰 N17)

from memlab.config import EMBEDDING_MODEL


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # 지연 import (무거움)

    return SentenceTransformer(model_name)


def embed(text: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """텍스트 → 384차원 벡터. MTM 등에 주입해 쓴다."""
    return _load_model(model_name).encode([text], convert_to_numpy=True)[0]


def cosine_top_k(
    vectors: list[np.ndarray], query: np.ndarray, k: int, tau: float | None = None
) -> list[int]:
    """cosine 유사도 내림차순 top-k의 인덱스. tau가 있으면 sim > tau만.

    임베딩은 비정규화 저장을 가정하고 여기서 norm으로 나눈다. top-k 후
    tau 필터 순서는 필터 후 top-k와 가환 — sim 내림차순에서 tau 미달이
    rank i에 있으면 그 뒤는 전부 미달이라 결과가 같다.
    """
    if not vectors:
        return []
    matrix = np.stack(vectors)
    # 1e-12는 zero-vector 가드 — norm 0(빈 문자열 임베딩 등)에서 0-나눗셈만
    # 막고 해당 sim은 0으로 떨어진다 (검증 리뷰 N10)
    sims = einsum(matrix, query, "n d, d -> n") / (
        np.maximum(np.linalg.norm(matrix, axis=1), 1e-12) * max(np.linalg.norm(query), 1e-12)
    )
    order = np.argsort(-sims)[:k]
    if tau is not None:
        order = [i for i in order if sims[i] > tau]
    return [int(i) for i in order]
