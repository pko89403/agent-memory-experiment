"""로컬 임베딩 — 메소드들이 공유하는 범용 인프라 (LLMProvider와 같은 계층).

모델은 all-MiniLM-L6-v2 (원본 eval과 동일, API 비용 0).

원본 utils.get_embedding은 호출할 때마다 SentenceTransformer를 새로
로드했다 (호출당 ~1초 낭비). 여기서는 모델을 한 번만 로드해 재사용한다 —
임베딩 값은 동일하고 속도만 다르다.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from memlab.config import EMBEDDING_MODEL


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # 지연 import (무거움)

    return SentenceTransformer(model_name)


def embed(text: str, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """텍스트 → 384차원 벡터. MTM 등에 주입해 쓴다."""
    return _load_model(model_name).encode([text], convert_to_numpy=True)[0]
