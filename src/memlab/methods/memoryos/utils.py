"""결정적(비-LLM) 헬퍼 — 원본 eval/utils.py에서 필요한 것만."""
from __future__ import annotations

import time
import uuid

import numpy as np

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def get_timestamp() -> str:
    """현재 시각 (원본과 동일 포맷)."""
    return time.strftime(TIMESTAMP_FMT, time.localtime())


def generate_id(prefix: str = "id") -> str:
    """접두사 + uuid4 앞 8자리 (원본과 동일)."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2 정규화, 영벡터는 그대로 (원본과 동일)."""
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
