"""원본 eval/utils.py 중 결정적(비-LLM) 헬퍼의 재구현."""
from __future__ import annotations

import time
import uuid


def get_timestamp() -> str:
    """원본과 동일: 현재 시각을 'YYYY-MM-DD HH:MM:SS'로."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def generate_id(prefix: str = "id") -> str:
    """원본과 동일: 접두사 + uuid 앞 8자리."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
