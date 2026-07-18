"""Nemori의 두 데이터베이스 — episodic D_e·semantic D_s의 인메모리 구현.

원본의 Postgres(본문)+Qdrant(벡터)는 순수 인프라 글루라 리스트 + numpy
cosine으로 대체한다 (러너가 대화당 method를 새로 만들어 영속성이 불필요).
LLM 콜이 없는 순수 저장·검색 계층이다.

논문 대응:

    여기                  논문
    ───────────────────  ─────────────────────────────────────────
    EpisodicStore         episodic database D_e (§3.2.3, §3.4)
    SemanticStore         semantic database D_s (§3.3, §3.4)
    .supersede            §3.2.3 "superseding U_k with M_ν"
    .evoke                Evoke(M_in, M)의 native 구현 (§3.3.1):
                          Top-Ks(S_r ∈ D_s | sim(v_in, u_r) > τ)
    .consolidate          Consolidate(K_in, M)의 A.2(Naive RAG) 구현 —
                          append만. native 판정(new/merge/conflict)은
                          미채택 (schema.py SemanticInsight 결정 참고)

논문이 침묵하거나 공개 코드와 다르게 정한 것:
- evoke의 τ threshold 필터는 공개 코드에 없다 (plain top-k,
  memory_system.py:170) — 논문 §3.3.1을 복원한 지점. τ·Ks 값은 호출자
  (method.py Config)가 준다.
- 유사도는 cosine 하나로 통일 — §3.2.3이 cosine을 명시하고, §3.3.1의
  sim()은 무명세라 같은 함수를 쓴다. 구현은 공용 인프라
  memlab.embedding.cosine_top_k (2026-07-17 합의 — 공통 수학은 상위 계층).
"""
from __future__ import annotations

import numpy as np

from memlab.embedding import cosine_top_k
from memlab.methods.nemori.schema import EpisodicMemory, SemanticInsight


class EpisodicStore:
    """D_e — 병합 후보 회수(§3.2.3, Ke)와 답변 검색(§3.4, k)이 소비."""

    def __init__(self) -> None:
        self._items: list[EpisodicMemory] = []
        self._vectors: list[np.ndarray] = []

    def add(self, episode: EpisodicMemory) -> None:
        self._items.append(episode)
        self._vectors.append(np.asarray(episode.embedding))

    def supersede(self, old: EpisodicMemory, merged: EpisodicMemory) -> None:
        """옛 episode를 병합본으로 대체 — 없는 uuid면 즉시 죽는다 (결함 노출)."""
        i = next(i for i, e in enumerate(self._items) if e.uuid == old.uuid)
        del self._items[i], self._vectors[i]
        self.add(merged)

    def search(self, query: np.ndarray, k: int) -> list[EpisodicMemory]:
        return [self._items[i] for i in cosine_top_k(self._vectors, query, k)]

    @property
    def items(self) -> tuple[EpisodicMemory, ...]:
        return tuple(self._items)  # 노트북 추적용 읽기 전용 뷰


class SemanticStore:
    """D_s — Evoke(§3.3.1, Ks·τ)와 답변 검색(§3.4, m)이 소비."""

    def __init__(self) -> None:
        self._items: list[SemanticInsight] = []
        self._vectors: list[np.ndarray] = []

    def consolidate(self, insights: list[SemanticInsight]) -> None:
        for insight in insights:
            self._items.append(insight)
            self._vectors.append(np.asarray(insight.embedding))

    def evoke(self, query: np.ndarray, ks: int, tau: float) -> list[SemanticInsight]:
        return [self._items[i] for i in cosine_top_k(self._vectors, query, ks, tau=tau)]

    def search(self, query: np.ndarray, m: int) -> list[SemanticInsight]:
        return [self._items[i] for i in cosine_top_k(self._vectors, query, m)]

    @property
    def items(self) -> tuple[SemanticInsight, ...]:
        return tuple(self._items)
