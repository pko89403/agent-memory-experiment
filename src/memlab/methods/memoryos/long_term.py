"""LPM — 논문 3.1의 Long-term Persona Memory (dual persona).

논문 명세:
- User Persona: ① User Profile (static: 성별·이름·출생 등)
                ② User KB (동적 사실, 100칸)
                ③ User Traits (90차원 3분류: basic needs & personality /
                   AI alignment / content interest tags — Li et al. 2025)
- Agent Persona: ① Agent Profile (고정 역할 self-description)
                 ② Agent Traits (동적 속성, 예: 추천 아이템, 100칸)
- 검색: User KB와 Agent Traits에서 각각 query와 가장 관련 높은 top-10.
  Profile(traits 포함)은 검색 없이 전량 활용.

논문이 침묵해 정한 것:
- 100칸 초과 시 밀림 방식 — FIFO (원본 코드 차용: 가장 오래된 항목 소멸)
- User Traits(90차원 등급)는 pypi 구현체 방식대로 **프로필 텍스트 안에**
  유지한다 — 승격 때마다 LLM이 90차원 근거로 프로필을 재작성(진화)하고,
  생성 프롬프트에 프로필이 통째로 들어가므로 "전량 활용"이 충족된다.

임베딩은 주입받아 항목 저장 시 미리 계산해 둔다 (검색은 cosine top-k).
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from memlab.methods.memoryos.utils import normalize_vector


@dataclass
class _Entry:
    text: str
    embedding: np.ndarray = field(repr=False)


class _EmbeddedFifo:
    """텍스트 + 임베딩을 함께 보관하는 고정 용량 FIFO."""

    def __init__(self, embed: Callable[[str], np.ndarray], capacity: int):
        self.embed = embed
        self.entries: deque[_Entry] = deque(maxlen=capacity)

    def add(self, text: str) -> None:
        vec = normalize_vector(self.embed(text))
        self.entries.append(_Entry(text=text, embedding=vec))

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[str]:
        if not self.entries:
            return []
        sims = [(float(np.dot(e.embedding, query_vec)), e.text) for e in self.entries]
        sims.sort(key=lambda t: t[0], reverse=True)
        return [text for _, text in sims[:top_k]]

    def all_texts(self) -> list[str]:
        return [e.text for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)


class LongTermMemory:
    def __init__(
        self,
        embed: Callable[[str], np.ndarray],
        kb_capacity: int = 100,  # 논문: User KB 100 entries
        agent_traits_capacity: int = 100,  # 논문: Agent Traits 100 entries
        agent_profile: str = "",  # 고정 self-description (실험에선 비워둠)
    ):
        self.embed = embed
        # User Persona
        self.user_profile: str = ""  # 90차원 traits를 포함하는 진화 프로필
        self.user_kb = _EmbeddedFifo(embed, kb_capacity)
        # Agent Persona
        self.agent_profile: str = agent_profile
        self.agent_traits = _EmbeddedFifo(embed, agent_traits_capacity)

    # ── 갱신 (승격된 세그먼트의 LLM 분석 결과를 받아 병합) ──────────

    def add_user_fact(self, fact: str) -> None:
        self.user_kb.add(fact)

    def add_agent_trait(self, trait: str) -> None:
        self.agent_traits.add(trait)

    def set_user_profile(self, profile: str) -> None:
        self.user_profile = profile

    # ── 검색 (논문 3.3) ──────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> dict:
        """KB·Agent Traits는 top-10 검색, User Profile은 전량."""
        q = normalize_vector(self.embed(query))
        return {
            "user_profile": self.user_profile,
            "user_kb": self.user_kb.search(q, top_k),
            "agent_traits": self.agent_traits.search(q, top_k),
        }
