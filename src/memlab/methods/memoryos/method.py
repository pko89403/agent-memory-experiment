"""MemoryOS — 논문 명세의 전체 조립 (MemoryMethod 소켓 구현).

    ingest(utterance):
      발화를 Q·R로 묶어 page 생성 → STM(chain 구성) → 차면 가장 오래된
      page를 MTM으로 FIFO 이관 (논문 3.2) → heat > τ 세그먼트를 LPM으로 승격
    answer(question):
      STM 전부 + MTM 2단계(top-m→top-k) + LPM(top-10) 검색 (논문 3.3)
      → 세 tier 통합 프롬프트로 생성 (논문 3.4)

논문이 침묵해 정한 것:
- Q·R 묶기: 논문은 user-AI 대화를 전제하므로 두 친구 대화는 speaker_a→Q,
  speaker_b→R로 묶는다. 유실 없는 방식 — 세션 경계에서 열린 page를 닫고,
  응답 없는 발화는 빈 칸으로 보존한다 (원본 eval의 유실 버그는 계승 안 함).
- 이미지 발화는 캡션을 텍스트에 병합: "(image description: ...)" (코드 차용)
- 승격 시 세그먼트 전체 page를 분석한다. 재승격은 L_interaction←0 리셋이
  자연 차단 (논문). 원본의 analyzed 플래그는 두지 않는다.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from memlab.embedding import embed as default_embed
from memlab.llm import LLMProvider
from memlab.methods.base import MemoryMethod, Utterance
from memlab.methods.memoryos.llm_ops import (
    AnswerGenerator,
    ChainLlmOps,
    PersonaLlmOps,
    SegmentLlmOps,
)
from memlab.methods.memoryos.long_term import LongTermMemory
from memlab.methods.memoryos.mid_term import MidTermMemory
from memlab.methods.memoryos.schema import Page
from memlab.methods.memoryos.short_term import ShortTermMemory


@dataclass(frozen=True)
class MemoryOSConfig:
    """하이퍼파라미터 — 기본값이 곧 논문 값.

    실험의 정체성이다: runs/ 아티팩트에 to_dict()로 직렬화되고,
    변형 실험은 이 객체 하나를 바꿔 만든다.
    """

    stm_capacity: int = 7  # STM 큐 길이
    mtm_capacity: int = 200  # MTM 최대 세그먼트 수
    kb_capacity: int = 100  # User KB 항목 수
    agent_traits_capacity: int = 100  # Agent Traits 항목 수
    heat_threshold: float = 5.0  # LPM 승격 임계 τ
    top_m: int = 5  # MTM 검색 1단계: 세그먼트
    top_k: int = 10  # MTM 검색 2단계: 페이지
    lpm_top_k: int = 10  # LPM 검색: KB·Agent Traits 각각

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryOS(MemoryMethod):
    def __init__(
        self,
        llm: LLMProvider,
        speaker_a: str,
        speaker_b: str,
        embed: Callable[[str], np.ndarray] = default_embed,
        config: MemoryOSConfig = MemoryOSConfig(),
    ):
        self.speaker_a = speaker_a
        self.speaker_b = speaker_b
        self.config = config

        self.stm = ShortTermMemory(
            chain_ops=ChainLlmOps(llm), max_capacity=config.stm_capacity
        )
        self.mtm = MidTermMemory(
            embed=embed, segment_ops=SegmentLlmOps(llm), max_capacity=config.mtm_capacity
        )
        self.lpm = LongTermMemory(
            embed=embed,
            kb_capacity=config.kb_capacity,
            agent_traits_capacity=config.agent_traits_capacity,
        )
        self.persona_ops = PersonaLlmOps(llm)
        self.answerer = AnswerGenerator(llm)

        self._open_page: Page | None = None  # 응답을 기다리는 page
        self._last_timestamp: str | None = None  # 세션 경계 감지용

    # ── ingest: 발화 → page → STM → MTM → LPM (논문 3.2) ──────────

    def ingest(self, utterance: Utterance) -> None:
        if self._last_timestamp is not None and utterance.timestamp != self._last_timestamp:
            self._flush_open_page()  # 세션이 바뀌면 열린 page를 닫는다 (경계 오염 방지)
        self._last_timestamp = utterance.timestamp

        text = utterance.text
        if utterance.blip_caption:
            text = f"{text} (image description: {utterance.blip_caption})"

        if utterance.speaker == self.speaker_a:
            self._flush_open_page()
            self._open_page = Page(user_input=text, timestamp=utterance.timestamp,
                                   agent_response="")
        else:
            if self._open_page is not None:
                self._open_page.agent_response = text
                self._process_page(self._open_page)
                self._open_page = None
            else:  # 세션이 speaker_b로 시작 — 빈 Q로 보존 (유실 없음)
                self._process_page(
                    Page(user_input="", agent_response=text, timestamp=utterance.timestamp)
                )

    # ── answer: 3원 검색 + 통합 생성 (논문 3.3, 3.4) ───────────────

    def answer(self, question: str) -> str:
        self._flush_open_page()  # 대화 끝에 열린 page가 남아 있으면 반영
        stm_pages = self.stm.get_all()  # 논문 3.3: STM은 전부
        mtm_hits = self.mtm.search(
            question, top_m=self.config.top_m, top_k=self.config.top_k
        )
        lpm_info = self.lpm.search(question, top_k=self.config.lpm_top_k)
        return self.answerer.generate(
            question,
            self.speaker_a,
            self.speaker_b,
            stm_pages,
            [page for page, _ in mtm_hits],
            lpm_info,
        )

    # ── 내부 ─────────────────────────────────────────────────────

    def _flush_open_page(self) -> None:
        if self._open_page is not None:
            self._process_page(self._open_page)
            self._open_page = None

    def _process_page(self, page: Page) -> None:
        self.stm.add_page(page)  # chain 구성 포함
        while self.stm.is_full():  # 논문 3.2: FIFO로 가장 오래된 page 이관
            self.mtm.add_page(self.stm.pop_oldest())
        self._promote_hot_segments()

    def _promote_hot_segments(self) -> None:
        """논문 3.2: heat > τ 세그먼트 → LPM 갱신 → L_interaction 리셋."""
        for segment in self.mtm.hot_segments(self.config.heat_threshold):
            profile = self.persona_ops.analyze_personality(
                self.lpm.user_profile, segment.details
            )
            self.lpm.set_user_profile(profile)
            facts, agent_knowledge = self.persona_ops.extract_knowledge(segment.details)
            for fact in facts:
                self.lpm.add_user_fact(fact)
            for knowledge in agent_knowledge:
                self.lpm.add_agent_trait(knowledge)
            self.mtm.reset_after_promotion(segment)
