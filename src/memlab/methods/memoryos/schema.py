"""MemoryOS 고유 스키마 — 논문 기호와 1:1 대응.

    page       = {Q_i, R_i, T_i}                   (논문 3.1, STM의 저장 단위)
    page_chain = {Q_i, R_i, T_i, meta_chain_i}     (chain 필드가 채워진 page)
    segment    = 같은 토픽 page들의 묶음             (MTM의 저장 단위)

생애 단계를 Optional 필드로 명시한다 — None = 아직 그 단계 전.

주의: 원본 코드는 segment를 "session"이라고 부른다 (LoCoMo의 대화 세션과
전혀 다른 것!). 우리는 논문 용어인 segment를 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields


class _ToDictMixin:
    def to_dict(self) -> dict:
        """직렬화·디버깅용: None 필드(아직 안 온 생애 단계)는 뺀 dict."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "details":  # Segment.details: list[Page]
                value = [page.to_dict() for page in value]
            out[f.name] = value
        return out


@dataclass
class Page(_ToDictMixin):
    # ── 논문의 {Q, R, T} — 탄생 시점(pair folding)에 채워짐 ──
    user_input: str  # Q
    agent_response: str  # R
    timestamp: str | None = None  # T (없으면 STM이 현재 시각으로 채움)

    # ── chain 단계 — STM 진입 시 채워짐 (논문 식 1) ──
    page_id: str | None = None
    meta_info: str | None = None  # 논문의 meta_chain
    pre_page: str | None = None  # 체인 링크 (이전 page_id)
    next_page: str | None = None

    # ── MTM 입장 시 채워짐 ──
    page_embedding: list[float] | None = None
    page_keywords: list[str] | None = None


def page_text(page: Page) -> str:
    """page의 정본 텍스트 표현 — 임베딩·키워드·프롬프트가 전부 이 포맷을 쓴다.

    (원본 코드는 경로마다 3가지 포맷이 섞여 있었다 — 한 곳으로 통일.)
    """
    return f"User: {page.user_input} Assistant: {page.agent_response}"


@dataclass
class Segment(_ToDictMixin):
    """MTM의 저장 단위. 원본 코드명 "session". 필드명은 논문 기호 기준."""

    id: str
    summary: str  # 논문: "summarized by a LLM based on the related dialogue pages"
    summary_keywords: list[str]  # K_s
    summary_embedding: list[float]  # e_s
    details: list[Page] = field(default_factory=list)
    L_interaction: int = 0  # segment 내 page 수 (heat의 β항)
    R_recency: float = 1.0  # 최근성 (heat의 γ항) — heat() 계산 시 갱신되는 캐시
    N_visit: int = 0  # 검색 히트 횟수 (heat의 α항)
    H_segment: float = 0.0  # heat 점수 — heat() 계산 시 갱신되는 캐시
    timestamp: str | None = None  # 생성 시각
    last_visit_time: str | None = None
