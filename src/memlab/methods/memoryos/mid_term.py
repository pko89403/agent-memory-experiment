"""MTM — 논문 3.1 Segmented Paging 저장 + 3.2 heat 기반 갱신 + 3.3 2단계 검색.

논문 명세 (괄호는 식 번호):
- segment_i = {page | F_score(page, segment_i) > θ}                    (2)
- F_score = cos(e_s, e_p) + Jaccard(K_s, K_p),  Jaccard = |∩|/|∪|      (3)
- Heat = α·N_visit + β·L_interaction + γ·R_recency                     (4)
  R_recency = exp(−Δt/μ), μ = 1e7초
- 세그먼트 수가 최대(200)를 넘으면 heat 최저 세그먼트 삭제
- 검색: F_score로 top-m 세그먼트 → 그 안에서 임베딩 유사도로 top-k page.
  검색 히트 시 해당 세그먼트의 N_visit·R_recency 갱신
- LPM 승격 후 L_interaction ← 0 (heat 급락 → 재승격 차단)
- 세그먼트의 내용은 "summarized by a LLM based on the related dialogue
  pages" — 대표(summary, e_s, K_s)는 member page들의 LLM 요약에서 나온다.

논문이 침묵해 정한 것:
- α=0.8, β=0.8, γ=0.0001 — 원본 코드의 상수 차용
- 세그먼트 요약의 갱신 주기 — 논문 미명세. 페이지가 병합될 때마다
  재요약한다 (요약이 항상 member 전체를 반영하는 가장 원칙적 해석.
  비용: 병합 1회당 LLM 1회)
- 페이지 임베딩 텍스트 포맷은 한 가지로 통일:
  "User: {Q} Assistant: {R}" (원본은 경로마다 3가지 포맷이 섞여 있었다)

외부 의존:
- embed(텍스트→벡터) — 단일 순수 함수라 콜러블로 주입
- LLM 연산(키워드·세그먼트 요약) — SegmentOps 인터페이스로 주입
  (MTM이 필요한 만큼만 선언한 소비자 소유 인터페이스)
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Protocol

import numpy as np

from memlab.methods.memoryos.schema import Page, Segment, page_text
from memlab.methods.memoryos.utils import (
    TIMESTAMP_FMT,
    generate_id,
    get_timestamp,
    normalize_vector,
)


class SegmentOps(Protocol):
    """세그먼트 유지에 필요한 LLM 연산."""

    def extract_keywords(self, text: str) -> set[str]:
        """식 (3)의 키워드 집합 K."""
        ...

    def summarize_segment(self, pages: list[Page]) -> str:
        """세그먼트 내용 요약 — 논문: member page들의 LLM 요약."""
        ...


def jaccard(a: set[str], b: set[str]) -> float:
    """식 (3)의 F_Jaccard = |a∩b| / |a∪b|."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def recency(last_visit_time: str | None, now: str, mu: float = 1e7) -> float:
    """식 (4)의 R_recency = exp(−Δt/μ). Δt는 마지막 방문 이후 초."""
    if last_visit_time is None:
        return 1.0
    last_visit = datetime.strptime(last_visit_time, TIMESTAMP_FMT)
    current = datetime.strptime(now, TIMESTAMP_FMT)
    return float(np.exp(-(current - last_visit).total_seconds() / mu))


class MidTermMemory:
    def __init__(
        self,
        embed: Callable[[str], np.ndarray],
        segment_ops: SegmentOps,
        max_capacity: int = 200,  # 논문: "maximum length of segments in MTM is 200"
        theta: float = 0.6,  # 세그먼트 병합 임계 θ
        alpha: float = 0.8,  # heat: N_visit 가중치   (코드 차용)
        beta: float = 0.8,  # heat: L_interaction 가중치 (코드 차용)
        gamma: float = 0.0001,  # heat: R_recency 가중치  (코드 차용)
        mu: float = 1e7,  # recency 시간 상수 (논문)
        on_evict: Callable[[Segment], None] | None = None,  # 삭제 관찰 훅
    ):
        self.embed = embed
        self.segment_ops = segment_ops
        self.max_capacity = max_capacity
        self.theta = theta
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.mu = mu
        self.on_evict = on_evict
        self.segments: dict[str, Segment] = {}

    # ── 저장: 식 (2)(3)에 따른 페이지 편입 ─────────────────────────

    def add_page(self, page: Page, now: str | None = None) -> str:
        """페이지를 F_score 최적 세그먼트에 편입하거나 새 세그먼트를 만든다.

        논문 (2)의 집합 표기를 이렇게 해석한다: F_score가 θ를 넘는 세그먼트
        중 최고점에 편입, 없으면 새 세그먼트. 반환값은 세그먼트 id.
        """
        now = now or get_timestamp()
        self._ensure_page_features(page)

        best_segment, best_score = None, float("-inf")
        for segment in self.segments.values():
            score = self.f_score(page, segment)
            if score > best_score:
                best_segment, best_score = segment, score

        if best_segment is not None and best_score > self.theta:
            best_segment.details.append(page)
            best_segment.L_interaction += 1
            self._refresh_summary(best_segment)
        else:
            best_segment = self._create_segment(page, now)

        self.heat(best_segment, now)
        if len(self.segments) > self.max_capacity:
            self.evict_lowest_heat(now)
        return best_segment.id

    # ── 검색: 논문 3.3의 2단계 (top-m 세그먼트 → top-k 페이지) ─────

    def search(
        self,
        query: str,
        top_m: int = 5,
        top_k: int = 10,
        now: str | None = None,
    ) -> list[tuple[Page, float]]:
        if not self.segments:
            return []
        now = now or get_timestamp()
        query_embedding = normalize_vector(self.embed(query))
        query_keywords = self.segment_ops.extract_keywords(query)

        # 1단계: F_score(질의 vs 세그먼트 대표)로 top-m
        top_segments = sorted(
            self.segments.values(),
            key=lambda segment: self._query_score(segment, query_embedding, query_keywords),
            reverse=True,
        )[:top_m]

        # 2단계: 후보 세그먼트의 페이지를 임베딩 유사도로 전역 top-k
        candidates: list[tuple[Page, float, Segment]] = []
        for segment in top_segments:
            for page in segment.details:
                embedding = np.asarray(page.page_embedding, dtype=np.float32)
                similarity = float(np.dot(embedding, query_embedding))
                candidates.append((page, similarity, segment))
        candidates.sort(key=lambda item: item[1], reverse=True)
        hits = candidates[:top_k]

        # 검색 히트 세그먼트의 N_visit·R_recency 갱신 (논문 3.3)
        for segment in {segment.id: segment for _, _, segment in hits}.values():
            segment.N_visit += 1
            segment.last_visit_time = now
            self.heat(segment, now)

        return [(page, similarity) for page, similarity, _ in hits]

    # ── forgetting과 승격: 논문 3.2 ────────────────────────────────

    def evict_lowest_heat(self, now: str | None = None) -> Segment | None:
        """heat 최저 세그먼트 삭제 (논문: segment deletion by Heat)."""
        if not self.segments:
            return None
        now = now or get_timestamp()
        coldest = min(self.segments.values(), key=lambda s: self.heat(s, now))
        del self.segments[coldest.id]
        if self.on_evict is not None:
            self.on_evict(coldest)
        return coldest

    def hot_segments(self, threshold: float = 5.0, now: str | None = None) -> list[Segment]:
        """LPM 승격 후보: heat > τ."""
        now = now or get_timestamp()
        return [s for s in self.segments.values() if self.heat(s, now) > threshold]

    def reset_after_promotion(self, segment: Segment, now: str | None = None) -> None:
        """논문: 승격 후 L_interaction ← 0 → heat 급락 → 재승격 차단."""
        segment.L_interaction = 0
        self.heat(segment, now or get_timestamp())

    # ── 점수 함수: 식 (3)(4) ──────────────────────────────────────

    def f_score(self, page: Page, segment: Segment) -> float:
        page_embedding = np.asarray(page.page_embedding, dtype=np.float32)
        segment_embedding = np.asarray(segment.summary_embedding, dtype=np.float32)
        cosine = float(np.dot(segment_embedding, page_embedding))  # 둘 다 정규화됨
        return cosine + jaccard(set(segment.summary_keywords), set(page.page_keywords))

    def heat(self, segment: Segment, now: str | None = None) -> float:
        now = now or get_timestamp()
        segment.R_recency = recency(segment.last_visit_time, now, self.mu)
        segment.H_segment = (
            self.alpha * segment.N_visit
            + self.beta * segment.L_interaction
            + self.gamma * segment.R_recency
        )
        return segment.H_segment

    # ── 내부 ─────────────────────────────────────────────────────

    def _ensure_page_features(self, page: Page) -> None:
        """MTM 입장에 필요한 필드(id·임베딩·키워드)를 채운다."""
        if page.page_id is None:
            page.page_id = generate_id("page")
        if page.page_embedding is None:
            page.page_embedding = normalize_vector(self.embed(page_text(page))).tolist()
        if page.page_keywords is None:
            page.page_keywords = sorted(self.segment_ops.extract_keywords(page_text(page)))

    def _query_score(
        self, segment: Segment, query_embedding: np.ndarray, query_keywords: set[str]
    ) -> float:
        """검색 1단계 점수 — 식 (3)을 질의에 적용."""
        embedding = np.asarray(segment.summary_embedding, dtype=np.float32)
        cosine = float(np.dot(embedding, query_embedding))
        return cosine + jaccard(set(segment.summary_keywords), set(query_keywords))

    def _create_segment(self, page: Page, now: str) -> Segment:
        segment = Segment(
            id=generate_id("segment"),
            summary="",
            summary_keywords=[],
            summary_embedding=[],
            details=[page],
            L_interaction=1,
            timestamp=now,
        )
        self._refresh_summary(segment)
        self.segments[segment.id] = segment
        return segment

    def _refresh_summary(self, segment: Segment) -> None:
        """논문: 세그먼트 내용 = member page들의 LLM 요약. e_s·K_s 동기화."""
        segment.summary = self.segment_ops.summarize_segment(segment.details)
        segment.summary_keywords = sorted(self.segment_ops.extract_keywords(segment.summary))
        segment.summary_embedding = normalize_vector(self.embed(segment.summary)).tolist()
