"""A-MEM 파이프라인의 LLM ops — 이 부품의 모든 LLM 콜이 이 파일을 거친다.

    클래스     담당 (논문 절)                          소비자
    ────────  ─────────────────────────────────────  ────────
    NoteOps    construct — Ps1 note 구성 (§3.1)        store.py
               evolve — Ps2+Ps3 병합 콜 (§3.2-3.3):
               링크 판정 + 이웃 갱신

note당 2콜(construct + evolve)이 전부다. 답변 콜은 없다 — 합성
(nemori_amem)에서 답변은 Nemori 경로가 담당한다.

응답 모델은 전부 model_config = ConfigDict(extra="forbid") + 생성 필드
maxLength/maxItems 상한 (zep db2d680 관례 — LM Studio 문법 수준 절단
실측 확인). evolution 배열 상한 10은 evo_k=5의 2배 여유.

실패 정책 (memlab.llm.degrade — 공용 강등 헬퍼, 2026-07-25 승격):

    construct 실패  → keywords [], context "General", tags []   원본 동일
                      (analyze_content except 경로 자구)
    evolve 실패     → evolution 없음 (링크·갱신 0건, note는 저장)  원본 동일

논문·코드가 침묵하거나 결함이 있어 정한 것:
- suggested_connections의 번호 → uuid 매핑을 여기서 한다 (trust
  boundary: 범위 밖 번호는 버림). 원본은 응답 문자열을 links에 그대로
  넣어 링크가 영영 해소되지 않는 버그 (prompt_templates docstring 참고).
- update_neighbor의 배열 처리(검증 리뷰 A1): 원본 semantics를 따른다 —
  tags 배열 기준으로 돌되, context 배열이 짧으면 해당 이웃의 기존
  context를 유지하고 tags만 적용한다 (원본 memory_layer.py:844-850).
  잔여 위험: 모델이 배열을 짧게 내면 앞 순위 이웃부터 positional 배정
  된다 — 원본 동일 한계, 스키마로 교차 길이 강제 불가, dry-run 계측.
- strengthen의 tags_to_update가 비면 태그 유지(None) — 원본은 빈
  리스트로 덮어써 태그를 지워버리는데, 프롬프트가 "updated tags"를
  물었을 때 빈 응답은 갱신 의사 없음으로 읽는 게 자연스럽다.
- 이웃 직렬화는 원본 find_related_memories 형식("memory index:…") 자구.
  talk start time은 note.t(datetime)의 기본 표기 — 원본은 세션 문자열
  그대로였으나 정보 등가.
- temperature 0.7 (원본 get_completion 기본값), max_tokens 2000
  (빠듯한 max_tokens 금지 규범).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from memlab.llm import LLMProvider, degrade
from memlab.methods.amem import prompt_templates as prompts
from memlab.methods.amem.schema import MemoryNote

GENERATION_MAX_TOKENS = 2000

# --- 응답 모델 (프롬프트와 1:1, extra="forbid") ---

Label = Annotated[str, Field(max_length=60)]  # keyword·tag 공용 (실측 최장 ~20자)


class ConstructResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keywords: list[Label] = Field(
        ..., max_length=20,  # "at least three" — 정상 3~7개의 3배
        description="Specific, distinct keywords, ordered from most to least important",
    )
    context: str = Field(
        ..., max_length=500,  # "one sentence" — 정상 ~150자의 3배
        description="One sentence summarizing the main topic, key points, and purpose",
    )
    tags: list[Label] = Field(
        ..., max_length=20,
        description="Broad categories/themes for classification",
    )


class EvolveDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    should_evolve: bool
    actions: list[Literal["strengthen", "update_neighbor"]] = Field(..., max_length=2)
    suggested_connections: list[Annotated[int, Field(ge=0)]] = Field(
        ..., max_length=10,  # evo_k=5의 2배 여유
        description="Memory index numbers to connect to",
    )
    tags_to_update: list[Label] = Field(..., max_length=20)
    new_context_neighborhood: list[Annotated[str, Field(max_length=600)]] = Field(
        ..., max_length=10,  # 항목 600자 = construct의 context 상한(500)+갱신 여유
    )
    new_tags_neighborhood: list[Annotated[list[Label], Field(max_length=20)]] = Field(
        ..., max_length=10,
    )


@dataclass(frozen=True)
class EvolveOutcome:
    """evolve 콜의 판정을 store가 적용할 형태로 매핑한 것."""

    link_uuids: list[str] = field(default_factory=list)  # strengthen — 연결할 이웃
    tags: list[str] | None = None  # strengthen — 새 note의 갱신 태그 (None=유지)
    neighbor_updates: list[tuple[MemoryNote, str, list[str]]] = field(
        default_factory=list
    )  # update_neighbor — (이웃, new_context, new_tags)


# --- 직렬화 (placeholder 값의 형식은 전부 여기서 결정) ---


def _neighbors_text(neighbors: list[MemoryNote]) -> str:
    # 원본 find_related_memories(memory_system.py:307)의 탭 구분 형식 자구
    return "\n".join(
        f"memory index:{i}\ttalk start time:{n.t}\tmemory content: {n.content}"
        f"\tmemory context: {n.context}\tmemory keywords: {n.keywords}"
        f"\tmemory tags: {n.tags}"
        for i, n in enumerate(neighbors)
    )


# --- Note ops (§3.1-3.3) ---


class NoteOps:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def construct(self, content: str) -> tuple[list[str], str, list[str]]:
        """Ps1 — (keywords, context, tags) 생성 (§3.1)."""
        try:
            response = self._llm.chat_model(
                "",
                prompts.CONSTRUCT.format(content=content),
                ConstructResponse,
                temperature=0.7,
                max_tokens=GENERATION_MAX_TOKENS,
            )
            return list(response.keywords), response.context, list(response.tags)
        except Exception as e:
            return degrade("construct", ([], "General", []), e)

    def evolve(self, note: MemoryNote, neighbors: list[MemoryNote]) -> EvolveOutcome:
        """Ps2+Ps3 병합 콜 — 링크 판정 + 이웃 갱신 결정 (§3.2-3.3).

        construct가 끝난 note를 통째로 받는다 — 파라미터 나열 금지 규범.
        """
        try:
            response = self._llm.chat_model(
                "",
                prompts.EVOLVE.format(
                    content=note.content,
                    context=note.context,
                    keywords=note.keywords,
                    nearest_neighbors_memories=_neighbors_text(neighbors),
                    neighbor_number=len(neighbors),
                ),
                EvolveDecision,
                temperature=0.7,
                max_tokens=GENERATION_MAX_TOKENS,
            )
        except Exception as e:
            # 공유 싱글턴 금지 — frozen이어도 리스트 필드는 변이 가능해서
            # 콜마다 새 인스턴스를 만든다 (검증 리뷰 A15)
            return degrade("evolve", EvolveOutcome(), e)

        if not response.should_evolve:
            return EvolveOutcome()

        link_uuids: list[str] = []
        tags: list[str] | None = None
        neighbor_updates: list[tuple[MemoryNote, str, list[str]]] = []
        if "strengthen" in response.actions:
            link_uuids = [
                neighbors[i].uuid
                # dict.fromkeys: 중복 번호 제거 (순서 유지) — 안 하면 links·
                # 백링크가 이중 등록되어 link degree가 부풀려진다 (검증 리뷰 A4)
                for i in dict.fromkeys(response.suggested_connections)
                if 0 <= i < len(neighbors)  # 범위 밖 번호는 버림 (trust boundary)
            ]
            tags = list(response.tags_to_update) or None
        if "update_neighbor" in response.actions:
            # 원본 semantics (A1): tags 배열 기준, context 부족분은 기존 유지
            for i in range(min(len(neighbors), len(response.new_tags_neighborhood))):
                new_context = (
                    response.new_context_neighborhood[i]
                    if i < len(response.new_context_neighborhood)
                    else neighbors[i].context
                )
                neighbor_updates.append(
                    (neighbors[i], new_context, list(response.new_tags_neighborhood[i]))
                )
        return EvolveOutcome(link_uuids, tags, neighbor_updates)
