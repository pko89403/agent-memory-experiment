"""Zep 파이프라인의 LLM ops — 이 메소드의 모든 LLM 콜이 이 파일을 거친다.

    클래스              담당                                  소비자(예정)
    ─────────────────  ───────────────────────────────────  ─────────────
    ExtractionOps       Sec 2.2 콜 6개 (extraction/resolution/  method.py
                        fact 병합 판정/temporal)
    CommunityOps        Sec 2.3 콜 2개 (map-reduce 요약, 이름)   communities.py
    AnswerGenerator     실험 경로의 chat agent (Sec 4) —        method.py
                        유일한 free-text 콜. 채점되는 출력에
                        형식 제약을 걸면 F1이 왜곡된다
                        (MemoryOS 때 결정 그대로).

응답 모델은 graphiti v0.5.2의 것을 따른다 (각 모델에 원본 클래스 주석).
예외는 EdgeDuplicate — HEAD 526dcad7의 병합판 (fact 판정 2콜→1콜 속도
결정, 2026-07-11). 전부 model_config = ConfigDict(extra="forbid") —
로컬 소형 모델의 스키마 이탈 방어 (하네스 규범).

논문이 침묵해 정한 것:
- 프롬프트에 넣는 후보 uuid는 로컬 별칭("0", "1", ...)이다. 원본은 진짜
  uuid를 넣지만, 36자 hex 옮겨적기는 로컬 소형 모델에서 오류 표면이 된다.
  프롬프트 자구와 응답 스키마는 불변 — 값의 어휘만 줄인다. 응답의 별칭이
  후보 집합에 없으면 no-duplicate/무시로 처리한다 (trust boundary 검증).
  uuid 필드의 Field description만 별칭 표기에 맞게 고쳤다.
- reflexion은 1회 콜로, 발견된 missed entities를 목록에 직접 추가한다.
  원본은 {custom_prompt} 슬롯으로 재추출 루프(MAX_REFLEXION_ITERATIONS=2)를
  돌지만, 논문 6.1.1 프롬프트에는 그 슬롯이 없다 — 자구 보존을 우선했다.
- ISO 파싱 실패는 None으로 처리한다 (날짜 하나 잃는 것이 ingest 중단보다
  낫다). tzinfo는 제거한다 — LoCoMo t_ref가 naive datetime이라 aware/naive
  비교 TypeError를 원천 차단.
- max_tokens는 extract_facts만 4000 (원본은 이 단계만 16384로 키웠는데 —
  PR #255 — 우리는 context 총 한도가 16384라 비례 축소), 나머지는 provider
  기본값. 빠듯한 max_tokens는 JSON을 자른다 (MemoryOS 때 실측).
- 후보 summary는 절단하지 않는다 — 프롬프트 길이는 dry-run에서 실측 후
  필요할 때만 손댄다.
- 후보 직렬화는 원본과 동일한 JSON이다. 처음엔 "uuid: 0 | name: ..."
  파이프 형식을 썼는데, qwen이 uuid 자리에 'node_uuid' 같은 placeholder를
  지어내는 것을 실측 (2026-07-10) — JSON key-value 형식에서는 alias를
  정확히 복사한다.
- invalidation 후보 직렬화에는 원본과 달리 valid_at/invalid_at을 넣는다.
  날짜 없는 직렬화로는 qwen이 명백한 모순(재직 vs 이직)에 무반응, 날짜를
  넣으면 정확히 판정함을 실측 (2026-07-10, 구형 개별 콜에서 — 병합 콜이
  승계). dedup 후보는 원본대로 날짜 없음 — "같은 정보" 판정에 날짜가 끼면
  같은 fact의 재언급을 다른 fact로 볼 위험. temporal overlap의 최종 판정은
  여전히 코드 소관 (원본 edge_operations.resolve_edge_contradictions,
  method.py가 이어받음).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from memlab.llm import LLMProvider
from memlab.methods.zep import prompt_templates as prompts
from memlab.methods.zep.schema import EntityNode, EpisodeNode, SemanticEdge

FACT_EXTRACTION_MAX_TOKENS = 4000
# 판정 응답(중복 판정·날짜·용의자 지목)은 정상일 때 ~200토큰 — 800은 4배
# 여유라 "빠듯한 max_tokens 금지" 규범과 충돌하지 않으면서, temp 0 폭주
# 생성(_fallback 경위 참고)이 타임아웃(300s)까지 가기 전에 잘리게 한다.
JUDGE_MAX_TOKENS = 800


def _fallback(op: str, default, error: Exception):
    """LLM 콜 실패를 op별 안전 기본값으로 강등 — 발화 하나의 판정 실패가
    대화(수 시간짜리 ingest)를 통째로 죽이지 않게 한다.

    근거 실측 (conv-26 발화 32, 2026-07-12): resolution 프롬프트가 temp 0
    폭주 생성에 빠지면 재시도 5회가 전부 같은 반복으로 타임아웃, 대화
    전체가 [fail] 처리됐다. 기본값은 전부 '판정 없음' 쪽(신규 노드/비중복/
    무날짜)이라 성공 경로의 semantics는 불변이다.
    """
    print(f"    [degrade] {op}: {error!r} — 기본값으로 진행")
    return default


# --- 응답 모델 (graphiti v0.5.2 응답 스키마 차용, extra="forbid") ---
# 모든 생성 필드에 maxLength/maxItems 상한 — LM Studio의 constrained decoding이
# 문법 수준에서 강제함을 실측 (36자 복사 요청이 10자에서, 8항목 요청이
# 3항목에서 절단, 2026-07-12). temp 0 폭주 생성(_fallback 경위)을 구조적으로
# 봉쇄한다. 상한은 정상 출력 최장의 4~10배 — 정상 경로는 절대 안 잘린다.

EntityName = Annotated[str, Field(max_length=100)]  # 실측 최장 ~40자


class ExtractedEntities(BaseModel):  # 원본: extract_nodes.ExtractedNodes
    model_config = ConfigDict(extra="forbid")
    extracted_node_names: list[EntityName] = Field(
        ..., max_length=30, description="Name of the extracted entity"
    )


class MissedEntities(BaseModel):  # 원본: extract_nodes.MissedEntities
    model_config = ConfigDict(extra="forbid")
    missed_entities: list[EntityName] = Field(
        ..., max_length=30, description="Names of entities that weren't extracted"
    )


class NodeDuplicate(BaseModel):  # 원본: dedupe_nodes.NodeDuplicate
    model_config = ConfigDict(extra="forbid")
    is_duplicate: bool = Field(..., description="true or false")
    uuid: str | None = Field(
        None, max_length=40,
        description="uuid of the existing node as listed, or null",
    )
    name: str = Field(
        ...,
        max_length=100,
        description=(
            "Updated name of the new node (use the best name between the "
            "new node's name, an existing duplicate name, or a combination "
            "of both)"
        ),
    )


class Summary(BaseModel):  # 원본: summarize_nodes.Summary
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(
        ..., max_length=4000,  # 프롬프트 규범 "under 500 words" ≈ 3000자
        description="Summary containing the important information from both summaries",
    )


class SummaryDescription(BaseModel):  # 원본: summarize_nodes.SummaryDescription
    model_config = ConfigDict(extra="forbid")
    description: str = Field(
        ..., max_length=400,
        description="One sentence description of the provided summary",
    )


class FactTriple(BaseModel):  # 원본: extract_edges.Edge (클래스명만 논문 어휘로)
    # triple = (주어, 술어, 목적어) 세 조각 + 원문 fact 문장 — 지식 그래프의
    # 최소 문장 단위. 그래프에선 source→target 엣지에 fact를 실은 것이 된다.
    model_config = ConfigDict(extra="forbid")
    relation_type: str = Field(..., max_length=60, description="RELATION_TYPE_IN_CAPS")
    source_entity_name: str = Field(..., max_length=100, description="name of the source entity")
    target_entity_name: str = Field(..., max_length=100, description="name of the target entity")
    fact: str = Field(..., max_length=600, description="extracted factual information")


class ExtractedFacts(BaseModel):  # 원본: extract_edges.ExtractedEdges
    model_config = ConfigDict(extra="forbid")
    edges: list[FactTriple] = Field(..., max_length=20)


class EdgeDuplicate(BaseModel):
    # 원본: dedupe_edges.EdgeDuplicate — 단 v0.5.2가 아니라 HEAD 526dcad7의
    # 병합판 (duplicate + contradicted 한 응답). 차용 경위는 prompt_templates
    # FACT_RESOLUTION 주석.
    model_config = ConfigDict(extra="forbid")
    duplicate_facts: list[int] = Field(
        ...,
        max_length=20,
        description=(
            "List of idx values of duplicate facts (only from EXISTING FACTS "
            "range). Empty list if none."
        ),
    )
    contradicted_facts: list[int] = Field(
        ...,
        max_length=20,
        description=(
            "List of idx values of contradicted facts (from full idx range). "
            "Empty list if none."
        ),
    )


class EdgeDates(BaseModel):  # 원본: extract_edge_dates.EdgeDates
    model_config = ConfigDict(extra="forbid")
    valid_at: str | None = Field(
        None,
        max_length=40,  # ISO 8601은 최장 32자
        description=(
            "The date and time when the relationship described by the edge "
            "fact became true or was established. YYYY-MM-DDTHH:MM:SS.SSSSSSZ "
            "or null."
        ),
    )
    invalid_at: str | None = Field(
        None,
        max_length=40,
        description=(
            "The date and time when the relationship described by the edge "
            "fact stopped being true or ended. YYYY-MM-DDTHH:MM:SS.SSSSSSZ "
            "or null."
        ),
    )


# --- 직렬화 헬퍼 (placeholder에 채우는 값의 형식은 전부 여기서 결정) ---


def _dialogue(episodes: list[EpisodeNode]) -> str:
    return "\n".join(f"{e.speaker}: {e.content}" for e in episodes) or "None"


def _unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for name in (n.strip() for n in names):
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            out.append(name)
    return out


def _entity_json(candidates: list[EntityNode]) -> tuple[str, dict[str, EntityNode]]:
    # resolution 후보는 원본처럼 uuid·name만 (node_operations.py:250) —
    # summary를 주면 소형 모델이 문맥 겹침을 동일성으로 오판해 과병합한다
    # ("LGBTQ support group"이 Caroline에 흡수, 2026-07-10 실측)
    alias = {str(i): node for i, node in enumerate(candidates)}
    text = json.dumps(
        [{"uuid": i, "name": node.name} for i, node in alias.items()],
        indent=2,
        ensure_ascii=False,
    )
    return text, alias


def _edge_entry(edge: SemanticEdge, with_dates: bool) -> dict:
    entry: dict = {"relation": edge.relation, "fact": edge.fact}
    if with_dates:
        entry["valid_at"] = edge.valid_at.isoformat() if edge.valid_at else None
        entry["invalid_at"] = edge.invalid_at.isoformat() if edge.invalid_at else None
    return entry


def _edge_text(edge: SemanticEdge, with_dates: bool = False) -> str:
    return json.dumps(_edge_entry(edge, with_dates), ensure_ascii=False)


def _edge_json(
    candidates: list[SemanticEdge], with_dates: bool, idx_offset: int = 0
) -> str:
    # 병합 판정의 연속 인덱스(upstream resolve_edge) — invalidation 목록은
    # dedup 목록이 끝난 idx에서 시작한다
    return json.dumps(
        [
            {"idx": idx_offset + i, **_edge_entry(edge, with_dates)}
            for i, edge in enumerate(candidates)
        ],
        indent=2,
        ensure_ascii=False,
    )


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


# --- ops ---


class ExtractionOps:
    """Sec 2.2의 LLM 콜 7개 — episode 하나를 그래프 조각으로 바꾸는 재료."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def extract_entities(
        self, episode: EpisodeNode, previous: list[EpisodeNode]
    ) -> list[str]:
        """6.1.1 + reflexion 보정 → entity 이름 목록 (Sec 2.2.1)."""
        prev = _dialogue(previous)
        cur = f"{episode.speaker}: {episode.content}"
        try:
            extracted = self.llm.chat_model(
                prompts.ENTITY_EXTRACTION_SYSTEM,
                prompts.ENTITY_EXTRACTION.format(
                    previous_messages=prev, current_message=cur
                ),
                ExtractedEntities,
            )
        except Exception as error:
            return _fallback("entity extraction", [], error)
        names = _unique(extracted.extracted_node_names)
        try:
            missed = self.llm.chat_model(
                prompts.ENTITY_REFLEXION_SYSTEM,
                prompts.ENTITY_REFLEXION.format(
                    previous_messages=prev,
                    current_message=cur,
                    extracted_entities=", ".join(names) or "None",
                ),
                MissedEntities,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as error:
            return _fallback("entity reflexion", names, error)
        return _unique(names + missed.missed_entities)

    def summarize_entity(
        self, name: str, episode: EpisodeNode, previous: list[EpisodeNode]
    ) -> str:
        """entity summary 추출 (Sec 2.2.1 언급, 프롬프트는 graphiti 차용).

        그래프를 보지 않고 대화(n=4 창)만 본다 — "이번 대화에서 이놈에
        대해 알게 된 것"의 새 메모. 기존 노드의 저장된 요약과 합치는 건
        병합(method.py ③)의 몫이다.
        """
        try:
            result = self.llm.chat_model(
                prompts.ENTITY_SUMMARY_SYSTEM,
                prompts.ENTITY_SUMMARY.format(
                    messages=_dialogue(previous + [episode]), entity_name=name
                ),
                Summary,
            )
        except Exception as error:
            return _fallback("entity summary", "", error)
        return result.summary

    def resolve_entity(
        self,
        name: str,
        candidates: list[EntityNode],
        episode: EpisodeNode,
        previous: list[EpisodeNode],
    ) -> tuple[EntityNode | None, str]:
        """6.1.2 → (병합 대상 기존 노드 | None, 최선의 이름) (Sec 2.2.1).

        문자열 비교가 아니라 LLM 판정인 이유: "Mel"↔"Melanie" 같은
        별명·변형은 문자열로 못 잡고, 반대로 철자가 같아도 다른
        놈(동명이인)일 수 있다 — 대화 맥락을 봐야 하는 문제다. 문제지는
        (대화 맥락, 후보 명단 uuid·name, 새 이름)이고 답안은 NodeDuplicate
        스키마 강제. 후보가 없으면 LLM 없이 즉시 신규 확정.

        방어 두 겹 (소형 모델 대비): 명단에 없는 uuid는 무시하고 신규
        취급, 이름 갱신은 duplicate일 때만 신뢰 — 신규 노드의 이름은
        추출된 그대로가 맞고, 모델이 지어낸 변형을 받을 이유가 없다.
        신규 노드도 name만 준다 — 원본은 summary를 dedupe와 병렬로 생성해
        판정 시점엔 빈 문자열이다 (node_operations.py:277-287). summary를
        채워 넣었더니 과병합이 실측돼(_entity_json 참고) 원본 식단으로 교정.
        """
        if not candidates:
            return None, name
        lines, alias = _entity_json(candidates)
        try:
            result = self.llm.chat_model(
                prompts.ENTITY_RESOLUTION_SYSTEM,
                prompts.ENTITY_RESOLUTION.format(
                    previous_messages=_dialogue(previous),
                    current_message=f"{episode.speaker}: {episode.content}",
                    existing_nodes=lines,
                    new_node=f"name: {name}",
                ),
                NodeDuplicate,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as error:
            return _fallback("entity resolution", (None, name), error)
        match = alias.get(result.uuid or "") if result.is_duplicate else None
        if match is None:
            return None, name
        best_name = result.name.strip() or match.name
        return match, best_name

    def extract_facts(
        self,
        episode: EpisodeNode,
        previous: list[EpisodeNode],
        entity_names: list[str],
    ) -> list[FactTriple]:
        """6.1.3 → 확정된 entity들 사이의 fact 목록 (Sec 2.2.2)."""
        try:
            result = self.llm.chat_model(
                prompts.FACT_EXTRACTION_SYSTEM,
                prompts.FACT_EXTRACTION.format(
                    previous_messages=_dialogue(previous),
                    current_message=f"{episode.speaker}: {episode.content}",
                    entities=", ".join(entity_names),
                ),
                ExtractedFacts,
                max_tokens=FACT_EXTRACTION_MAX_TOKENS,
            )
        except Exception as error:
            return _fallback("fact extraction", [], error)
        return result.edges

    def resolve_fact(
        self,
        edge: SemanticEdge,
        dedup_candidates: list[SemanticEdge],
        invalidation_candidates: list[SemanticEdge],
    ) -> tuple[SemanticEdge | None, list[SemanticEdge]]:
        """dedup(Sec 2.2.2) + invalidation 선별(Sec 2.2.3)을 LLM 1콜로.

        구형은 콜 2개(6.1.4 dedup, INVALIDATION v2 선별)였다 — triple당
        3콜이 전량 런을 지배해 upstream의 병합 프롬프트로 교체 (2026-07-11,
        prompt_templates FACT_RESOLUTION 주석). 판정의 소관은 그대로다:
        여기선 중복·모순을 지목만 하고, 승패와 invalid_at 판결은 날짜
        논리(method.py _invalidate_edges)의 몫.

        응답은 연속 idx 목록 둘 — duplicate는 dedup 후보 범위만 신뢰,
        contradicted는 두 목록 전체. 범위 밖 idx는 버린다 (trust boundary,
        entity resolution의 alias 방어와 동일). 후보가 둘 다 없으면 LLM
        없이 (None, []).
        dedup 항목엔 날짜 없음·invalidation 항목엔 날짜 포함의 비대칭은
        구형 개별 콜의 실측 결정 승계 (파일 docstring).
        """
        if not dedup_candidates and not invalidation_candidates:
            return None, []
        offset = len(dedup_candidates)
        try:
            result = self.llm.chat_model(
                prompts.FACT_RESOLUTION_SYSTEM,
                prompts.FACT_RESOLUTION.format(
                    existing_edges=_edge_json(dedup_candidates, with_dates=False),
                    invalidation_candidates=_edge_json(
                        invalidation_candidates, with_dates=True, idx_offset=offset
                    ),
                    new_edge=_edge_text(edge),
                ),
                EdgeDuplicate,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as error:
            return _fallback("fact resolution", (None, []), error)
        duplicate = next(
            (dedup_candidates[i] for i in result.duplicate_facts if 0 <= i < offset),
            None,
        )
        pool = dedup_candidates + invalidation_candidates
        contradicted = {
            pool[i].uuid: pool[i]
            for i in result.contradicted_facts
            if 0 <= i < len(pool)
        }
        return duplicate, list(contradicted.values())

    def extract_temporal(
        self,
        fact: str,
        episode: EpisodeNode,
        previous: list[EpisodeNode],
        t_ref: datetime,
    ) -> tuple[datetime | None, datetime | None]:
        """6.1.5 temporal extraction → (valid_at, invalid_at) (Sec 2.2.3)."""
        try:
            result = self.llm.chat_model(
                prompts.TEMPORAL_EXTRACTION_SYSTEM,
                prompts.TEMPORAL_EXTRACTION.format(
                    previous_messages=_dialogue(previous),
                    current_message=f"{episode.speaker}: {episode.content}",
                    reference_timestamp=t_ref.isoformat(),
                    fact=fact,
                ),
                EdgeDates,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as error:
            return _fallback("temporal extraction", (None, None), error)
        return parse_iso(result.valid_at), parse_iso(result.invalid_at)


class CommunityOps:
    """Sec 2.3의 LLM 콜 2개 — communities.py의 map-reduce 요약 재료.

    summarize는 entity 병합(Sec 2.2.1)에도 쓰인다 — 원본도 node 병합에
    같은 프롬프트를 재사용한다 (node_operations.py:300).
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def summarize(self, summaries: list[str]) -> str:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(summaries, 1))
        try:
            result = self.llm.chat_model(
                prompts.SUMMARY_COMBINE_SYSTEM,
                prompts.SUMMARY_COMBINE.format(summaries=numbered),
                Summary,
            )
        except Exception as error:
            # 결합 실패 → 첫 입력 유지 (병합이면 새 요약, community면 첫 member)
            return _fallback("summary combine", summaries[0] if summaries else "", error)
        return result.summary

    def name(self, summary: str) -> str:
        try:
            result = self.llm.chat_model(
                prompts.COMMUNITY_NAME_SYSTEM,
                prompts.COMMUNITY_NAME.format(summary=summary),
                SummaryDescription,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as error:
            return _fallback("community name", summary[:80], error)
        return result.description


class AnswerGenerator:
    """context 문자열(Sec 3 χ의 출력) + 질문 → free-text 답변 (Sec 4)."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate(
        self, context: str, question: str, speaker_a: str, speaker_b: str
    ) -> str:
        return self.llm.chat(
            prompts.ANSWER_SYSTEM.format(speaker_a=speaker_a, speaker_b=speaker_b),
            prompts.ANSWER_USER.format(
                context=context,
                question=question,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
            ),
            temperature=0.7,
        )
