"""Nemori 파이프라인의 LLM ops — 이 메소드의 모든 LLM 콜이 이 파일을 거친다.

    클래스            담당 (논문 절)                          소비자
    ───────────────  ─────────────────────────────────────  ──────────
    EpisodicOps       §3.2 콜 4개 — partition / narrate /     method.py
                      select_target / integrate
    SemanticOps       §3.3 콜 3개 — anticipate + distill      method.py
                      (2콜 연쇄), direct_distill (cold start)
    generate_answer   §3.4 — 유일한 free-text 답변 콜.        method.py
                      상태가 없어 함수다 (ponytail 리뷰).
                      채점되는 출력에 형식 제약을 걸면
                      F1이 왜곡된다 (MemoryOS 때 결정).

응답 모델은 전부 model_config = ConfigDict(extra="forbid") + 생성 필드
maxLength/maxItems 상한 — temp 0 폭주 생성의 구조적 봉쇄 (zep 커밋
db2d680 교훈, 상한은 정상 출력 최장의 3~10배라 정상 경로는 안 잘린다).
PartitionResponse.episodes의 상한 20은 폭주 봉쇄가 아니라 논문 형식화
그 자체다 (§3.2.1: n ≤ w = 20).

실패 정책 — 콜 하나의 실패가 대화(수 시간짜리 ingest)를 죽이지 않게
op별 안전 기본값으로 강등한다 (zep _fallback 관례):

    partition 실패     → 통짜 한 그룹 (topic "conversation")   원본 동일
    narrate 실패       → cue "Conversation (N messages)",      원본 동일
                         narrative = 대화 원문
    select_target 실패 → "new" (병합 안 함)                    원본 동일
    integrate 실패     → 병합 포기, 새 episode 별도 저장        원본 동일
    distill 경로 실패  → insight 0개 (episode는 이미 저장됨)    원본 동일
    answer 실패        → 전파 — 러너가 QA 단위 error로 기록

json_object 폴백(llm.py의 BadRequestError 경로)은 LM Studio에서 트리거
불가 실측 (zep 전량 런 무발생 + 스키마 전 기능 수용 확인) — Groq 폴백
경로에서만 키 무지시 프롬프트가 약점이 된다 (검증 리뷰 N16).

논문이 침묵해 정한 것 (직렬화·값의 형식은 전부 이 파일 소관):
- 메시지 나열의 timestamp는 LoCoMo date_time 문자열("1:56 pm on 8 May,
  2023")을 그대로 쓴다. 원본은 datetime 파싱 후 "%Y-%m-%d %H:%M:%S"로
  재조판하는데, 정보량이 같고 파싱 왕복만 늘어난다. NARRATIVE의 시간
  추출 지시가 "message metadata or content"라 형식은 무관.
- blip_caption은 "[Image: {caption}]"로 본문에 덧붙인다 — 원저자 LoCoMo
  하네스(evaluation/locomo/add.py:96)와 동일 형식 (2026-07-17 합의).
- partition의 커버리지 복구: 응답에 안 나온 index는 직전 index가 속한
  그룹에 편입한다 (첫 메시지가 빠지면 첫 그룹). §3.2.1의 "union covers"
  요건을 기계적으로 보장 — 프롬프트 신설 조항만으로는 로컬 9B가 새는
  것을 막을 수 없다. 범위 밖 index는 버린다 (원본 동일).
- select_target 후보는 번호(1-based)로 나열하고 번호로 응답받는다 —
  논문 형식화(idx ∈ {1..Ke} ∪ {-1})대로. 후보 narrative는 원본처럼
  200자 절단 (merger.py:181), 판정 자료는 cue·시각·길이가 주다.
- 병합 시각은 integrate 응답의 timestamp를 쓰되 파싱 실패면 두 episode
  중 이른 t — 원본(merger.py:136-152)과 동일한 우선순위.
- temperature는 원본 차용: partition 0.2 (segmenter.py:62), 생성·판정류
  0.7 (client.py:31 기본값), answer 0.0 (evaluation/locomo/search.py:168).
- max_tokens: partition 4096 (원본), 생성류 2000 (원본 기본값 — 빠듯한
  max_tokens 금지 규범과 일치), select_target만 800 (판정 응답은 정상
  ~100토큰 — zep JUDGE_MAX_TOKENS 관례, 폭주가 타임아웃 전에 잘리게).
- 답변 컨텍스트의 빈 섹션은 "None" — 빈 문자열 섹션은 소형 모델이 지시
  누락으로 오독한다 (zep _dialogue 관례).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from memlab.llm import LLMProvider, parse_iso
from memlab.methods.base import Utterance
from memlab.methods.nemori import prompt_templates as prompts
from memlab.methods.nemori.schema import EpisodicMemory, SemanticInsight

PARTITION_MAX_TOKENS = 4096
GENERATION_MAX_TOKENS = 2000
JUDGE_MAX_TOKENS = 800

CANDIDATE_PREVIEW_CHARS = 200  # 원본 merger.py:181의 후보 절단 길이


def _fallback(op: str, default, error: Exception):
    """LLM 콜 실패를 op별 안전 기본값으로 강등 (zep 관례)."""
    print(f"    [degrade] {op}: {error!r} — 기본값으로 진행")
    return default


# --- 응답 모델 (프롬프트와 1:1, extra="forbid") ---

class EpisodeGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    indices: list[Annotated[int, Field(ge=1)]] = Field(
        ..., max_length=40,  # 윈도우 w=20의 2배 여유
        description="List of message numbers (1-based) belonging to this episode",
    )
    topic: str = Field(
        ..., max_length=200,
        description="Brief, specific description of what this episode is about",
    )


class PartitionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episodes: list[EpisodeGroup] = Field(..., max_length=20)  # 논문: n ≤ w


class NarrativeResponse(BaseModel):
    # NARRATIVE와 INTEGRATE가 공유 — 둘 다 (c, N, t) 산출 (§3.2.2, §3.2.3)
    model_config = ConfigDict(extra="forbid")
    episodic_cue: str = Field(
        ..., max_length=300,  # "10-20 words" ≈ 150자의 2배
        description="A concise, descriptive title that accurately summarizes the theme (10-20 words)",
    )
    narrative_episode: str = Field(
        ..., max_length=6000,  # 실측 예시 ~700자의 8배 (윈도우 최대 20 메시지)
        description="A detailed third-person narrative description of the conversation",
    )
    timestamp: str = Field(
        ..., max_length=40,  # ISO 19자의 2배
        description='YYYY-MM-DDTHH:MM:SS format timestamp representing when this episode occurred',
    )


class MergeDecision(BaseModel):
    # 논문 형식화 그대로 단일 idx ∈ {1..Ke} ∪ {-1} (§3.2.3) — merge/new 별도
    # 필드는 target_index에서 파생 가능한 중복이고, 두 필드가 모순될 자유도만
    # 모델에 준다 (검증 리뷰 N8). 상한은 코드의 범위 검증이 담당 (N9).
    model_config = ConfigDict(extra="forbid")
    target_index: int = Field(
        ..., ge=-1,
        description="Number of the target candidate (1-based) to merge with, or -1 to keep as a separate new episode",
    )
    reason: str = Field(..., max_length=500, description="Brief explanation of the decision")


class InsightStatements(BaseModel):
    # DISTILL과 DIRECT_DISTILL이 공유 — 둘 다 fact 진술 목록 산출
    model_config = ConfigDict(extra="forbid")
    statements: list[Annotated[str, Field(max_length=400)]] = Field(
        ..., max_length=20,  # "quality over quantity" — 정상 2~6개의 3배+
        description="Self-contained factual statements, present tense, specific details",
    )


# --- 직렬화 헬퍼 (placeholder에 채우는 값의 형식은 전부 여기서 결정) ---


def utterance_text(u: Utterance) -> str:
    """발화 본문 — blip_caption은 원저자 하네스 형식으로 덧붙인다."""
    if u.blip_caption:
        return f"{u.text} [Image: {u.blip_caption}]"
    return u.text


def _numbered(utterances: tuple[Utterance, ...]) -> str:
    return "\n".join(
        f"{i}. [{u.timestamp}] {u.speaker}: {utterance_text(u)}"
        for i, u in enumerate(utterances, 1)
    )


def _dialogue(utterances: tuple[Utterance, ...]) -> str:
    # distill의 original_messages용 — 원본(semantic.py _extract_text)대로 무시각
    return "\n".join(f"{u.speaker}: {utterance_text(u)}" for u in utterances)


def _timestamped_dialogue(utterances: tuple[Utterance, ...]) -> str:
    # narrate의 conversation용 — 원본 format_conversation(prompts.py:350)의
    # "[ts] role: content" 형식. 시각을 빼면 NARRATIVE의 시간 추출이 찾을
    # 원천이 없어 프롬프트 예시 날짜(2024-01-15)를 베낀다 (2026-07-17 실측).
    return "\n".join(
        f"[{u.timestamp}] {u.speaker}: {utterance_text(u)}" for u in utterances
    )


def _time_range(t: datetime, n_messages: int) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S} ({n_messages} messages)"


def _candidates_text(candidates: list[EpisodicMemory]) -> str:
    blocks = []
    for i, ep in enumerate(candidates, 1):
        blocks.append(
            f"{i}. Time: {_time_range(ep.occurred_at, len(ep.raw))}\n"
            f"   Title: {ep.cue}\n"
            f"   Content: {ep.narrative[:CANDIDATE_PREVIEW_CHARS]}..."
        )
    return "\n\n".join(blocks)


def _statements_text(insights: list[SemanticInsight]) -> str:
    return "\n".join(f"- {s.statement}" for s in insights) or "None"


# --- Episodic Memory Integration ops (§3.2) ---


class EpisodicOps:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def partition(
        self, utterances: tuple[Utterance, ...]
    ) -> list[tuple[tuple[Utterance, ...], str]]:
        """Ppar — 윈도우를 (발화 묶음, topic) 그룹들로 분할 (§3.2.1).

        커버리지 복구까지 끝낸 결과만 내보낸다: 반환 그룹들의 합집합은
        항상 입력 전체다. 실패 시 통짜 한 그룹.
        """
        try:
            response = self._llm.chat_model(
                "",
                prompts.PARTITION.format(count=len(utterances), messages=_numbered(utterances)),
                PartitionResponse,
                temperature=0.2,
                max_tokens=PARTITION_MAX_TOKENS,
            )
        except Exception as e:
            return _fallback("partition", [(utterances, "conversation")], e)

        # index → 그룹 매핑 (범위 밖은 버림, 중복 배정은 첫 그룹이 이김)
        assignment: dict[int, int] = {}
        topics: list[str] = []
        for g, group in enumerate(response.episodes):
            topics.append(group.topic)
            for idx in group.indices:
                if 1 <= idx <= len(utterances) and idx not in assignment:
                    assignment[idx] = g

        if not assignment:
            return _fallback("partition", [(utterances, "conversation")], ValueError("empty partition"))

        # 커버리지 복구: 안 나온 index는 직전 index의 그룹으로
        prev_group = assignment[min(assignment)]
        for idx in range(1, len(utterances) + 1):
            if idx in assignment:
                prev_group = assignment[idx]
            else:
                assignment[idx] = prev_group

        groups: dict[int, list[Utterance]] = {}
        for idx in sorted(assignment):
            groups.setdefault(assignment[idx], []).append(utterances[idx - 1])
        return [(tuple(members), topics[g]) for g, members in sorted(groups.items())]

    def narrate(
        self, utterances: tuple[Utterance, ...], topic: str
    ) -> tuple[str, str, datetime | None]:
        """Pnar — (cue, narrative, t) 생성 (§3.2.2). t는 파싱 실패 시 None
        (세션 timestamp fallback은 method.py 소관 — schema.py 결정 참고)."""
        try:
            response = self._llm.chat_model(
                "",
                prompts.NARRATIVE.format(
                    conversation=_timestamped_dialogue(utterances), boundary_reason=topic
                ),
                NarrativeResponse,
                temperature=0.7,
                max_tokens=GENERATION_MAX_TOKENS,
            )
            return response.episodic_cue, response.narrative_episode, parse_iso(response.timestamp)
        except Exception as e:
            return _fallback(
                "narrate",
                (f"Conversation ({len(utterances)} messages)", _dialogue(utterances), None),
                e,
            )

    def select_target(
        self, new_narrative: str, new_occurred_at: datetime,
        new_len: int, candidates: list[EpisodicMemory],
    ) -> EpisodicMemory | None:
        """Psel — 병합 대상 선택 (§3.2.3). None = 병합 안 함 (idx = -1).

        새 episode는 narrative만 넘긴다 — 부록 D.1.3 프롬프트에 Title
        슬롯이 없다 (형식화와의 불일치는 prompt_templates docstring 참고).
        """
        try:
            response = self._llm.chat_model(
                "",
                prompts.SELECT_TARGET.format(
                    new_time_range=_time_range(new_occurred_at, new_len),
                    new_content=new_narrative,
                    candidates=_candidates_text(candidates),
                ),
                MergeDecision,
                temperature=0.7,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as e:
            return _fallback("select_target", None, e)
        if not 1 <= response.target_index <= len(candidates):
            return None  # -1(새 episode) 또는 범위 밖 지목 (trust boundary)
        return candidates[response.target_index - 1]

    def integrate(
        self, target: EpisodicMemory, new_cue: str, new_narrative: str,
        new_occurred_at: datetime, new_len: int,
    ) -> tuple[str, str, datetime] | None:
        """Pint — 병합 서사 (cν, Nν, t) 생성 (§3.2.3). None = 병합 포기."""
        try:
            response = self._llm.chat_model(
                "",
                prompts.INTEGRATE.format(
                    original_time_range=_time_range(target.occurred_at, len(target.raw)),
                    original_title=target.cue,
                    original_content=target.narrative,
                    new_time_range=_time_range(new_occurred_at, new_len),
                    new_title=new_cue,
                    new_content=new_narrative,
                    combined_events=f"Original: {target.narrative}\n\nNew: {new_narrative}",
                ),
                NarrativeResponse,
                temperature=0.7,
                max_tokens=GENERATION_MAX_TOKENS,
            )
        except Exception as e:
            return _fallback("integrate", None, e)
        merged_t = parse_iso(response.timestamp) or min(target.occurred_at, new_occurred_at)
        return response.episodic_cue, response.narrative_episode, merged_t


# --- Semantic Knowledge Distillation ops (§3.3) ---


class SemanticOps:
    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def distill(self, episode: EpisodicMemory, evoked: list[SemanticInsight]) -> list[str]:
        """Pant + Pdis 연쇄 — 예측하고 gap만 추출 (§3.3.1-3.3.2)."""
        try:
            prediction = self._llm.chat(
                "",
                prompts.ANTICIPATE.format(
                    episodic_cue=episode.cue, evoked_context=_statements_text(evoked)
                ),
                temperature=0.7,
                max_tokens=GENERATION_MAX_TOKENS,
            )
            response = self._llm.chat_model(
                "",
                prompts.DISTILL.format(
                    original_messages=_dialogue(episode.raw), predicted_episode=prediction
                ),
                InsightStatements,
                temperature=0.7,
                max_tokens=GENERATION_MAX_TOKENS,
            )
            return list(response.statements)
        except Exception as e:
            return _fallback("distill", [], e)

    def direct_distill(self, episode: EpisodicMemory) -> list[str]:
        """D.2 — cold start 직접 추출 (semantic DB가 빌 때, method.py 분기)."""
        episodes_text = f"Episode 1:\nTitle: {episode.cue}\nContent: {episode.narrative}"
        try:
            response = self._llm.chat_model(
                "",
                prompts.DIRECT_DISTILL.format(episodes=episodes_text),
                InsightStatements,
                temperature=0.7,
                max_tokens=GENERATION_MAX_TOKENS,
            )
            return list(response.statements)
        except Exception as e:
            return _fallback("direct_distill", [], e)


# --- Response Generation (§3.4) ---


def generate_answer(
    llm: LLMProvider,
    question: str,
    episodes: list[EpisodicMemory],
    insights: list[SemanticInsight],
    include_raw_top: int,
) -> str:
    # 컨텍스트 상한 없음 — 실측 전형 ~4.5k 토큰(ctx 16384의 1/3.6). 폭주
    # 서사(스키마 상한 근처) + 연쇄 병합 raw의 병리적 조합만 ~22k로 초과
    # 가능 (검증 리뷰 N15, 스모크 관찰 항목)
    episodic_lines: list[str] = []
    for i, ep in enumerate(episodes):
        episodic_lines.append(f"- [{ep.occurred_at.isoformat()}] {ep.narrative}")
        if i < include_raw_top:  # 상위 r개는 원문 첨부 (§4.1, r=2)
            episodic_lines.append("    Source Messages:")
            episodic_lines.extend(
                f"    - {u.speaker}: {utterance_text(u)}" for u in ep.raw
            )
    return llm.chat(
        "",
        prompts.ANSWER.format(
            episodic="\n".join(episodic_lines) or "None",
            semantic=_statements_text(insights),
            question=question,
        ),
        temperature=0.0,
        max_tokens=GENERATION_MAX_TOKENS,
    )
