"""MemoryOS의 LLM 연산 — 템플릿(prompt_templates)을 채워 프로바이더를 부른다.

소비자별로 쪼갠 작은 ops 클래스 4개. 각 tier가 선언한 Protocol과 1:1로
마주 본다 (저장 역학과 인지의 분리 — 판단만 교체하는 실험이 가능해진다):

    ChainLlmOps      → short_term.ChainOps      (chain 판단·재요약)
    SegmentLlmOps    → mid_term.SegmentOps      (키워드·세그먼트 요약)
    PersonaLlmOps    → LPM 승격 분석             (method가 소비)
    AnswerGenerator  → 답변 생성 (논문 3.4)       (method가 소비)

구조화 출력이 필요한 연산(판정·추출)은 Pydantic 응답 모델로 스키마를
선언한다 — 검증·파싱은 LLMProvider.chat_model이 담당.
로컬 소형 모델 하드닝: 모든 응답 모델에 extra="forbid"(스키마 밖 필드를
문법 수준에서 차단), max_tokens는 여유 있게 — 빠듯하면 JSON이 잘린다
(75발화 스모크 실패로 실측).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from memlab.llm import LLMProvider
from memlab.methods.memoryos import prompt_templates as prompts
from memlab.methods.memoryos.schema import Page, page_text


# ── 응답 모델 (구조화 출력 스키마) ──────────────────────────────────


class Continuity(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 스키마 밖 필드를 문법 수준에서 금지

    continuous: bool


class Keywords(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(max_length=3)


class ExtractedKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_facts: list[str]
    assistant_knowledge: list[str]


# 프로필은 승격마다 자라는 유일한 무한정 항. 생성 프롬프트가 600단어를
# 지시하지만(prompt_templates.PERSONALITY_USER 8번 규칙) 모델이 어길 수
# 있으므로, 답변 프롬프트 조립 시 하드 상한으로 이중 방어한다.
PROFILE_CHAR_LIMIT = 4000


def _dialogue(pages: list[Page]) -> str:
    return "\n".join(page_text(p) for p in pages)


def _stripped(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


# ── STM: dialogue chain (논문 식 1) ────────────────────────────────


class ChainLlmOps:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def judge_continuity(self, chain: list[Page], page: Page) -> bool:
        """chain의 마지막 page와 새 page가 이어지는 대화인가.

        판정 호출이 끝내 실패하면 False(체인 리셋)로 강등한다 — 원본 eval도
        비정상 응답을 전부 False로 처리했다. 최악의 결과가 "체인이 하나 더
        쪼개짐"이라 안전하며, 대화 전체를 죽이는 것보다 낫다.
        """
        prev = chain[-1]
        try:
            out = self.llm.chat_model(
                prompts.CONTINUITY_SYSTEM,
                prompts.CONTINUITY_USER.format(
                    prev_user=prev.user_input,
                    prev_agent=prev.agent_response,
                    curr_user=page.user_input,
                    curr_agent=page.agent_response,
                ),
                Continuity,
                max_tokens=200,
            )
        except Exception as error:
            print(f"    [judge 강등] {error!r} → chain 리셋")
            return False
        return out.continuous

    def summarize_chain(self, chain: list[Page]) -> str:
        """chain 전체를 meta_chain으로 재요약."""
        return self.llm.chat(
            prompts.CHAIN_SUMMARY_SYSTEM,
            prompts.CHAIN_SUMMARY_USER.format(dialogue=_dialogue(chain)),
            temperature=0.3,
            max_tokens=100,
        )


# ── MTM: 세그먼트 유지 (논문 식 2, 3) ──────────────────────────────


class SegmentLlmOps:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def extract_keywords(self, text: str) -> set[str]:
        """식 (3)의 키워드 집합 K — 최대 3개."""
        out = self.llm.chat_model(
            prompts.KEYWORDS_SYSTEM,
            prompts.KEYWORDS_USER.format(text=text),
            Keywords,
            temperature=0.7,
        )
        return {w.strip().lower() for w in out.keywords if w.strip()}

    def summarize_segment(self, pages: list[Page]) -> str:
        """논문: 세그먼트 내용은 member page들의 LLM 요약."""
        return self.llm.chat(
            prompts.SEGMENT_SUMMARY_SYSTEM,
            prompts.SEGMENT_SUMMARY_USER.format(dialogue=_dialogue(pages)),
            temperature=0.7,
        )


# ── LPM: 승격된 세그먼트 분석 ──────────────────────────────────────


class PersonaLlmOps:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def analyze_personality(self, existing_profile: str, pages: list[Page]) -> str:
        """90차원 근거로 갱신된 프로필 텍스트."""
        return self.llm.chat(
            prompts.PERSONALITY_SYSTEM,
            prompts.PERSONALITY_USER.format(
                dimensions=prompts.PERSONALITY_DIMENSIONS,
                existing_profile=existing_profile or "None",
                conversation=_dialogue(pages),
            ),
            temperature=0.7,
        )

    def extract_knowledge(self, pages: list[Page]) -> tuple[list[str], list[str]]:
        """(사용자 사실들, 어시스턴트 지식들)."""
        out = self.llm.chat_model(
            prompts.KNOWLEDGE_SYSTEM,
            prompts.KNOWLEDGE_USER.format(dialogue=_dialogue(pages)),
            ExtractedKnowledge,
            temperature=0.7,
        )
        return _stripped(out.user_facts), _stripped(out.assistant_knowledge)


# ── 답변 생성 (논문 3.4: 세 tier 통합) ─────────────────────────────


class AnswerGenerator:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def generate(
        self,
        question: str,
        speaker_a: str,
        speaker_b: str,
        stm_pages: list[Page],
        retrieved_pages: list[Page],
        lpm_info: dict,
    ) -> str:
        history = "\n".join(
            f"{speaker_a}: {p.user_input}\n{speaker_b}: {p.agent_response}\n"
            f"Time: ({p.timestamp})"
            for p in stm_pages
        )
        # 같은 chain의 페이지들은 meta_info가 동일하다 — 반복 출력은 토큰만
        # 낭비하므로 chain overview는 처음 등장할 때 한 번만 붙인다.
        seen_overviews: set[str] = set()
        retrieval_parts = []
        for p in retrieved_pages:
            part = (
                f"【Historical Memory】 {speaker_a}: {p.user_input}\n"
                f"{speaker_b}: {p.agent_response}\nTime:({p.timestamp})\n"
            )
            if p.meta_info and p.meta_info not in seen_overviews:
                seen_overviews.add(p.meta_info)
                part += f"Conversation chain overview:({p.meta_info})\n"
            retrieval_parts.append(part)
        retrieval = "\n".join(retrieval_parts)
        profile = (lpm_info["user_profile"] or "None")[:PROFILE_CHAR_LIMIT]
        background = f"【User Profile】\n{profile}\n\n"
        background += "".join(f"{fact}\n" for fact in lpm_info["user_kb"])
        assistant_knowledge = "【Assistant Knowledge】\n"
        assistant_knowledge += "".join(f"- {t}\n" for t in lpm_info["agent_traits"])

        return self.llm.chat(
            prompts.ANSWER_SYSTEM.format(
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                assistant_knowledge=assistant_knowledge,
            ),
            prompts.ANSWER_USER.format(
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                history=history,
                retrieval=retrieval,
                background=background,
                question=question,
            ),
            temperature=0.7,
        )
