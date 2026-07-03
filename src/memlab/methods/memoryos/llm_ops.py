"""MemoryOS의 LLM 연산 — 템플릿(prompt_templates)을 채워 프로바이더를 부른다.

소비자별로 쪼갠 작은 ops 클래스 4개. 각 tier가 선언한 Protocol과 1:1로
마주 본다 (저장 역학과 인지의 분리 — 판단만 교체하는 실험이 가능해진다):

    ChainLlmOps      → short_term.ChainOps      (chain 판단·재요약)
    SegmentLlmOps    → mid_term.SegmentOps      (키워드·세그먼트 요약)
    PersonaLlmOps    → LPM 승격 분석             (method가 소비)
    AnswerGenerator  → 답변 생성 (논문 3.4)       (method가 소비)

구조화 출력이 필요한 연산(판정·추출)은 Pydantic 응답 모델로 스키마를
선언한다 — 검증·파싱은 LLMProvider.chat_model이 담당.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from memlab.llm import LLMProvider
from memlab.methods.memoryos import prompt_templates as prompts
from memlab.methods.memoryos.schema import Page, page_text


# ── 응답 모델 (구조화 출력 스키마) ──────────────────────────────────


class Continuity(BaseModel):
    continuous: bool


class Keywords(BaseModel):
    keywords: list[str] = Field(max_length=3)


class ExtractedKnowledge(BaseModel):
    user_facts: list[str]
    assistant_knowledge: list[str]


def _dialogue(pages: list[Page]) -> str:
    return "\n".join(page_text(p) for p in pages)


def _stripped(items: list[str]) -> list[str]:
    return [item.strip() for item in items if item.strip()]


# ── STM: dialogue chain (논문 식 1) ────────────────────────────────


class ChainLlmOps:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def judge_continuity(self, chain: list[Page], page: Page) -> bool:
        """chain의 마지막 page와 새 page가 이어지는 대화인가."""
        prev = chain[-1]
        out = self.llm.chat_model(
            prompts.CONTINUITY_SYSTEM,
            prompts.CONTINUITY_USER.format(
                prev_user=prev.user_input,
                prev_agent=prev.agent_response,
                curr_user=page.user_input,
                curr_agent=page.agent_response,
            ),
            Continuity,
            max_tokens=20,
        )
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
        retrieval = "\n".join(
            f"【Historical Memory】 {speaker_a}: {p.user_input}\n"
            f"{speaker_b}: {p.agent_response}\nTime:({p.timestamp})\n"
            f"Conversation chain overview:({p.meta_info})\n"
            for p in retrieved_pages
        )
        background = f"【User Profile】\n{lpm_info['user_profile'] or 'None'}\n\n"
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
