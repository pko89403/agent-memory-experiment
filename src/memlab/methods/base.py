"""모든 메모리 메소드가 구현하는 공통 인터페이스(소켓).

실험 반복문은 이 두 메서드만 안다:

    student = SomeMethod()
    for utt in utterances:             # ① 수업: 발화를 순서대로
        student.ingest(utt)
    prediction = student.answer(q)     # ② 시험 → ③ 채점은 evaluation.score()

입력 단위는 **발화(Utterance)** — 데이터가 주는 그대로의 중립 단위다.
발화를 어떻게 묶어 기억할지는 각 메소드의 몫이다. 예: MemoryOS는 내부에서
(user, assistant) 쌍으로 접어 자기 스키마인 page {Q, R, T}를 만들고,
거기에 meta_chain 같은 고유 필드를 덧붙인다. 그런 스키마는 전부
methods/memoryos/ 안에만 존재한다 — 이 소켓에는 어떤 메소드의 고유
개념도 넣지 않는다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    """대화 발화 하나 — 벤치마크가 주는 그대로."""

    speaker: str
    text: str
    timestamp: str  # 이 발화가 속한 세션의 date_time
    blip_caption: str | None = None  # 이미지 공유 발화면 캡션


class MemoryMethod(ABC):
    @abstractmethod
    def ingest(self, utterance: Utterance) -> None:
        """발화 하나를 기억에 넣는다."""

    @abstractmethod
    def answer(self, question: str) -> str:
        """기억만으로 질문에 답한다."""
