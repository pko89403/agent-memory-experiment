"""모든 메모리 메소드가 구현하는 공통 인터페이스(소켓).

실험 반복문은 이 두 메서드만 안다:

    method = SomeMethod()
    for utt in utterances:             # ① ingest: 발화를 순서대로
        method.ingest(utt)
    prediction = method.answer(q)      # ② QA → ③ 채점은 evaluation.score()

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

    def end_ingest(self) -> None:
        """ingest 종료 신호 — 버퍼형 메소드가 잔여분을 처리한다 (기본 no-op).

        (검증 리뷰 N1) 버퍼를 가진 메소드(nemori)가 잔여 flush를 answer()
        안에서 하면 flush 실패가 러너의 QA 단위 격리에 삼켜진다 — buffer가
        안 비워져 QA마다 재실행되고, all-error checkpoint가 남아 resume이
        대화를 영구 스킵한다. 러너가 ingest 루프 직후 이 훅을 불러 실패를
        대화 단위 에러 도메인(재개 시 재시도)에 둔다. zep 때 "finalize 훅
        없이"로 결정했으나(2026-07-10) 그건 버퍼 없는 메소드 시절 — 세 번째
        메소드에서 설계 신호가 실현됐다 (2026-07-17 합의).
        """

    @abstractmethod
    def answer(self, question: str) -> str:
        """기억만으로 질문에 답한다."""
