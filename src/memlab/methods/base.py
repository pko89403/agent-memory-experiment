"""모든 메모리 메소드가 구현하는 공통 인터페이스(소켓).

실험 반복문은 이 두 메서드만 안다:

    student = SomeMethod()
    for page in pages:                 # ① 수업
        student.ingest(page)
    prediction = student.answer(q)     # ② 시험 → ③ 채점은 evaluation.score()

MemoryOS 재구현도, 이후의 cause-aware forgetting 변형도 이 소켓에 꽂힌다.
같은 반복문 + 같은 채점기 = 공정한 비교.

MCP 서버(로드맵 07)도 이 두 메서드를 @mcp.tool로 노출하는 얇은 포장이 된다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """대화 한 쌍 — 학생이 한 번에 먹는 단위. 원본의 dialogue page와 동일:
    {user_input, agent_response, timestamp}."""

    user_input: str
    agent_response: str
    timestamp: str


class MemoryMethod(ABC):
    @abstractmethod
    def ingest(self, page: Page) -> None:
        """대화 한 쌍을 기억에 넣는다."""

    @abstractmethod
    def answer(self, question: str) -> str:
        """기억만으로 질문에 답한다."""
