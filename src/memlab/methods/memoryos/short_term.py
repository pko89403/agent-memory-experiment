"""STM 재구현 — 원본 eval/short_term_memory.py와 동작 동일.

원본 대비 의도적 차이 (결과에 영향 없음, 차분 테스트로 확인):
- 파일 persistence(save/load) 제거. 원본은 add마다 JSON을 쓰고 init 때
  읽는데, 이전 실행이 남긴 파일을 조용히 load해서 실행 간 오염 위험이 있다.
  깨끗한 디렉토리에서 시작하는 한 결과는 동일하므로 메모리로만 간다.
- 디버그 print 제거.

내부 저장 단위는 dict다 (Page가 아니라). 원본 파이프라인이 페이지에
meta_info/page_id/체인 링크를 계속 덧붙이며 변형하기 때문 — 인터페이스
경계(MemoryMethod.ingest)에서만 Page를 dict로 바꿔 넣는다.

주의: add_qa_pair는 원본과 똑같이 입력 dict를 변형한다(timestamp 채움).
"""
from __future__ import annotations

from collections import deque

from memlab.methods.memoryos.utils import get_timestamp


class ShortTermMemory:
    def __init__(self, max_capacity: int = 1):
        self.max_capacity = max_capacity
        # deque(maxlen=...)의 의미까지 원본과 동일: 가득 찬 상태에서 append하면
        # 가장 오래된 것이 조용히 밀려난다. (eval 루프는 add 직후 항상 evict해서
        # 실전에서는 발생하지 않지만, 의미론은 보존한다.)
        self.memory: deque[dict] = deque(maxlen=max_capacity)

    def add_qa_pair(self, qa_pair: dict) -> None:
        qa_pair.setdefault("timestamp", get_timestamp())
        self.memory.append(qa_pair)

    def get_all(self) -> list[dict]:
        return list(self.memory)

    def is_full(self) -> bool:
        return len(self.memory) == self.max_capacity

    def pop_oldest(self) -> dict | None:
        if self.memory:
            return self.memory.popleft()
        return None
