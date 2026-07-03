"""STM — 논문 3.1: dialogue page 큐(7칸) + dialogue chain.

논문 명세:
- page = {Q, R, T}. 큐는 고정 7칸, 차면 가장 오래된 page가 MTM으로 FIFO 이관
  (이관은 updater의 몫 — STM은 큐와 chain만 안다).
- page_chain_i = {Q_i, R_i, T_i, meta_chain_i}                          (1)
  새 page가 들어올 때 LLM이 2단계로 meta를 생성:
  ① 기존 chain(prior pages)과 이어지는지 판단 → 잇거나, 끊기면 현재
     page로 리셋   ② chain의 모든 page를 meta_chain으로 재요약.
  재요약된 meta는 chain의 모든 page에 소급 반영된다 — "한 page만 꺼내도
  맥락이 통째로 딸려오게".

chain은 page 객체 참조 목록이므로, page가 MTM으로 이관된 뒤에도 chain이
이어지면 소급 갱신이 그대로 닿는다 (같은 Page 객체를 공유).

LLM 의존은 ChainOps 인터페이스로 주입받는다 — STM이 필요한 만큼만
선언한 소비자 소유 인터페이스. MemoryOSLlmOps가 이를 구조적으로 충족한다.
"""
from __future__ import annotations

from collections import deque
from typing import Protocol

from memlab.methods.memoryos.schema import Page
from memlab.methods.memoryos.utils import generate_id, get_timestamp


class ChainOps(Protocol):
    """dialogue chain 유지에 필요한 LLM 연산 (논문 식 1의 2단계)."""

    def judge_continuity(self, chain: list[Page], page: Page) -> bool:
        """① 새 page가 기존 chain과 이어지는가."""
        ...

    def summarize_chain(self, chain: list[Page]) -> str:
        """② chain 전체의 meta_chain 재요약."""
        ...


class ShortTermMemory:
    def __init__(
        self,
        chain_ops: ChainOps,
        max_capacity: int = 7,  # 논문: "fixed length ... is 7"
    ):
        self.chain_ops = chain_ops
        self.max_capacity = max_capacity
        self.memory: deque[Page] = deque(maxlen=max_capacity)
        self.chain: list[Page] = []  # 현재 이어지고 있는 chain (page 참조)

    # ── 큐 ────────────────────────────────────────────────────────

    def add_page(self, page: Page) -> None:
        """page를 chain에 연결(또는 리셋)하고 큐에 넣는다.

        주의: 큐가 가득 찬 채 호출하면 deque(maxlen)이 가장 오래된 page를
        조용히 버린다. updater는 add 후 is_full()이면 pop_oldest()로
        이관할 것 (원본 eval의 add→evict 순서와 동일).
        """
        if page.timestamp is None:
            page.timestamp = get_timestamp()
        if page.page_id is None:
            page.page_id = generate_id("page")
        self._link_to_chain(page)
        self.memory.append(page)

    def get_all(self) -> list[Page]:
        return list(self.memory)

    def is_full(self) -> bool:
        return len(self.memory) == self.max_capacity

    def pop_oldest(self) -> Page | None:
        if self.memory:
            return self.memory.popleft()
        return None

    # ── dialogue chain (식 1) ────────────────────────────────────

    def _link_to_chain(self, page: Page) -> None:
        if self.chain and self.chain_ops.judge_continuity(self.chain, page):  # ①
            prev = self.chain[-1]
            page.pre_page = prev.page_id
            prev.next_page = page.page_id
            self.chain.append(page)
            meta = self.chain_ops.summarize_chain(self.chain)  # ② chain 전체 재요약
            for chained in self.chain:  # 소급 반영
                chained.meta_info = meta
        else:  # 첫 page이거나 끊김 → 현재 page로 리셋
            self.chain = [page]
            page.meta_info = self.chain_ops.summarize_chain([page])
