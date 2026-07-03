"""STM 재구현 vs 원본 차분 테스트.

원본 short_term_memory.py는 `from utils import get_timestamp`를 하는데,
원본 utils는 sentence-transformers 등 무거운 의존성을 끌고 온다.
STM에 필요한 건 get_timestamp뿐이므로 가짜 utils를 주입해서 로드한다.
"""
import importlib.util
import sys
import types

import pytest

from memlab.config import MEMORYOS_DIR
from memlab.methods.memoryos import ShortTermMemory
from memlab.methods.memoryos.utils import get_timestamp


@pytest.fixture()
def original_cls(tmp_path, monkeypatch):
    fake_utils = types.ModuleType("utils")
    fake_utils.get_timestamp = get_timestamp
    monkeypatch.setitem(sys.modules, "utils", fake_utils)

    path = MEMORYOS_DIR / "eval" / "short_term_memory.py"
    spec = importlib.util.spec_from_file_location("orig_stm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    counter = iter(range(1000))

    def make(max_capacity, file_name=None):
        # 기본은 인스턴스마다 새 파일 = "깨끗한 실행" 조건.
        # (같은 파일을 재사용하면 원본은 이전 상태를 조용히 load한다 —
        #  아래 test_original_loads_leftover_state가 그걸 시연한다.)
        name = file_name or f"stm_{next(counter)}.json"
        return mod.ShortTermMemory(
            max_capacity=max_capacity, file_path=str(tmp_path / name)
        )

    return make


def page(i):
    return {"user_input": f"q{i}", "agent_response": f"r{i}", "timestamp": f"t{i}"}


def test_differential_same_ops_same_state(original_cls):
    """add/evict를 섞은 동일 시나리오에서 원본과 상태가 늘 같아야 한다."""
    for capacity in (1, 2, 10):
        ours, orig = ShortTermMemory(capacity), original_cls(capacity)
        for i in range(25):
            ours.add_qa_pair(page(i))
            orig.add_qa_pair(page(i))
            assert ours.is_full() == orig.is_full()
            if ours.is_full():
                assert ours.pop_oldest() == orig.pop_oldest()
            assert ours.get_all() == orig.get_all()


def test_fifo_order():
    stm = ShortTermMemory(max_capacity=3)
    for i in range(3):
        stm.add_qa_pair(page(i))
    assert stm.pop_oldest()["user_input"] == "q0"
    assert stm.pop_oldest()["user_input"] == "q1"
    assert [p["user_input"] for p in stm.get_all()] == ["q2"]


def test_maxlen_silent_drop():
    # 원본 deque(maxlen) 의미론: 가득 찬 채 add하면 가장 오래된 것이 밀려난다
    stm = ShortTermMemory(max_capacity=2)
    for i in range(3):
        stm.add_qa_pair(page(i))
    assert [p["user_input"] for p in stm.get_all()] == ["q1", "q2"]


def test_timestamp_default_fill():
    stm = ShortTermMemory()
    p = {"user_input": "q", "agent_response": "r"}
    stm.add_qa_pair(p)
    assert "timestamp" in p  # 원본과 동일: 입력 dict를 직접 변형
    assert len(p["timestamp"]) == 19  # 'YYYY-MM-DD HH:MM:SS'


def test_pop_empty_returns_none():
    assert ShortTermMemory().pop_oldest() is None


def test_original_loads_leftover_state(original_cls):
    """원본의 오염 버그 시연 — 우리가 파일 persistence를 뺀 이유.

    원본은 init 때 file_path의 기존 파일을 조용히 load한다. 이전 실행이
    남긴 파일이 있으면 그 기억을 갖고 시작한다. 우리 구현은 항상 빈 상태다.
    """
    first = original_cls(2, file_name="reused.json")
    first.add_qa_pair(page(0))
    second = original_cls(2, file_name="reused.json")  # 같은 경로 재사용
    assert second.get_all() == [page(0)]  # 이전 인스턴스의 기억이 살아 있음!
    assert ShortTermMemory(2).get_all() == []  # 우리는 항상 깨끗하게 시작
