"""채점 함수 테스트.

핵심은 차분 테스트: 우리 set_f1을 원본 repo의 calculate_f1과
같은 입력으로 돌려 같은 출력인지 확인한다. API 호출 없이 무료다.
"""
import importlib.util
import math
import sys

import pytest

from memlab.config import MEMORYOS_DIR
from memlab.data import load_locomo
from memlab.evaluation.metrics import bleu1, set_f1, standard_f1, tokenize


# ── 손계산 값 ──────────────────────────────────────────────

def test_tokenize():
    assert tokenize("It raised awareness, for Mental Health!") == [
        "it", "raised", "awareness", "for", "mental", "health",
    ]
    assert tokenize(None) == []
    assert tokenize("") == []


def test_set_f1_hand_computed():
    # precision 2/7, recall 2/2 → F1 = 4/9
    pred = "It raised awareness for mental health issues."
    assert set_f1(pred, "mental health") == pytest.approx(4 / 9)
    assert set_f1("mental health", "mental health") == 1.0
    assert set_f1("", "mental health") == 0.0
    assert set_f1("anything", "") == 0.0  # 정답이 비면 항상 0 (cat5 함정)


def test_set_vs_standard_f1():
    # 반복 단어: set은 중복을 무시해 점수가 부풀고, standard는 빈도를 벌한다
    pred, ref = "health health health", "mental health"
    assert set_f1(pred, ref) == pytest.approx(2 / 3)       # p=1/1, r=1/2
    assert standard_f1(pred, ref) == pytest.approx(0.4)     # p=1/3, r=1/2
    # 반복이 없으면 두 방식은 같다
    pred2 = "it raised awareness for mental health issues"
    assert set_f1(pred2, ref) == pytest.approx(standard_f1(pred2, ref))


def test_bleu1_hand_computed():
    assert bleu1("mental health", "mental health") == 1.0
    # 한 단어만 답하면: precision 1.0이지만 벌점 exp(1-2/1)=e^-1
    assert bleu1("mental", "mental health") == pytest.approx(math.exp(-1))
    # 정답보다 길면 벌점 없음: precision 2/7
    assert bleu1("it raised awareness for mental health issues", "mental health") \
        == pytest.approx(2 / 7)
    assert bleu1("", "mental health") == 0.0
    assert bleu1("anything", "") == 0.0


# ── 차분 테스트: 원본 채점기와 완전 일치 ──────────────────────

@pytest.fixture(scope="module")
def original():
    """원본 repo의 evalution_loco.py를 모듈로 로드한다."""
    path = MEMORYOS_DIR / "eval" / "evalution_loco.py"
    spec = importlib.util.spec_from_file_location("evalution_loco", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evalution_loco"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_set_f1_matches_original(original):
    samples = load_locomo()
    answers = [q.answer for s in samples for q in s.qa if q.answer]
    adversarial = [q.adversarial_answer for s in samples for q in s.qa
                   if q.adversarial_answer]

    cases = []
    # 정답 vs 자기 자신 (F1=1이어야 하는 경우들)
    cases += [(a, a) for a in answers[:200]]
    # 정답 vs 다른 정답 (부분 겹침)
    cases += list(zip(answers[:500], answers[1:501]))
    # 함정 답 vs 정답
    cases += list(zip(adversarial[:200], answers[:200]))
    # 경계값
    cases += [("", ""), ("a", ""), ("", "a"), ("Hello, world!", "hello world"),
              ("don't", "dont"), ("2023", "in 2023")]

    for pred, ref in cases:
        assert set_f1(pred, ref) == pytest.approx(
            original.calculate_f1(pred, ref)
        ), f"불일치: pred={pred!r} ref={ref!r}"
