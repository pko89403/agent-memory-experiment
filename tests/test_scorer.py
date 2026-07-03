"""배치 채점기 테스트."""
import pytest

from memlab.data import QA, Category, load_locomo
from memlab.evaluation.scorer import format_report, reference_for, score


def qa(category, answer=None, adversarial=None, question="q"):
    return QA(
        question=question,
        answer=answer,
        evidence=(),
        category=Category(category),
        adversarial_answer=adversarial,
    )


def test_reference_selection_matches_original():
    # 원본 규칙: answer가 있으면 answer, 비면 adversarial_answer
    assert reference_for(qa(4, answer="mental health")) == "mental health"
    assert reference_for(qa(5, adversarial="self-care")) == "self-care"
    assert reference_for(qa(5, answer="real", adversarial="trap")) == "real"  # cat5 예외 2건


def test_score_perfect_and_wrong():
    pairs = [
        (qa(4, answer="mental health"), "mental health"),  # 만점
        (qa(4, answer="mental health"), "pizza"),          # 0점
        (qa(1, answer="adoption agencies"), "adoption agencies"),
        (qa(5, adversarial="self-care"), "I don't know"),  # 함정에 안 속음 → 0 (좋음)
    ]
    r = score(pairs)
    assert r.per_category[Category.SINGLE_HOP].n == 2
    assert r.per_category[Category.SINGLE_HOP].set_f1 == pytest.approx(0.5)
    assert r.per_category[Category.MULTI_HOP].set_f1 == 1.0
    assert r.per_category[Category.ADVERSARIAL].set_f1 == 0.0
    # overall은 cat5를 제외한 3문항 평균
    assert r.overall.n == 3
    assert r.overall.set_f1 == pytest.approx((1.0 + 0.0 + 1.0) / 3)


def test_overall_requires_non_adversarial():
    with pytest.raises(ValueError):
        score([(qa(5, adversarial="trap"), "answer")])


def test_full_dataset_perfect_student():
    # 정답을 그대로 답하는 가상의 만점 학생: cat1~4는 전부 1.0이어야 한다
    samples = load_locomo()
    pairs = [(q, q.answer or "I don't know") for s in samples for q in s.qa]
    r = score(pairs)
    assert r.overall.n == 1540  # 1986 - 446(cat5)
    assert r.overall.set_f1 == 1.0
    assert r.overall.bleu1 == 1.0
    # cat5는 함정 답 기준이라 만점 학생도 0에 가깝다 (겹치는 단어가 거의 없음)
    assert r.per_category[Category.ADVERSARIAL].set_f1 < 0.1
    assert "OVERALL" in format_report(r)
