"""배치 채점기 — (QA, 시스템 답) 쌍 목록을 받아 리포트를 만든다.

메모리 메소드가 답을 뽑아내면, 그 결과를 이 함수에 그대로
물린다. 정답 선택 규칙은 원본(main_loco_parse.py:251-255)과 동일:
answer가 있으면 answer, 없으면 adversarial_answer.

cat5(adversarial)는 함정 오답을 기준으로 채점되므로 점수가 높을수록
나쁘다(환각). 그래서 overall 평균은 cat1~4만으로 내고, cat5는 별도 줄이다.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean

from memlab.data import QA, Category
from memlab.evaluation.metrics import bleu1, set_f1, standard_f1


@dataclass(frozen=True)
class MetricRow:
    n: int
    set_f1: float
    standard_f1: float
    bleu1: float


@dataclass(frozen=True)
class Report:
    per_category: dict[Category, MetricRow]
    overall: MetricRow  # cat1~4만. cat5는 방향이 반대라 섞지 않는다.


def reference_for(qa: QA) -> str:
    """채점 기준 답. 원본과 동일: answer 없으면 adversarial_answer."""
    return qa.answer or qa.adversarial_answer or ""


def score(pairs: Iterable[tuple[QA, str]]) -> Report:
    rows: dict[Category, list[tuple[float, float, float]]] = {}
    for qa, prediction in pairs:
        ref = reference_for(qa)
        rows.setdefault(qa.category, []).append(
            (set_f1(prediction, ref), standard_f1(prediction, ref), bleu1(prediction, ref))
        )

    def to_row(scores: list[tuple[float, float, float]]) -> MetricRow:
        return MetricRow(
            n=len(scores),
            set_f1=mean(s[0] for s in scores),
            standard_f1=mean(s[1] for s in scores),
            bleu1=mean(s[2] for s in scores),
        )

    per_category = {cat: to_row(s) for cat, s in sorted(rows.items())}
    non_adversarial = [
        s for cat, scores in rows.items() if cat != Category.ADVERSARIAL for s in scores
    ]
    if not non_adversarial:
        raise ValueError("cat1~4 문제가 하나도 없어 overall을 낼 수 없다")
    return Report(per_category=per_category, overall=to_row(non_adversarial))


def format_report(report: Report) -> str:
    """리포트를 표 문자열로."""
    lines = ["category      |    n | set_f1 | std_f1 | bleu1"]
    lines.append("--------------+------+--------+--------+------")
    for cat, row in report.per_category.items():
        note = "  (높을수록 나쁨)" if cat == Category.ADVERSARIAL else ""
        lines.append(
            f"{cat.name:<13} | {row.n:4d} | {row.set_f1:.4f} | "
            f"{row.standard_f1:.4f} | {row.bleu1:.4f}{note}"
        )
    o = report.overall
    lines.append("--------------+------+--------+--------+------")
    lines.append(
        f"{'OVERALL(1~4)':<13} | {o.n:4d} | {o.set_f1:.4f} | "
        f"{o.standard_f1:.4f} | {o.bleu1:.4f}"
    )
    return "\n".join(lines)
