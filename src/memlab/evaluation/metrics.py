"""채점 함수 — 시스템의 답과 정답이 "얼마나 겹치는지"를 숫자로.

set_f1은 원본 repo(eval/evalution_loco.py)의 계산을 그대로 재현한다.
재현이 정확하다는 것은 tests/test_metrics.py의 차분 테스트가 보증한다
(원본 모듈을 import해서 같은 입력에 같은 출력인지 직접 비교).

주의: cat5(adversarial)는 원본이 정답 칸에 함정 오답(adversarial_answer)을
넣으므로, F1이 높을수록 나쁘다(환각). cat1~4와 평균을 섞지 말 것.
"""
from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str | None) -> list[str]:
    """소문자로 바꾸고, 문장부호를 떼고, 단어로 쪼갠다. 원본과 동일."""
    if not text:
        return []
    return re.findall(r"\b\w+\b", str(text).lower())


def set_f1(prediction: str | None, reference: str | None) -> float:
    """원본 방식 F1 — 토큰을 set에 넣으므로 같은 단어의 반복은 무시된다."""
    pred = set(tokenize(prediction))
    ref = set(tokenize(reference))
    common = pred & ref
    p = len(common) / len(pred) if pred else 0.0
    r = len(common) / len(ref) if ref else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def standard_f1(prediction: str | None, reference: str | None) -> float:
    """표준(SQuAD 방식) F1 — 토큰 빈도까지 반영한다."""
    pred = Counter(tokenize(prediction))
    ref = Counter(tokenize(reference))
    common = sum((pred & ref).values())
    if common == 0:
        return 0.0
    p = common / sum(pred.values())
    r = common / sum(ref.values())
    return 2 * p * r / (p + r)


def bleu1(prediction: str | None, reference: str | None) -> float:
    """BLEU-1 — 예측 단어 중 정답에 있는 비율(클리핑) × 짧은 답 벌점.

    F1과 달리 recall이 없어서, 벌점(brevity penalty)이 없으면 한 단어만
    맞혀도 만점이 된다. 그래서 정답보다 짧은 답은 exp(1 - 정답길이/답길이)로
    깎는다.
    """
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    ref_counts = Counter(ref)
    clipped = sum(min(n, ref_counts[t]) for t, n in Counter(pred).items())
    precision = clipped / len(pred)
    if precision == 0.0:
        return 0.0
    bp = 1.0 if len(pred) >= len(ref) else math.exp(1 - len(ref) / len(pred))
    return bp * precision
