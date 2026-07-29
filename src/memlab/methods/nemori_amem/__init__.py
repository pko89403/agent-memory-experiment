"""nemori×amem 합성 — Nemori distillation + A-MEM note management.

연구 질문 (2026-07-25 합의): "distill된 fact에 memory management(링크·
evolution)를 얹으면 naive append보다 나은가?" 대조군은 Nemori 전량
baseline(같은 하이퍼파라미터, semantic store만 다름).

Nemori 논문 §3.1·부록 A.3이 명세하는 third-party integration의 통제판:
- 논문 A.3은 host가 답변까지 전담(episodic 폐기)하지만, 그러면 temporal
  붕괴(Table 14)와 management 효과가 뒤섞인다 — 우리는 semantic store만
  교체하고 답변 경로는 Nemori 그대로 (단일 변인).
- 답변 컨텍스트는 statement 20개 형식·예산 고정 — A-MEM의 K/G/X는
  임베딩·링크·evolution으로만 기여 (결정 (i)).

교차 import 예외: 합성 메소드는 본질적으로 두 부품(methods/nemori,
methods/amem)의 소비자다 — 메소드 간 격리 규칙의 취지(공용 계층 오염
방지)에 어긋나지 않는 방향성 의존.
"""
from memlab.methods.nemori_amem.method import NemoriAmemConfig, build_nemori_amem

__all__ = ["NemoriAmemConfig", "build_nemori_amem"]
