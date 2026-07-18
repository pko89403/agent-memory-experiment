"""Nemori (arXiv 2508.03341) 재구현 — adaptive memory distillation 메모리.

논문(v4)이 명세서다. 원본 repo(external/nemori, HEAD d2a6dff 핀)는 논문이
침묵하는 배선·상수의 차용처로만 쓴다. 모든 해석·결정은 각 모듈
docstring에 기록한다.
"""
from memlab.methods.nemori.method import NemoriConfig, NemoriMethod

__all__ = ["NemoriConfig", "NemoriMethod"]
