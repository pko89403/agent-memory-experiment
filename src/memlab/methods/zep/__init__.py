"""Zep (arXiv 2501.13956) 재구현 — temporal knowledge graph 메모리.

논문이 명세서다. 원본 graphiti(external/graphiti, v0.5.2 핀)는 논문이
침묵하는 상수·프롬프트의 차용처로만 쓴다. 모든 해석·결정은 각 모듈
docstring에 기록한다.
"""
from memlab.methods.zep.method import ZepConfig, ZepMethod

__all__ = ["ZepConfig", "ZepMethod"]
