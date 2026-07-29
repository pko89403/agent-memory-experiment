"""합성 조립 — NemoriMethod에 AmemNoteStore를 주입한다.

클래스 래퍼가 없다: 합성판은 "semantic store가 다른 NemoriMethod"라서
구성(construction)이 전부다 (YAGNI). 러너·노트북은 반환된 NemoriMethod를
그대로 쓰며, method._d_s가 AmemNoteStore다.

하이퍼파라미터 (2026-07-25 확정):
- nemori 측 전부 baseline 값 그대로 — 통제 변인 (NemoriConfig 기본값)
- evo_k=5 — A-MEM evolution 이웃 수 (원본 find_related_memories k=5 차용)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from memlab.embedding import embed as default_embed
from memlab.llm import LLMProvider
from memlab.methods.amem import AmemNoteStore
from memlab.methods.nemori import NemoriConfig, NemoriMethod
from typing import Callable


@dataclass(frozen=True)
class NemoriAmemConfig:
    nemori: NemoriConfig = field(default_factory=NemoriConfig)
    evo_k: int = 5  # evolution 이웃 수 (A-MEM 원본 코드 차용)


def build_nemori_amem(
    llm: LLMProvider,
    config: NemoriAmemConfig = NemoriAmemConfig(),
    embed: Callable[[str], np.ndarray] = default_embed,
) -> NemoriMethod:
    # embed를 양쪽에 명시적으로 꿴다 — evoke가 method 질의 벡터와 store
    # note 벡터를 비교하므로 두 embed는 반드시 동일해야 한다. 각자 기본값에
    # 맡기면 한쪽만 커스텀 주입될 때 조용히 다른 공간이 된다 (검증 리뷰 A2)
    store = AmemNoteStore(llm, embed=embed, evo_k=config.evo_k)
    return NemoriMethod(llm, embed=embed, config=config.nemori, semantic_store=store)
