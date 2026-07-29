"""A-MEM 메모리의 스키마 — 논문 Sec 3.1의 형식화를 코드로 옮긴 것.

논문 형식화와 대응 (arXiv 2502.12110 v11, Eq.1):

    논문 (Sec 3.1)                        여기
    ─────────────────────────────────    ──────────────────
    m_i = {c_i, t_i, K_i, G_i, X_i,      MemoryNote
           e_i, L_i}
      c_i  원문 내용                        .content
      t_i  상호작용 시각                     .t
      K_i  LLM 생성 keywords               .keywords
      G_i  LLM 생성 tags                   .tags
      X_i  LLM 생성 contextual description  .context
      e_i = f_enc(concat(c,K,G,X)) (Eq.3)  .embedding
      L_i  링크된 note 집합                 .links (uuid 목록)

이 클래스는 **mutable**이다 — nemori(frozen + supersede)와 반대 결정.
Memory Evolution(§3.3)이 이웃 note의 context·tags를 제자리 갱신하고
strengthen이 links를 덧붙이는 것이 방법의 본질이라, 불변 객체로는
방법 자체를 표현할 수 없다. 갱신 시 임베딩 일관성은 store가 책임진다
(K/G/X가 바뀌면 Eq.3대로 재임베딩 — 논문 침묵, 우리 결정 2026-07-25).

합성 맥락 (nemori_amem): 이 note의 content는 Nemori가 distill한
SemanticInsight.statement다. t는 출처 episode의 occurred_at을 물려받는다
(insight 자체는 무시간 fact — A-MEM 검색 컨텍스트의 "talk start time"이
의미를 갖게 하는 결정, 2026-07-25 합의).

논문에 없어 제외한 것 (원본 코드의 추가 필드 — "원본에 없으면 제외" 합의):
- retrieval_count, last_accessed: 코드에서 증가·갱신되는 곳만 있고
  소비하는 곳이 없다 (죽은 필드).
- evolution_history: 기록만 하고 아무도 읽지 않는다.
- category: 기본값 "Uncategorized"로 태어나 바뀌는 로직이 없다.
- source_episode_uuid(초기 설계): 우리 스스로도 write-only로 만들었다가
  같은 기준으로 제거 (검증 리뷰 A14). provenance가 필요해지는 시점
  (가이드 노트북)에 한 줄로 복원 가능.

statement 프로퍼티는 nemori semantic store 프로토콜과의 브리지다 —
nemori의 답변·예측 컨텍스트 조립(_statements_text)이 .statement를 읽는데,
이 store가 그 자리에 주입되므로 note가 같은 표면을 제공한다. 표시되는
것은 content(=insight 원문)뿐이고 K/G/X는 임베딩·evolution·링크로만
일한다 (통제 실험 결정 (i), 2026-07-25 합의).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


def new_uuid() -> str:
    return str(uuid4())


@dataclass
class MemoryNote:
    """m = {c, t, K, G, X, e, L} — note store의 엔트리 (Eq.1). mutable."""

    content: str  # c — 원문 (합성에서는 Nemori insight의 statement)
    t: datetime  # t — 상호작용 시각 (합성에서는 출처 episode의 occurred_at)
    keywords: list[str]  # K — Ps1 생성
    tags: list[str]  # G — Ps1 생성, evolution이 갱신
    context: str  # X — Ps1 생성 한 문장 요약, evolution이 갱신
    embedding: tuple[float, ...]  # e = f_enc(concat(c,K,G,X)) — 갱신 시 재계산
    # 논문 밖 필드 — nemori evoke 게이트 전용 벡터 (f_emb(content), 검증 리뷰
    # A27). τ=0.55는 statement 공간 실측값인데 concat 공간은 분포가 통째로
    # 내려앉아 게이트가 닫힌다 (실측 3/14 vs baseline 6/7). content가 불변
    # 이라 evolution의 재임베딩과 무관하게 고정이다.
    statement_embedding: tuple[float, ...]
    links: list[str] = field(default_factory=list)  # L — 링크된 note uuid들
    uuid: str = field(default_factory=new_uuid)

    @property
    def statement(self) -> str:
        """nemori semantic store 프로토콜 브리지 — 표시용 표면은 content뿐."""
        return self.content
