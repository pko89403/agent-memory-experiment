"""A-MEM note store — nemori ManagementSystem 프로토콜의 A-MEM 구현.

nemori의 SemanticStore(naive append)와 같은 표면(consolidate/evoke/search)을
제공해서, NemoriMethod의 semantic 저장소 자리에 그대로 꽂힌다 (합성 실험
nemori_amem의 핵심 부품). insight가 들어오면 note로 승격시키고(construct),
이웃과 링크·evolution시킨다(evolve).

    프로토콜 메서드     하는 일
    ─────────────────  ─────────────────────────────────────────────
    consolidate         insight마다: Ps1 구성 → concat 임베딩 →
                        이웃 evo_k 회수 → evolve → 링크·이웃 갱신·
                        재임베딩 → 저장 (insight당 LLM 2콜)
    evoke               cosine top-ks + τ 필터 — nemori 계약 그대로,
                        **statement 공간**(insight 임베딩)에서. 링크
                        없음: 예측 컨텍스트의 선발 메커니즘을 baseline
                        과 동일하게 유지. concat 공간은 τ=0.55 경계가
                        통째로 내려앉아(실측 3/14 vs baseline 6/7)
                        게이트가 닫히고 cascade가 direct_distill로
                        붕괴한다 (검증 리뷰 A6 실측 → A27 처방)
    search              직접 히트(순위순)와 히트당 링크 1개를 교차
                        배치, 총량 m 캡 (아래 A5 결정)

원본 코드보다 잘 구현한 것 (사용자 방침: 논문을 따르는 선에서 개선):
- 링크가 실제로 해소된다 — 원본은 인덱스 문자열이 links에 들어가 검색
  에서 영영 죽는 버그 (llm_ops·prompt_templates docstring 참고).
- update_neighbor가 **실제 이웃**을 갱신한다 — 원본은 검색 순위 인덱스로
  전역 메모리 리스트를 잘못 인덱싱해서 무관한 note들을 갱신하는 버그
  (memory_system.py:687-716).
- **양방향 링크** — 논문의 링크 정의("linked memories that *share*
  semantic relationships", §3.1)와 Zettelkasten의 유연한 연결 원리에
  따라, n→j 링크 시 j→n도 잇는다. 원본은 단방향이라 먼저 저장된 note를
  회수해도 나중 note가 링크로 딸려오지 않았다.
- **Eq.3 일관성** — 임베딩은 concat(c,K,G,X) (원본 store는 content만),
  evolution으로 K/G/X가 바뀐 note는 재임베딩. 무변경 응답(원값 그대로)은
  감지해서 재임베딩을 생략한다 — 단 이 가드는 자구 일치만 잡는다:
  9B가 같은 뜻을 바꿔 쓰면(paraphrase) 갱신·재임베딩이 일어나 표현이
  서서히 드리프트할 수 있다 (원본은 가드 자체가 없어 항상 갱신 — 우리가
  엄격히 낫지만 잔여 위험은 dry-run 계측, 검증 리뷰 A7).

검색·예산 결정 (검증 리뷰 A5):
- 원본 답변 검색(find_related_memories_raw)은 직접 히트 전원 보장 +
  히트당 링크를 상한 없이 첨부한다(총량 무제한). 우리는 답변 컨텍스트
  예산이 baseline 파리티로 m 고정이라 그대로 옮기면 허브 note(양방향
  백링크로 링크가 무한 성장)의 이웃이 하위 직접 히트를 밀어낸다.
  절충: **히트당 링크 1개** 교차 배치 — 직접 히트 최소 ⌈m/2⌉ 보장,
  링크 채널 유지. 슬롯 점유율(직접 vs 링크)은 dry-run 계측 항목.

논문이 침묵해 정한 것:
- concat의 직렬화: "{c} {K 공백 결합} {G 공백 결합} {X}" — Eq.3의 항
  순서대로. (nemori의 f_emb(c ∥ N) 공백 결합과 같은 해석.)
- 이웃 회수 시점은 새 note 저장 전 — 자기 자신이 이웃으로 잡히지 않는다
  (원본도 동일: process_memory가 retriever 등록 전에 돈다. 초기 docstring
  이 원본을 "저장 후 검색" 버그로 잘못 기록했던 것을 정정 — 검증 리뷰 A8).
- **evolution 이웃 검색 쿼리도 concat 임베딩** — 원본은 content 텍스트로
  질의하지만, 논문 Eq.4가 s_{n,j} = e_n·e_j로 양변 모두 note 임베딩
  (=Eq.3 concat)을 명세한다. 논문 우선 (검증 리뷰 A9). 부작용 위험:
  boilerplate 태그("communication" 류)가 쿼리를 지배해 무관 이웃을 모을
  수 있음 — dry-run 계측 항목.
- 답변 검색의 질의는 nemori 계약대로 **질문 임베딩** — 원본 eval의
  키워드 추출 LLM 콜(generate_query_llm)은 채택하지 않는다. 답변 경로를
  baseline과 동일하게 고정하는 통제 실험 설계 (검증 리뷰 A10).
- evolve는 이웃이 있으면 유사도 게이트 없이 매번 호출 — 원본 충실
  (nemori N13과 같은 계열: 게이트 추가는 원본 이탈, 비용 절감은 변형
  실험 재료. 이웃 max sim은 dry-run 계측 — 검증 리뷰 A24).
- store는 **두 공간을 색인한다**: nemori가 insight마다 넘기는 statement
  임베딩은 evoke 전용 벡터가 되고(A27 — 초기엔 불용·낭비로 기록했으나
  (A11) 게이트 파리티의 핵심이 됐다), search·이웃 검색은 Eq.3 concat.
  content 불변이라 statement 벡터는 evolution 재임베딩과 무관하게 고정.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Protocol

import numpy as np

from memlab.embedding import cosine_top_k
from memlab.embedding import embed as default_embed
from memlab.llm import LLMProvider
from memlab.methods.amem.llm_ops import NoteOps
from memlab.methods.amem.schema import MemoryNote


class DistilledInsight(Protocol):
    """소비자 선언 Protocol — nemori SemanticInsight가 충족한다 (교차 import
    금지). frozen dataclass가 충족할 수 있게 읽기 전용 property로 선언
    (검증 리뷰 A25). embedding은 statement 공간(f_emb(statement)) 벡터 —
    evoke 게이트가 이 공간에서 돈다 (검증 리뷰 A27)."""

    @property
    def statement(self) -> str: ...

    @property
    def embedding(self) -> tuple[float, ...]: ...


def _eq3_text(content: str, keywords: list[str], tags: list[str], context: str) -> str:
    # Eq.3 concat(c, K, G, X)의 직렬화
    return f"{content} {' '.join(keywords)} {' '.join(tags)} {context}"


class AmemNoteStore:
    def __init__(
        self,
        llm: LLMProvider,
        embed: Callable[[str], np.ndarray] = default_embed,
        *,
        evo_k: int,  # 기본값 없음 — 단일 진실은 NemoriAmemConfig (검증 리뷰 A19)
    ):
        self._ops = NoteOps(llm)
        self._embed = embed
        self._evo_k = evo_k
        self._notes: list[MemoryNote] = []
        self._vectors: list[np.ndarray] = []  # concat 공간 — search·이웃 (Eq.3)
        self._statement_vectors: list[np.ndarray] = []  # statement 공간 — evoke 전용 (A27)
        self._position: dict[str, int] = {}  # uuid → 리스트 위치 (재임베딩·링크 해소)

    # ── nemori ManagementSystem 프로토콜 ─────────────────────────────

    def consolidate(
        self, insights: Iterable[DistilledInsight], occurred_at: datetime
    ) -> None:
        for insight in insights:
            self._add_note(insight, occurred_at)

    def evoke(self, query: np.ndarray, ks: int, tau: float) -> list[MemoryNote]:
        # statement 공간 — τ=0.55의 실측 경계가 사는 공간에서 게이트를 돌려
        # baseline과 개폐가 같게 한다 (A27; concat 공간은 3/14로 닫혔었다)
        return [
            self._notes[i]
            for i in cosine_top_k(self._statement_vectors, query, ks, tau=tau)
        ]

    def search(self, query: np.ndarray, m: int) -> list[MemoryNote]:
        # 직접 히트 + 히트당 링크 1개 교차 배치 (A5 결정 — docstring 참고).
        # 링크 해소는 hard index — 삭제가 없는 지금 dangling은 불변식 위반
        # 이므로 조용히 넘기지 않는다 (fail-loud, 검증 리뷰 A16)
        picked: dict[str, MemoryNote] = {}
        for i in cosine_top_k(self._vectors, query, m):
            note = self._notes[i]
            picked.setdefault(note.uuid, note)
            if len(picked) >= m:
                break
            if note.links:
                linked = self._notes[self._position[note.links[0]]]
                picked.setdefault(linked.uuid, linked)
                if len(picked) >= m:
                    break
        return list(picked.values())

    @property
    def items(self) -> tuple[MemoryNote, ...]:
        return tuple(self._notes)  # 노트북 추적용 읽기 전용 뷰 (프로토콜 밖)

    # ── 내부 ─────────────────────────────────────────────────────────

    def _add_note(self, insight: DistilledInsight, occurred_at: datetime) -> None:
        keywords, context, tags = self._ops.construct(insight.statement)
        embedding = self._embed(_eq3_text(insight.statement, keywords, tags, context))
        note = MemoryNote(
            content=insight.statement,
            t=occurred_at,
            keywords=keywords,
            tags=tags,
            context=context,
            embedding=tuple(embedding),  # 완성 상태로만 태어난다 (검증 리뷰 A18)
            statement_embedding=tuple(insight.embedding),
        )

        # 이웃은 저장 전에 회수 — 자기 자신 배제
        neighbors = [
            self._notes[i] for i in cosine_top_k(self._vectors, embedding, self._evo_k)
        ]
        if neighbors:
            outcome = self._ops.evolve(note, neighbors)
            if outcome.link_uuids:
                note.links = list(outcome.link_uuids)
                for uuid in outcome.link_uuids:  # 양방향 링크
                    self._notes[self._position[uuid]].links.append(note.uuid)
            if outcome.tags is not None and outcome.tags != note.tags:
                note.tags = outcome.tags
                note.embedding = tuple(
                    self._embed(_eq3_text(note.content, note.keywords, note.tags, note.context))
                )
            for neighbor, new_context, new_tags in outcome.neighbor_updates:
                if new_context == neighbor.context and new_tags == neighbor.tags:
                    continue  # 무변경(자구 일치) — 재임베딩 생략
                neighbor.context = new_context
                neighbor.tags = new_tags
                vector = self._embed(
                    _eq3_text(neighbor.content, neighbor.keywords, neighbor.tags, neighbor.context)
                )
                self._vectors[self._position[neighbor.uuid]] = vector
                neighbor.embedding = tuple(vector)

        self._position[note.uuid] = len(self._notes)
        self._notes.append(note)
        self._vectors.append(np.asarray(note.embedding))
        self._statement_vectors.append(np.asarray(note.statement_embedding))
