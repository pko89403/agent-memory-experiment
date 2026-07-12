"""Memory retrieval — 논문 Sec 3: f(α) = χ(ρ(φ(α))) = β.

질문 α 하나가
    φ  search      — cosine + Okapi BM25 (+ edge·entity는 BFS)로
                     semantic edge·entity·community 수집
    ρ  reranker    — RRF (Reciprocal Rank Fusion)
    χ  constructor — FACTS/ENTITIES/COMMUNITIES context 조립
를 거쳐 context 문자열 β가 된다. β는 AnswerGenerator의 {context}로 들어간다.

검색 필드는 Sec 3.1의 표: E_s → fact, N_s → name, N_c → name (entity
fulltext는 원본처럼 name+summary 인덱스 — 아래 참고). READ 경로의 Cypher는
이 파일 소유 (graph.py 분담 참조).

논문이 침묵해 정한 것 (원본 graphiti search.py/search_utils.py 대조):
- reranker는 RRF만 구현 (2026-07-10 합의). 논문은 5종(RRF·MMR·episode-
  mentions·node distance·cross-encoder)을 "지원"한다고만 하고 실험 config는
  침묵 — 원본 기본 recipe가 RRF(rank_const=1)라 그대로 차용. 나머지는 실험
  증거가 없어 미구현이고, cross-encoder는 모델이 하나 더 필요해 로컬 환경과
  안 맞는다. LLM 계열 reranker(ranksmith 등)는 baseline 확보 후 ablation 후보.
- limit은 edge 10 + entity 10 — DMR 실험의 "top 10 most relevant nodes and
  edges" (Sec 4.1). 각 sub-search는 원본대로 2×limit씩 모아 RRF 후 자른다.
- community 검색은 name에 BM25·cosine 두 방법 + RRF (원본
  community_search와 동일 — BFS 축은 원본에도 없다). limit 10은 논문이
  침묵해 원본 DEFAULT_SEARCH_LIMIT 차용. χ는 summary만 노출한다 (Sec 3의
  χ 정의: N_c → summary field). 논문 template·실험 서술(Sec 4)은
  FACTS/ENTITIES만 다루지만 χ의 형식 정의를 따라 COMMUNITIES 섹션을
  포함한다 (2026-07-10 결정 — 섹션 신설 경위는 prompt_templates.py).
- BFS: depth는 원본 Cypher가 하드코딩한 {1,3}(파라미터는 받고도 안 쓰는
  quirk), directed 순회도 원본 그대로. seed는 논문의 "recent episodes"가
  아니라 같은 질의의 BM25·cosine hit들이다 (edge는 hit의 source entity,
  entity는 hit 자신) — recent seed는 실시간 대화 시나리오용이고 post-hoc
  QA에는 원본의 fallback(search.py:155-159)이 맞다. seed가 항상 entity라
  원본 패턴의 Episodic origin과 MENTIONS hop은 제외했다.
- min_score 0.6 (원본 DEFAULT_MIN_SCORE): 원본은 vector.similarity.cosine의
  [0,1] 정규화 스코어에 적용하고, 우리 vector index score도 같은 정규화
  (graph.py 참조) — 동일 semantics.
- entity fulltext는 name+summary 인덱스를 그대로 쓴다 — Sec 3.1 표는 name만
  적지만 원본 node 검색 인덱스가 name+summary고, resolution(Sec 2.2.1)과
  인덱스를 공유한다.
- BM25 질의는 sanitize 후 32단어 이상이면 skip (원본 MAX_QUERY_LENGTH) —
  Lucene이 긴 질의에 빈 결과를 내는 문제의 원본 대처를 차용.
- invalidated edge도 검색에 포함된다 (원본에 필터 없음) — bi-temporal의
  요점대로 χ가 date range를 노출하고 판단은 답변 LLM의 몫.
- χ의 날짜 표기(논문 침묵): 채점 요구 형식과 동일한 "15 July 2023",
  둘 다 없으면 suffix 생략, valid만 있으면 "- present", invalid만 있으면
  "until". 첫 결정(2026-07-10: ISO + unknown/present 필러)은 conv-26
  temporal 채점에서 필러·ISO가 답변에 그대로 복사되는 것이 실측돼 교체
  (2026-07-12, _fact_line 주석).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Callable

from memlab.methods.zep import prompt_templates as prompts
from memlab.methods.zep.graph import (
    COMMUNITY_RETURN,
    EDGE_RETURN,
    ENTITY_RETURN,
    Runner,
    community_from,
    edge_from,
    entity_from,
    sanitize_fulltext,
)
from memlab.methods.zep.schema import CommunityNode, EntityNode, SemanticEdge

EDGE_LIMIT = 10  # DMR 실험 config (Sec 4.1)
ENTITY_LIMIT = 10
COMMUNITY_LIMIT = 10  # 논문 침묵 — 원본 DEFAULT_SEARCH_LIMIT
MIN_SCORE = 0.6  # 원본 DEFAULT_MIN_SCORE — [0,1] 정규화 cosine 기준
MAX_QUERY_WORDS = 32  # 원본 MAX_QUERY_LENGTH — 이상이면 BM25 skip

_DATE_FMT = "%-d %B %Y"  # 채점이 요구하는 답변 형식("15 July 2023")과 동일 표기


def rrf(rankings: list[list[str]], rank_const: int = 1) -> list[str]:
    """ρ — Reciprocal Rank Fusion (Sec 3.2). 원본 search_utils.rrf 그대로."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for i, uuid in enumerate(ranking):
            scores[uuid] += 1 / (i + rank_const)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def _fact_line(edge: SemanticEdge) -> str:
    # 무날짜 fact는 suffix 생략 — "(Date range: unknown - present)"를 fact마다
    # 붙였더니 그 필러가 답변에 그대로 복사되는 것을 실측 (conv-26 temporal
    # 채점, 2026-07-12). ISO 표기도 같은 경로로 새서 _DATE_FMT를 교체.
    if not edge.valid_at and not edge.invalid_at:
        return edge.fact
    if edge.valid_at and edge.invalid_at:
        rng = f"{edge.valid_at.strftime(_DATE_FMT)} - {edge.invalid_at.strftime(_DATE_FMT)}"
    elif edge.valid_at:
        rng = f"{edge.valid_at.strftime(_DATE_FMT)} - present"
    else:
        rng = f"until {edge.invalid_at.strftime(_DATE_FMT)}"
    return f"{edge.fact} (Date range: {rng})"


def construct_context(
    edges: list[SemanticEdge],
    entities: list[EntityNode],
    communities: list[CommunityNode],
) -> str:
    """χ (Sec 3) — FACTS/ENTITIES/COMMUNITIES를 template에 채운다."""
    return prompts.CONTEXT_TEMPLATE.format(
        facts="\n".join(_fact_line(e) for e in edges),
        entities="\n".join(f"{n.name}: {n.summary}" for n in entities),
        communities="\n".join(c.summary for c in communities),
    )


class MemoryRetrieval:
    """f: 질문 → context 문자열. run은 ZepGraph.run, embed는 공용 임베딩."""

    def __init__(
        self,
        run: Runner,
        embed: Callable[[str], Sequence[float]],
        edge_limit: int = EDGE_LIMIT,
        entity_limit: int = ENTITY_LIMIT,
        community_limit: int = COMMUNITY_LIMIT,
    ):
        self._run = run
        self._embed = embed
        self._edge_limit = edge_limit
        self._entity_limit = entity_limit
        self._community_limit = community_limit

    def retrieve(self, query: str) -> str:
        """f(α) = χ(ρ(φ(α))) = β — 아래 세 축이 φ+ρ, construct_context가 χ."""
        vec = [float(x) for x in self._embed(query)]
        return construct_context(
            self._search_edges(query, vec),  # E_s: fact로 검색
            self._search_entities(query, vec),  # N_s: name으로 검색
            self._search_communities(query, vec),  # N_c: name으로 검색
        )

    # --- φ + ρ: 축별로 BM25 → cosine → (hit seed) BFS 랭킹 3개를 RRF ---

    def _search_edges(self, query: str, vec: list[float]) -> list[SemanticEdge]:
        limit = 2 * self._edge_limit
        bm25 = self._edge_bm25(query, limit)  # φ_bm25 — 단어 겹침 (Sec 3.1)
        cosine = self._edge_cosine(vec, limit)  # φ_cos — 의미 유사
        origins = list(dict.fromkeys(e.source_uuid for e in bm25 + cosine))
        bfs = self._edge_bfs(origins, limit)  # φ_bfs — hit 이웃의 맥락 유사
        return _fuse(bm25, cosine, bfs, self._edge_limit)  # ρ = RRF (Sec 3.2)

    def _search_entities(self, query: str, vec: list[float]) -> list[EntityNode]:
        limit = 2 * self._entity_limit
        bm25 = self._entity_bm25(query, limit)
        cosine = self._entity_cosine(vec, limit)
        origins = list(dict.fromkeys(n.uuid for n in bm25 + cosine))
        bfs = self._node_bfs(origins, limit)
        return _fuse(bm25, cosine, bfs, self._entity_limit)

    def _search_communities(self, query: str, vec: list[float]) -> list[CommunityNode]:
        limit = 2 * self._community_limit
        bm25 = self._community_bm25(query, limit)
        cosine = self._community_cosine(vec, limit)
        return _fuse(bm25, cosine, [], self._community_limit)

    # --- sub-search 8종: E_s(fact) 3개 + N_s(name) 3개 + N_c(name) 2개 ---

    def _edge_bm25(self, query: str, limit: int) -> list[SemanticEdge]:
        text = _bm25_query(query)
        if not text:
            return []
        records = self._run(
            "CALL db.index.fulltext.queryRelationships('fact_fulltext', $q, "
            "{limit: $limit}) YIELD relationship AS r " + EDGE_RETURN,
            q=text,
            limit=limit,
        )
        return [edge_from(r) for r in records]

    def _edge_cosine(self, vec: list[float], limit: int) -> list[SemanticEdge]:
        records = self._run(
            "CALL db.index.vector.queryRelationships('fact_vec', $limit, $vec) "
            "YIELD relationship AS r, score WHERE score > $min_score " + EDGE_RETURN,
            vec=vec,
            limit=limit,
            min_score=MIN_SCORE,
        )
        return [edge_from(r) for r in records]

    def _edge_bfs(self, origins: list[str], limit: int) -> list[SemanticEdge]:
        if not origins:
            return []
        records = self._run(
            "UNWIND $origins AS origin "
            "MATCH path = (:Entity {uuid: origin})-[:RELATES_TO]->{1,3}(:Entity) "
            "UNWIND relationships(path) AS r WITH DISTINCT r "
            + EDGE_RETURN + " LIMIT $limit",
            origins=origins,
            limit=limit,
        )
        return [edge_from(r) for r in records]

    def _entity_bm25(self, query: str, limit: int) -> list[EntityNode]:
        text = _bm25_query(query)
        if not text:
            return []
        records = self._run(
            "CALL db.index.fulltext.queryNodes('entity_fulltext', $q, "
            "{limit: $limit}) YIELD node AS n " + ENTITY_RETURN,
            q=text,
            limit=limit,
        )
        return [entity_from(r) for r in records]

    def _entity_cosine(self, vec: list[float], limit: int) -> list[EntityNode]:
        records = self._run(
            "CALL db.index.vector.queryNodes('entity_vec', $limit, $vec) "
            "YIELD node AS n, score WHERE score > $min_score " + ENTITY_RETURN,
            vec=vec,
            limit=limit,
            min_score=MIN_SCORE,
        )
        return [entity_from(r) for r in records]

    def _node_bfs(self, origins: list[str], limit: int) -> list[EntityNode]:
        if not origins:
            return []
        records = self._run(
            "UNWIND $origins AS origin "
            "MATCH (:Entity {uuid: origin})-[:RELATES_TO]->{1,3}(n:Entity) "
            "WITH DISTINCT n " + ENTITY_RETURN + " LIMIT $limit",
            origins=origins,
            limit=limit,
        )
        return [entity_from(r) for r in records]

    def _community_bm25(self, query: str, limit: int) -> list[CommunityNode]:
        text = _bm25_query(query)
        if not text:
            return []
        records = self._run(
            "CALL db.index.fulltext.queryNodes('community_fulltext', $q, "
            "{limit: $limit}) YIELD node AS n " + COMMUNITY_RETURN,
            q=text,
            limit=limit,
        )
        return [community_from(r) for r in records]

    def _community_cosine(self, vec: list[float], limit: int) -> list[CommunityNode]:
        records = self._run(
            "CALL db.index.vector.queryNodes('community_vec', $limit, $vec) "
            "YIELD node AS n, score WHERE score > $min_score " + COMMUNITY_RETURN,
            vec=vec,
            limit=limit,
            min_score=MIN_SCORE,
        )
        return [community_from(r) for r in records]


def _bm25_query(query: str) -> str:
    """sanitize + 32단어 가드 — 빈 문자열이면 fulltext를 건너뛰라는 뜻."""
    text = sanitize_fulltext(query)
    return "" if len(text.split()) >= MAX_QUERY_WORDS else text


def _fuse(bm25: list, cosine: list, bfs: list, limit: int) -> list:
    """ρ 적용: 랭킹 3개 RRF → 원본 순서대로 hydrate된 객체 상위 limit개."""
    pool = {item.uuid: item for item in bm25 + cosine + bfs}
    ranked = rrf([[item.uuid for item in ranking] for ranking in (bm25, cosine, bfs)])
    return [pool[uuid] for uuid in ranked][:limit]
