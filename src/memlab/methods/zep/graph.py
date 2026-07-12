"""Neo4j 접근 계층 — 논문 Sec 2의 그래프 G와 3계층 subgraph를 클래스로 옮긴 것.

    논문                         여기                          소유물
    ────────────────────────    ──────────────────────       ─────────────────────────
    G = (N, E, φ)               ZepGraph                     driver, reset, close
    G_e  episode subgraph       EpisodeSubgraph (.episode)   (:Episodic) + [:MENTIONS]
    G_s  semantic entity        SemanticEntitySubgraph       (:Entity) + [:RELATES_TO]
         subgraph                 (.semantic)
    G_c  community subgraph     CommunitySubgraph            (:Community) + [:HAS_MEMBER]
                                  (.community)

논문의 subgraph 정의가 각자 자기 엣지 집합을 포함하므로(E_e ⊆ G_e 등)
episodic 엣지는 .episode가, community 엣지는 .community가 소유한다.

이 계층의 소임은 "Cypher 문자열 + 실행 + hydrate"뿐이다. 계층별 write
로직은 여기 없다 — G_s의 resolution·dedup·invalidation은 llm_ops.py/
method.py에, G_c의 label propagation은 communities.py에 산다. READ
경로(Sec 3의 φ·ρ·χ)는 search.py가 자기 Cypher를 직접 가진다.

논문 근거: 그래프 통합은 LLM이 쿼리를 생성하는 게 아니라 "predefined
Cypher queries"에 값만 채운다 (Sec 2.2.1) — 이 파일의 함수들이 그
"미리 정의된 쿼리"들이다.

논문이 침묵해 정한 것:
- vector 검색은 `db.index.vector.query*` 프로시저를 쓴다. 서버
  2026.06.0에서 deprecated 경고가 뜨지만(신문법 SEARCH로 대체 예정)
  정상 동작을 실측했고, 서버는 로컬 고정이다. 경고는 드라이버의
  notifications_min_severity="OFF"로 끈다.
  # ponytail: 서버 업그레이드로 프로시저가 제거되면 SEARCH 문법으로 이행
- Neo4j vector index의 cosine 스코어는 (1+cos)/2로 정규화된다 —
  직교 벡터가 0.5로 나온다 (2026-07-09 실측). 순위만 쓰므로 무해.
- edges_between은 방향을 무시한다 — 논문의 dedup 제약은 "edges existing
  between the same entity pairs" (Sec 2.2.2)로 방향을 말하지 않고,
  A→B와 B→A의 같은 fact는 중복이 맞다.
- invalidation 후보는 endpoint 공유 엣지다 (edges_touching). 논문의
  "semantically related existing edges" (Sec 2.2.3)는 선별 함수를 정의하지
  않는데, 원본은 source/target을 공유하는 엣지 pool로 좁힌다
  (graphiti.py:384-448) — 모순은 같은 entity에 붙은 fact 사이에서만
  성립하고, 전역 cosine은 무관한 entity의 유사 문장을 끌어와 오판 위험만
  키운다 (2026-07-10 합의). 원본은 pool을 hybrid search로 뽑지만 우리
  규모에선 entity 차수가 작아 최신순 상한으로 충분.
- recent()는 created_at(T′) 정렬이다 — LoCoMo의 t_ref는 세션 단위라
  세션 내 message 순서를 담지 못한다 (Sec 2.2.1의 "last n messages"는
  ingestion 순서로 해석).
- fulltext 질의는 Lucene 특수문자를 공백으로 치환해 문법 주입을 막는다.
  질의 32단어 초과 시 Lucene이 빈 결과를 내는 문제(원본 search_utils.py
  MAX_QUERY_LENGTH)는 긴 질의가 생기는 search.py에서 다룬다.
- reset()은 인덱스를 drop 후 재생성한다 — CREATE ... IF NOT EXISTS는
  이름이 같으면 속성 정의가 달라도 조용히 재사용해 검색이 빈 결과를
  내는 사고를 친다 (2026-07-09 실측). 빈 DB 위 재생성이라 비용 없음.
- embedding은 plain SET으로 저장한다. 원본은 db.create.set*VectorProperty
  프로시저를 쓰는데(구버전 Neo4j는 float array 타입만 인덱싱), 2026.06은
  list도 vector index에 들어감을 실측했다. 서버를 낮추면 프로시저로 전환.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Callable

from neo4j import GraphDatabase

from memlab.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from memlab.methods.zep.schema import (
    CommunityNode,
    EpisodeNode,
    EntityNode,
    SemanticEdge,
)

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 (memlab.embedding.embed)

_LUCENE_SPECIALS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')

_VECTOR_OPTIONS = (
    "OPTIONS {indexConfig: {`vector.dimensions`: %d, "
    "`vector.similarity_function`: 'cosine'}}" % EMBEDDING_DIM
)

_INDEXES = {
    # 검색 필드는 논문 Sec 3.1의 표 그대로: E_s→fact, N_s→name, N_c→name.
    # entity fulltext에 summary가 들어가는 건 resolution 후보 검색(Sec 2.2.1:
    # "full-text search on existing entity names and summaries") 때문.
    "fact_fulltext": "CREATE FULLTEXT INDEX fact_fulltext "
    "FOR ()-[r:RELATES_TO]-() ON EACH [r.fact]",
    "entity_fulltext": "CREATE FULLTEXT INDEX entity_fulltext "
    "FOR (n:Entity) ON EACH [n.name, n.summary]",
    "community_fulltext": "CREATE FULLTEXT INDEX community_fulltext "
    "FOR (n:Community) ON EACH [n.name]",
    "fact_vec": "CREATE VECTOR INDEX fact_vec "
    "FOR ()-[r:RELATES_TO]-() ON r.fact_embedding " + _VECTOR_OPTIONS,
    "entity_vec": "CREATE VECTOR INDEX entity_vec "
    "FOR (n:Entity) ON n.name_embedding " + _VECTOR_OPTIONS,
    "community_vec": "CREATE VECTOR INDEX community_vec "
    "FOR (n:Community) ON n.name_embedding " + _VECTOR_OPTIONS,
}

EDGE_RETURN = (
    "RETURN r.uuid AS uuid, startNode(r).uuid AS source_uuid, "
    "endNode(r).uuid AS target_uuid, r.relation AS relation, r.fact AS fact, "
    "r.fact_embedding AS fact_embedding, r.episode_uuids AS episode_uuids, "
    "r.created_at AS created_at, r.valid_at AS valid_at, "
    "r.invalid_at AS invalid_at, r.expired_at AS expired_at"
)

ENTITY_RETURN = (
    "RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary, "
    "n.name_embedding AS name_embedding, n.created_at AS created_at"
)

COMMUNITY_RETURN = ENTITY_RETURN  # CommunityNode 필드가 EntityNode와 동일 shape

# subgraph들이 주입받는 실행기: (query, **params) -> records
Runner = Callable[..., list]


def sanitize_fulltext(text: str) -> str:
    """Lucene 문법 문자를 공백으로 — 값이 쿼리 문법으로 해석되는 것을 차단."""
    return _LUCENE_SPECIALS.sub(" ", text).strip()


def _dt(value) -> datetime | None:
    return value.to_native() if value is not None else None


def _episode_from(record) -> EpisodeNode:
    return EpisodeNode(
        uuid=record["uuid"],
        content=record["content"],
        speaker=record["speaker"],
        t_ref=_dt(record["t_ref"]),
        created_at=_dt(record["created_at"]),
        semantic_edge_uuids=tuple(record["semantic_edge_uuids"] or ()),
    )


def entity_from(record) -> EntityNode:
    return EntityNode(
        uuid=record["uuid"],
        name=record["name"],
        summary=record["summary"],
        name_embedding=tuple(record["name_embedding"]),
        created_at=_dt(record["created_at"]),
    )


def community_from(record) -> CommunityNode:
    return CommunityNode(
        uuid=record["uuid"],
        name=record["name"],
        summary=record["summary"],
        name_embedding=tuple(record["name_embedding"]),
        created_at=_dt(record["created_at"]),
    )


def edge_from(record) -> SemanticEdge:
    return SemanticEdge(
        uuid=record["uuid"],
        source_uuid=record["source_uuid"],
        target_uuid=record["target_uuid"],
        relation=record["relation"],
        fact=record["fact"],
        fact_embedding=tuple(record["fact_embedding"]),
        episode_uuids=tuple(record["episode_uuids"] or ()),
        created_at=_dt(record["created_at"]),
        valid_at=_dt(record["valid_at"]),
        invalid_at=_dt(record["invalid_at"]),
        expired_at=_dt(record["expired_at"]),
    )


class EpisodeSubgraph:
    """G_e — episodic 노드 + episodic 엣지 E_e ⊆ φ*(N_e×N_s) (Sec 2.1)."""

    def __init__(self, run: Runner):
        self._run = run

    def save(self, ep: EpisodeNode) -> None:
        self._run(
            "MERGE (n:Episodic {uuid: $uuid}) "
            "SET n.content = $content, n.speaker = $speaker, n.t_ref = $t_ref, "
            "n.created_at = $created_at, n.semantic_edge_uuids = $edge_uuids",
            uuid=ep.uuid,
            content=ep.content,
            speaker=ep.speaker,
            t_ref=ep.t_ref,
            created_at=ep.created_at,
            edge_uuids=list(ep.semantic_edge_uuids),
        )

    def link_mentions(self, episode_uuid: str, entity_uuid: str) -> None:
        """E_e — episode → 언급된 entity (속성 없는 순수 연결)."""
        self._run(
            "MATCH (e:Episodic {uuid: $ep}), (n:Entity {uuid: $ent}) "
            "MERGE (e)-[:MENTIONS]->(n)",
            ep=episode_uuid,
            ent=entity_uuid,
        )

    def recent(self, limit: int) -> list[EpisodeNode]:
        """직전 limit개 message, 시간순 — entity extraction 맥락 (Sec 2.2.1 n=4)."""
        records = self._run(
            "MATCH (n:Episodic) "
            "RETURN n.uuid AS uuid, n.content AS content, n.speaker AS speaker, "
            "n.t_ref AS t_ref, n.created_at AS created_at, "
            "n.semantic_edge_uuids AS semantic_edge_uuids "
            "ORDER BY n.created_at DESC LIMIT $limit",
            limit=limit,
        )
        return [_episode_from(r) for r in reversed(records)]


class SemanticEntitySubgraph:
    """G_s — entity 노드 + semantic 엣지 E_s ⊆ φ*(N_s×N_s) (Sec 2.2)."""

    def __init__(self, run: Runner):
        self._run = run

    def save_entity(self, node: EntityNode) -> None:
        self._run(
            "MERGE (n:Entity {uuid: $uuid}) "
            "SET n.name = $name, n.summary = $summary, "
            "n.name_embedding = $embedding, n.created_at = $created_at",
            uuid=node.uuid,
            name=node.name,
            summary=node.summary,
            embedding=list(node.name_embedding),
            created_at=node.created_at,
        )

    def save_edge(self, edge: SemanticEdge) -> None:
        self._run(
            "MATCH (a:Entity {uuid: $source}), (b:Entity {uuid: $target}) "
            "MERGE (a)-[r:RELATES_TO {uuid: $uuid}]->(b) "
            "SET r.relation = $relation, r.fact = $fact, "
            "r.fact_embedding = $embedding, r.episode_uuids = $episode_uuids, "
            "r.created_at = $created_at, r.valid_at = $valid_at, "
            "r.invalid_at = $invalid_at, r.expired_at = $expired_at",
            source=edge.source_uuid,
            target=edge.target_uuid,
            uuid=edge.uuid,
            relation=edge.relation,
            fact=edge.fact,
            embedding=list(edge.fact_embedding),
            episode_uuids=list(edge.episode_uuids),
            created_at=edge.created_at,
            valid_at=edge.valid_at,
            invalid_at=edge.invalid_at,
            expired_at=edge.expired_at,
        )

    def entity_candidates(
        self, name_embedding: tuple[float, ...], text: str, limit: int
    ) -> list[EntityNode]:
        """resolution 후보 — name cosine + name·summary fulltext 합집합 (Sec 2.2.1).

        질의는 새 이름 하나뿐이고, fulltext가 뒤지는 summary는 기존 노드에
        '저장된' 과거 요약이다 — 이름은 안 비슷해도 요약 본문에 그 단어가
        등장하는 노드까지 후보로 올린다 (연락처 중복 검사에서 메모란까지
        검색하는 격). 그물은 넓게(recall), 판정은 resolve_entity가 이름만
        보고 좁게(precision) — 역할 분담이 의도다.
        """
        records = self._run(
            "CALL db.index.vector.queryNodes('entity_vec', $limit, $vec) "
            "YIELD node AS n " + ENTITY_RETURN,
            limit=limit,
            vec=list(name_embedding),
        )
        merged = {r["uuid"]: entity_from(r) for r in records}
        query = sanitize_fulltext(text)
        if query:
            records = self._run(
                "CALL db.index.fulltext.queryNodes('entity_fulltext', $q, "
                "{limit: $limit}) YIELD node AS n " + ENTITY_RETURN,
                q=query,
                limit=limit,
            )
            for r in records:
                merged.setdefault(r["uuid"], entity_from(r))
        return list(merged.values())

    def edges_between(self, a_uuid: str, b_uuid: str, limit: int) -> list[SemanticEdge]:
        """dedup 후보 — 같은 entity pair 사이 엣지만, 방향 무시, 최신순 상한
        (Sec 2.2.2; 상한은 원본 RELEVANT_SCHEMA_LIMIT 상응 — hot pair가
        세션을 거듭해 쌓이면 dedup 프롬프트가 context를 넘친다, 2026-07-10)."""
        records = self._run(
            "MATCH (a:Entity {uuid: $a})-[r:RELATES_TO]-(b:Entity {uuid: $b}) "
            + EDGE_RETURN + " ORDER BY r.created_at DESC LIMIT $limit",
            a=a_uuid,
            b=b_uuid,
            limit=limit,
        )
        return [edge_from(r) for r in records]

    def entity_summaries(self) -> dict[str, str]:
        """전체 entity의 uuid → summary — community 요약의 map-reduce 재료 (Sec 2.3)."""
        records = self._run(
            "MATCH (n:Entity) RETURN n.uuid AS uuid, n.summary AS summary"
        )
        return {r["uuid"]: r["summary"] for r in records}

    def neighbor_counts(self) -> dict[str, dict[str, int]]:
        """uuid → {이웃 uuid: RELATES_TO 개수} — label propagation의 가중 projection.

        이웃 없는 entity는 결과에 없다 — 호출부(communities.py)가 전체
        uuid 집합으로 채운다. invalidated 엣지 포함 여부는 communities.py
        docstring 참고.
        """
        records = self._run(
            "MATCH (n:Entity)-[r:RELATES_TO]-(m:Entity) "
            "RETURN n.uuid AS uuid, m.uuid AS neighbor, count(r) AS weight"
        )
        counts: dict[str, dict[str, int]] = {}
        for r in records:
            counts.setdefault(r["uuid"], {})[r["neighbor"]] = r["weight"]
        return counts

    def edges_touching(self, entity_uuid: str, limit: int) -> list[SemanticEdge]:
        """invalidation 후보 — 이 entity가 한쪽 끝인 엣지들, 최신순 (Sec 2.2.3)."""
        records = self._run(
            "MATCH (n:Entity {uuid: $uuid})-[r:RELATES_TO]-() " + EDGE_RETURN +
            " ORDER BY r.created_at DESC LIMIT $limit",
            uuid=entity_uuid,
            limit=limit,
        )
        return [edge_from(r) for r in records]


class CommunitySubgraph:
    """G_c — community 노드 + community 엣지 E_c ⊆ φ*(N_c×N_s) (Sec 2.3)."""

    def __init__(self, run: Runner):
        self._run = run

    def save(self, node: CommunityNode) -> None:
        self._run(
            "MERGE (n:Community {uuid: $uuid}) "
            "SET n.name = $name, n.summary = $summary, "
            "n.name_embedding = $embedding, n.created_at = $created_at",
            uuid=node.uuid,
            name=node.name,
            summary=node.summary,
            embedding=list(node.name_embedding),
            created_at=node.created_at,
        )

    def link_member(self, community_uuid: str, entity_uuid: str) -> None:
        """E_c — community → 소속 entity (속성 없는 순수 연결)."""
        self._run(
            "MATCH (c:Community {uuid: $comm}), (n:Entity {uuid: $ent}) "
            "MERGE (c)-[:HAS_MEMBER]->(n)",
            comm=community_uuid,
            ent=entity_uuid,
        )

    def member_of(self, entity_uuid: str) -> CommunityNode | None:
        """entity가 속한 community — dynamic extension의 소속 확인 (Sec 2.3)."""
        records = self._run(
            "MATCH (n:Community)-[:HAS_MEMBER]->(:Entity {uuid: $uuid}) "
            + COMMUNITY_RETURN,
            uuid=entity_uuid,
        )
        return community_from(records[0]) if records else None

    def neighbor_communities(self, entity_uuid: str) -> list[CommunityNode]:
        """entity 이웃들이 속한 community — (member, edge) 조합당 1행.

        중복이 곧 다수결 가중치다 (원본 determine_entity_community의
        Cypher도 DISTINCT 없이 행을 센다).
        """
        records = self._run(
            "MATCH (n:Community)-[:HAS_MEMBER]->(:Entity)"
            "-[:RELATES_TO]-(:Entity {uuid: $uuid}) " + COMMUNITY_RETURN,
            uuid=entity_uuid,
        )
        return [community_from(r) for r in records]

    def remove_all(self) -> None:
        """full rebuild 준비 — 원본 build_communities도 전부 지우고 재구축."""
        self._run("MATCH (n:Community) DETACH DELETE n")


class ZepGraph:
    """G = (N, E, φ) 전체 — subgraph 셋의 조립체. 대화 하나 = 그래프 하나.

    생성 시 reset()으로 전체 삭제 + 인덱스 구축. 인덱스는 subgraph를 가로지르는
    전역 스키마라 G 소관이다.
    """

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(
            uri, auth=(user, password), notifications_min_severity="OFF"
        )
        self.episode = EpisodeSubgraph(self.run)  # G_e
        self.semantic = SemanticEntitySubgraph(self.run)  # G_s
        self.community = CommunitySubgraph(self.run)  # G_c
        self.reset()

    def close(self) -> None:
        self._driver.close()

    def run(self, query: str, **params):
        records, _, _ = self._driver.execute_query(query, **params)
        return records

    def reset(self) -> None:
        self.run("MATCH (n) DETACH DELETE n")
        for name, ddl in _INDEXES.items():
            self.run(f"DROP INDEX {name} IF EXISTS")
            self.run(ddl)
        self.run("CALL db.awaitIndexes(60)")
