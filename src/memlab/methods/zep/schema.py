"""Zep 지식 그래프의 스키마 — 논문 Sec 2의 3계층 subgraph를 코드로 옮긴 것.

논문 형식화: G = (N, E, φ). 노드 3종 + 엣지 3종이 있고, 여기서는
데이터를 가진 4종만 dataclass로 표현한다:

    논문 (Sec 2)                          여기                Neo4j
    ─────────────────────────────────    ─────────────       ──────────────────
    N_e  episode 노드                     EpisodeNode         (:Episodic)
    N_s  entity 노드                      EntityNode          (:Entity)
    N_c  community 노드                   CommunityNode       (:Community)
    E_s ⊆ φ*(N_s×N_s)  semantic edge     SemanticEdge        [:RELATES_TO {…}]
    E_e ⊆ φ*(N_e×N_s)  episodic edge     (dataclass 없음)     [:MENTIONS]
    E_c ⊆ φ*(N_c×N_s)  community edge   (dataclass 없음)     [:HAS_MEMBER]

E_e·E_c는 양 끝점 외에 아무 정보가 없는 순수 연결이라 Cypher
relationship으로만 존재한다 (BFS 순회와 mention 카운트는 graph.py의
Cypher가 relationship을 직접 다룬다). 원본 graphiti v0.5.2도 이 두
클래스에 추가 필드가 0개다 (edges.py:77, 341) — 클래스가 있는 이유는
라이브러리로서 uuid 단위 CRUD API가 필요해서였고, 우리는 아니다.

bi-temporal 모델 (Sec 2.1, 2.2.3) — 시간 축이 둘이다:
    T   현실 시간 (사실이 실제로 유효했던 구간)
    T'  ingestion 시간 (시스템이 그 사실을 알았던 구간)
논문 표기와 필드 대응 (SemanticEdge):
    t_valid → valid_at, t_invalid → invalid_at   (T)
    t'_created → created_at, t'_expired → expired_at   (T')
invalidation은 삭제가 아니다: 모순되는 새 fact가 오면 옛 엣지의
invalid_at에 새 엣지의 valid_at을 찍는다 (Sec 2.2.3). 그래서 dataclass는
frozen — 상태 변경은 dataclasses.replace() 후 재저장으로만 일어나고,
그래프의 진실은 항상 Neo4j 쪽이다 (Python 객체는 hydrate된 작업 사본).

논문이 침묵해 정한 것:
- embedding 차원: 논문은 entity name에 1024-dim(Sec 2.2.1), 실험은
  BGE-m3(Sec 4.1). 우리는 하네스 공용 all-MiniLM-L6-v2 384-dim을 쓴다 —
  메소드 간(MemoryOS vs Zep) 비교 조건을 맞추는 게 재현보다 우선.
- 원본의 group_id(멀티테넌트 파티션)·labels·source/source_description은
  제외 — 러너가 대화당 method를 새로 만들고 DB를 wipe하므로 파티션이
  필요 없고, episode 유형은 message뿐이다 (논문도 message만 다룸).
- relation 필드는 원본의 edge.name에 해당 — fact extraction 프롬프트
  (Appendix 6.1.3)의 all-caps relation_type이 들어간다 (예: WORKS_FOR).
- 클래스 이름은 논문 표기를 따라 SemanticEdge (원본 graphiti는 EntityEdge).
  fact는 엣지 자체가 아니라 엣지에 실리는 내용(검색 필드)이고, 같은 fact가
  여러 엣지로 실체화될 수 있어(hyper-edge, Sec 2.2.2) 클래스명 Fact는
  1:N 관계를 뒤틀어 부적합.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


def new_uuid() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class EpisodeNode:
    """G_e 노드 — 원본 message 한 덩이 (Sec 2.1). non-lossy 보관.

    semantic_edge_uuids ↔ SemanticEdge.episode_uuids가 논문의 "bidirectional
    indices" (Sec 2.1): 원문 → 파생 fact, fact → 출처 원문(citation)을
    양방향으로 추적한다.
    """

    content: str  # message 원문 ("speaker: text"가 아니라 text만; speaker는 별도 필드)
    speaker: str  # 발화 actor (Sec 2.1: "the associated actor who produced the utterance")
    t_ref: datetime  # reference timestamp — 상대 날짜("2주 전")를 절대화하는 기준 (Sec 2.1)
    created_at: datetime  # T' — ingestion 시각
    semantic_edge_uuids: tuple[str, ...] = ()
    uuid: str = field(default_factory=new_uuid)


@dataclass(frozen=True)
class EntityNode:
    """G_s 노드 — episode에서 추출·병합된 entity (Sec 2.2.1).

    name이 cosine·BM25 검색 필드다 (Sec 3.1). summary는 resolution 대조와
    context 조립(Sec 3의 constructor)에 쓰인다.
    """

    name: str
    summary: str
    name_embedding: tuple[float, ...]  # 임베딩은 구축 시점에 확정 — Optional 없음
    created_at: datetime
    uuid: str = field(default_factory=new_uuid)


@dataclass(frozen=True)
class SemanticEdge:
    """E_s — semantic edge (Sec 2.2.2) + bi-temporal 스탬프 4개 (Sec 2.2.3).

    fact는 이 엣지에 실리는 내용이자 검색 필드 (Sec 3.1). 같은 fact 문장이
    entity pair마다 별도 엣지로 실체화될 수 있다 (hyper-edge, Sec 2.2.2).
    valid_at/invalid_at이 None이면 "시간 정보가 fact에 없음" (temporal
    extraction 프롬프트 guideline 4, Appendix 6.1.5).
    """

    source_uuid: str  # entity
    target_uuid: str  # entity
    relation: str  # all-caps 술어 (예: WORKS_FOR)
    fact: str  # 관계 서술 문장 (예: "Caroline works at Google")
    fact_embedding: tuple[float, ...]
    episode_uuids: tuple[str, ...]  # 출처 episode들 — 같은 fact의 재언급이 쌓인다
    created_at: datetime  # t'_created (T')
    valid_at: datetime | None = None  # t_valid (T)
    invalid_at: datetime | None = None  # t_invalid (T)
    expired_at: datetime | None = None  # t'_expired (T') — invalidation된 시각
    uuid: str = field(default_factory=new_uuid)


@dataclass(frozen=True)
class CommunityNode:
    """G_c 노드 — 강하게 연결된 entity 클러스터의 요약 (Sec 2.3).

    name은 요약에서 뽑은 핵심 용어·주제 모음이고, 임베딩되어 cosine 검색
    필드가 된다 (Sec 2.3, 3.1). summary는 member를 map-reduce로 요약한 것.
    """

    name: str
    summary: str
    name_embedding: tuple[float, ...]
    created_at: datetime
    uuid: str = field(default_factory=new_uuid)
