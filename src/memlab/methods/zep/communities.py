"""G_c 구축·유지 — G_s 위 community detection + map-reduce 요약 (Sec 2.3).

논문은 GraphRAG의 community 개념을 따르되 Leiden 대신 label propagation을
쓴다 — 동적 확장이 간단해서다. 생애주기는 원본 graphiti와 같은 2단이고
(2026-07-10 합의), 호출은 전부 method.py의 ingest 경로에서 일어난다:

    community detection  detect_communities() — 세션 경계마다 full rebuild.
                       전부 지운 뒤 전체 label propagation + packed map-reduce
                       (원본 build_communities와 동일한 삭제-후-재구축 semantics).
    dynamic extension  extend_community() — 발화가 건드린 entity마다.
                       무소속이면 이웃 다수결로 배정하고, 해당 community의
                       summary·name을 갱신 (원본 update_community 이식).
                       drift는 다음 rebuild가 수습한다 (논문의 "주기적
                       refresh"를 LoCoMo의 세션 주기에 맞춘 것). 단 마지막
                       세션은 경계가 없어 rebuild가 다시 안 온다 — 그 세션의
                       무소속 entity는 G_c 밖에 남는다 (facts·entities 검색은
                       무관; 스모크 오답 분석에서 영향 재평가, 2026-07-10).

논문이 침묵해 정한 것 (원본 graphiti community_operations.py 대조):
- 이웃 가중치 = entity pair 사이 RELATES_TO 엣지 개수 (원본 그대로).
  invalidated 엣지도 센다 — fact가 뒤집혀도 두 entity가 엮인 구조는 남는다.
- 원본 label propagation은 표준이 아니라 monotone 변형: 새 라벨 =
  max(가중 plurality 승자, 현재 라벨) (community_operations.py:110).
  라벨이 단조 증가라 수렴이 보장되고(표준 sync LPA는 진동 가능) 난수
  tie-break가 없어 deterministic. 그대로 차용하되, 초기 라벨을 uuid 정렬
  순으로 줘 실행 간 재현성을 마저 확보한다.
- map-reduce는 원본의 엄격한 pairwise(k−1콜) 대신 context 예산만큼 묶는
  packing (2026-07-10 합의) — 논문은 "iterative map-reduce-style"이라고만
  하고, 이 개념의 원류인 GraphRAG가 원래 token budget packing이다.
  세션 경계 rebuild를 감당하기 위한 비용 절감.
- name은 최종 summary에서 1회 생성·임베딩되어 cosine 검색 필드가 된다.
- 이웃 없는 entity도 1인 community가 된다 (원본 동일) — 요약 LLM 콜 0회,
  name 콜 1회.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from typing import Callable

from memlab.config import LLM_PARALLEL
from memlab.methods.zep.graph import ZepGraph
from memlab.methods.zep.llm_ops import CommunityOps
from memlab.methods.zep.schema import CommunityNode, EntityNode

# 요약 배치의 입력 상한 — 로컬 16K 토큰 context에서 프롬프트·출력 여유를
# 남긴 값 (~4자/토큰 추정). ponytail: 초과분은 다음 라운드로 접힐 뿐이라
# 값이 빡빡할 필요 없음.
PACK_BUDGET_CHARS = 16000


def label_propagation(neighbors: dict[str, dict[str, int]]) -> list[list[str]]:
    """가중 이웃 projection → cluster(uuid 리스트)들.

    neighbors는 전체 entity를 key로 가져야 한다 (이웃 없으면 빈 dict).
    """
    order = sorted(neighbors)
    # 1) 모든 노드가 자기만의 라벨로 시작 (n개 노드 = n개 라벨)
    community = {uuid: i for i, uuid in enumerate(order)}
    changed = True
    while changed:  # 3) 라벨이 안 바뀔 때까지 반복 → 수렴
        changed = False
        new_community: dict[str, int] = {}
        for uuid in order:
            # 2) 이웃 다수의 라벨로 교체 — 표의 무게 = 그 이웃과의 엣지 수
            votes: dict[int, int] = defaultdict(int)
            for neighbor, weight in neighbors[uuid].items():
                votes[community[neighbor]] += weight
            winner = max(
                votes, key=lambda label: (votes[label], label), default=community[uuid]
            )
            label = max(winner, community[uuid])  # 원본의 monotone 변형 — 라벨은 커지기만
            changed |= label != community[uuid]
            new_community[uuid] = label
        community = new_community
    # 4) 같은 라벨 = 하나의 community
    clusters: dict[int, list[str]] = defaultdict(list)
    for uuid in order:
        clusters[community[uuid]].append(uuid)
    return list(clusters.values())


def _fold(ops: CommunityOps, summaries: list[str]) -> str:
    """packed map-reduce — context 예산만큼 묶어 한 개가 될 때까지 접는다.

    배치는 예산과 무관하게 최소 2개를 집어 라운드마다 개수가 줄어드는
    것(수렴)을 보장한다. 홀로 남은 요약은 다음 라운드로 그냥 넘어간다
    (원본 pairwise의 odd_one_out과 같은 처리).
    """
    while len(summaries) > 1:
        batches: list[list[str]] = []
        batch: list[str] = []
        size = 0
        for summary in summaries:
            if len(batch) >= 2 and size + len(summary) > PACK_BUDGET_CHARS:
                batches.append(batch)
                batch, size = [], 0
            batch.append(summary)
            size += len(summary)
        batches.append(batch)
        summaries = [
            ops.summarize(b) if len(b) > 1 else b[0] for b in batches
        ]
    return summaries[0]


def detect_communities(
    graph: ZepGraph,
    ops: CommunityOps,
    embed: Callable[[str], Sequence[float]],
    created_at: datetime,
) -> list[CommunityNode]:
    """community detection (Sec 2.3) — G_c를 전부 지우고 다시 구축.

    cluster별 요약·name 생성은 fan-out (LLM 콜만 병렬 — method.py의
    fan-out과 같은 근거), 임베딩·저장은 메인 스레드 순차.
    """
    graph.community.remove_all()
    summaries = graph.semantic.entity_summaries()
    neighbors: dict[str, dict[str, int]] = {uuid: {} for uuid in summaries}
    neighbors.update(graph.semantic.neighbor_counts())

    def summarize_cluster(cluster: list[str]):
        summary = _fold(ops, [summaries[uuid] for uuid in cluster])
        return cluster, summary, ops.name(summary)

    with ThreadPoolExecutor(LLM_PARALLEL) as pool:
        built = list(pool.map(summarize_cluster, label_propagation(neighbors)))

    communities = []
    for cluster, summary, name in built:
        node = CommunityNode(
            name=name,
            summary=summary,
            name_embedding=tuple(float(x) for x in embed(name)),
            created_at=created_at,
        )
        graph.community.save(node)
        for uuid in cluster:
            graph.community.link_member(node.uuid, uuid)
        communities.append(node)
    return communities


def extend_community(
    graph: ZepGraph,
    ops: CommunityOps,
    embed: Callable[[str], Sequence[float]],
    entity: EntityNode,
) -> None:
    """dynamic extension (Sec 2.3) — 원본 update_community 이식 (entity당 LLM 2콜).

    소속이 있으면 그 community를, 없으면 이웃 다수결(행 수 = 가중치)로
    배정한 community를 entity summary와 합쳐 갱신한다. 이웃도 전부
    무소속이면 아무것도 안 한다 — 다음 full rebuild가 수습 (원본 동일).
    호출은 순차 전제 — 두 entity가 같은 community를 동시에 갱신하면
    한쪽 갱신이 사라진다 (원본은 동시 실행으로 이를 감수하지만, 우리
    규모에선 순차 비용이 작아 fan-out하지 않는다).
    """
    member = graph.community.member_of(entity.uuid)  # 기존 소속 (없으면 None)
    if member is not None:
        community = member
    else:
        # 이웃 다수가 속한 community로 배정 — 논문의 "라벨 전파" 한 걸음
        neighbors = graph.community.neighbor_communities(entity.uuid)
        if not neighbors:
            return
        winner = Counter(c.uuid for c in neighbors).most_common(1)[0][0]
        community = next(c for c in neighbors if c.uuid == winner)
    # 편입한 entity의 정보로 community summary·name을 갱신
    summary = ops.summarize([entity.summary, community.summary])
    name = ops.name(summary)
    graph.community.save(
        replace(
            community,
            name=name,
            summary=summary,
            name_embedding=tuple(float(x) for x in embed(name)),
        )
    )
    if member is None:  # 새로 편입한 경우에만 소속 엣지 생성
        graph.community.link_member(community.uuid, entity.uuid)
