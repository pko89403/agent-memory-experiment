"""Zep 메소드 조립 — 소켓(ingest/answer)에 그래프·LLM ops·retrieval 연결.

ingest 파이프라인 (발화 = episode 하나, Sec 2.2):

    세션 경계?      → G_c full rebuild (주기적 refresh, Sec 2.3)
    entity          추출 → resolution → 병합/신규 저장 (Sec 2.2.1)
    fact            추출 → dedup → temporal → invalidation (Sec 2.2.2~2.2.3)
    episode         semantic_edge_uuids 확정 후 저장 + MENTIONS 연결 (Sec 2.1)
    community       건드린 entity마다 dynamic extension (Sec 2.3)

answer는 READ 전용: MemoryRetrieval.retrieve(question) → AnswerGenerator.
WRITE는 전부 ingest 경로에 있다 (2026-07-10 합의 — finalize 훅 없이).

논문이 침묵해 정한 것 (원본 graphiti 대조):
- created_at(T')은 datetime.now(UTC) — bi-temporal의 ingestion 시각은
  실제 벽시계가 맞고, recent()의 "last n messages" 정렬 근거다 (graph.py).
  코드베이스에서 유일하게 시계를 읽는 곳.
- t_ref는 LoCoMo date_time("1:56 pm on 8 May, 2023")을 파싱 — 형식이
  다르면 조용한 fallback 없이 ValueError로 죽인다 (데이터 결함은 드러낼 것).
- blip_caption은 "(image description: ...)"로 본문에 덧붙인다 —
  MemoryOS 재구현과 동일 조건 (메소드 간 비교 공정성).
- entity 병합: 원본대로 이름은 resolution의 best name, summary는 신·구
  결합 LLM 1콜 (node_operations.py:290-307). 이름이 바뀌면 재임베딩.
- fact가 dedup돼도 temporal 추출은 수행해 기존 엣지의 빈 날짜를 채운다 —
  원본 resolve_extracted_edge가 dedup·dates·contradictions를 전부 돌리고
  날짜를 덮어쓴다 (edge_operations.py:272-281). 예외는 fast path의 자구
  동일 재언급 (_ingest_facts judge 주석).
- invalidation은 2단: LLM(resolve_fact의 병합 판정)이 모순 후보를 고르고,
  원본의 시간 필터(edge_operations.py:233-296)를 적용한다 — 유효 구간이
  겹치지 않으면 무시, 후보가 더 과거면 후보를 invalidate, 후보가 더
  최신이면 새 엣지 쪽을 invalidate ("expire new edge"). 원본의 tzinfo
  요구는 제외 — parse_iso가 naive로 통일한다.
- 후보 풀: resolution·dedup은 원본 RELEVANT_SCHEMA_LIMIT(10),
  invalidation은 endpoint당 최신순 10 (graph.py edges_touching 결정 참고).
- fact triple의 entity 이름이 확정 명단에 없으면 그 triple은 버린다 —
  원본도 이름→노드 매핑 실패 엣지를 버린다 (edge_operations.py 유사).
- 세션 경계 감지 = utterance.timestamp 문자열 변화 (MemoryOS와 동일 패턴).
  첫 발화 전에는 경계가 아니므로 첫 rebuild는 두 번째 세션 시작 시 —
  단일 세션 대화라면 G_c가 비는데, LoCoMo는 전 대화가 다중 세션이다.
- entity·fact 판정은 ThreadPoolExecutor(LLM_PARALLEL) fan-out — 원본의
  semaphore_gather 상응 (graphiti도 node resolution·edge resolution을
  동시 실행). LLM·그래프 READ만 병렬이고, 임베딩(SentenceTransformer
  스레드 안전성 미보장)과 저장은 메인 스레드 순차. LLM_PARALLEL의 근거는
  config.py (LM Studio --parallel 4 실측 포화점, 2026-07-10 — 4.16x).
  fact judge의 내부 2-way와 겹치면 동시 요청이 최대 8까지 가는데, 서버가
  4로 자르므로 처리량은 동일하고 초과분은 대기만 한다 (검증 리뷰 M1).
- 같은 발화의 이름 변형 여럿이 한 노드로 병합되면 마지막 병합만 남는다
  (last-write-wins) — 원본의 동시 resolution + bulk save와 동일 semantics
  (검증 리뷰 P3-7). 같은 이유로 한 발화의 triple 여럿이 같은 기존 엣지로
  dedup되면 episode.semantic_edge_uuids에 그 uuid가 중복 기재된다 (M4,
  retrieval 무해).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from memlab.config import LLM_PARALLEL
from memlab.embedding import embed as default_embed
from memlab.llm import LLMProvider
from memlab.methods.base import MemoryMethod, Utterance
from memlab.methods.zep.communities import detect_communities, extend_community
from memlab.methods.zep.graph import ZepGraph
from memlab.methods.zep.llm_ops import (
    AnswerGenerator,
    CommunityOps,
    ExtractionOps,
)
from memlab.methods.zep.schema import EntityNode, EpisodeNode, SemanticEdge
from memlab.methods.zep.search import MemoryRetrieval

T_REF_FORMAT = "%I:%M %p on %d %B, %Y"  # LoCoMo date_time 원문 형식


@dataclass(frozen=True)
class ZepConfig:
    """하이퍼파라미터 — 기본값의 출처는 각 주석. runs/ meta.json에 직렬화."""

    recent_context: int = 4  # entity·fact 추출 맥락 "last n messages" (Sec 2.2.1)
    # False면 ingest가 G_c를 아예 안 만든다 (세션 경계 rebuild + dynamic
    # extension 모두 생략, retrieval의 COMMUNITIES는 빈 채로). 원본 graphiti도
    # add_episode(update_communities=False)가 기본값이고 G_c는 on-demand다
    # (graphiti.py:262,597). LoCoMo 전량 런은 False로 간다 — steady state
    # 발화당 entity×2콜 + 세션 경계마다 entity 수 규모의 rebuild가 전체
    # 비용의 ~1/3인 것을 실측 후 결정 (2026-07-11 합의).
    update_communities: bool = True
    candidate_limit: int = 10  # resolution 후보 상한 — 원본 RELEVANT_SCHEMA_LIMIT
    invalidation_limit: int = 10  # invalidation 풀: endpoint당 최신순 상한 (동일 값)
    edge_limit: int = 10  # retrieval limit 3종 — DMR 실험 config (Sec 4.1)
    entity_limit: int = 10
    community_limit: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


class ZepMethod(MemoryMethod):
    def __init__(
        self,
        llm: LLMProvider,
        speaker_a: str,
        speaker_b: str,
        embed: Callable[[str], np.ndarray] = default_embed,
        config: ZepConfig = ZepConfig(),
    ):
        self.speaker_a = speaker_a
        self.speaker_b = speaker_b
        self.config = config
        self._embed = embed
        self.graph = ZepGraph()  # 생성 = reset(전체 삭제 + 인덱스) — 실험 격리: 대화 하나 = 그래프 하나
        self._extract = ExtractionOps(llm)
        self._community_ops = CommunityOps(llm)
        self._answerer = AnswerGenerator(llm)
        self.retrieval = MemoryRetrieval(
            self.graph.run,
            embed,
            edge_limit=config.edge_limit,
            entity_limit=config.entity_limit,
            community_limit=config.community_limit,
        )
        self._last_timestamp: str | None = None  # 세션 경계 감지용

    # ── ingest: 발화 → episode → G_s → G_c (Sec 2.2~2.3, 전부 WRITE) ──

    def ingest(self, utterance: Utterance) -> None:
        # 세션 경계 → community detection 전체 재실행 (논문의 "주기적 refresh", Sec 2.3)
        if (
            self.config.update_communities
            and self._last_timestamp is not None
            and utterance.timestamp != self._last_timestamp
        ):
            detect_communities(self.graph, self._community_ops, self._embed, _now())
        self._last_timestamp = utterance.timestamp

        text = utterance.text
        if utterance.blip_caption:
            text = f"{text} (image description: {utterance.blip_caption})"
        # bi-temporal의 두 시간축이 여기서 갈라진다 (Sec 2.1):
        #   t_ref = 발화가 속한 현실 시간(T)  /  created_at = ingestion 시각(T')
        episode = EpisodeNode(
            content=text,
            speaker=utterance.speaker,
            t_ref=datetime.strptime(utterance.timestamp, T_REF_FORMAT),
            created_at=_now(),
        )
        # 추출 맥락 = 현재 message + 직전 n=4 messages (Sec 2.2.1)
        previous = self.graph.episode.recent(self.config.recent_context)

        entities = self._ingest_entities(episode, previous)  # G_s 노드 (Sec 2.2.1)
        edge_uuids = self._ingest_facts(episode, previous, entities)  # E_s (Sec 2.2.2~3)

        # G_e 완성: 원문 비손실 저장 + episode↔fact 양방향 인덱스 (Sec 2.1)
        self.graph.episode.save(replace(episode, semantic_edge_uuids=tuple(edge_uuids)))
        for node in entities:
            self.graph.episode.link_mentions(episode.uuid, node.uuid)  # E_e: MENTIONS
            if self.config.update_communities:
                # dynamic extension — 새 entity를 이웃 다수결로 community에 편입 (Sec 2.3)
                extend_community(self.graph, self._community_ops, self._embed, node)

    def _ingest_entities(
        self, episode: EpisodeNode, previous: list[EpisodeNode]
    ) -> list[EntityNode]:
        """Sec 2.2.1 — 추출·resolution·병합을 거친 이번 episode의 entity들.

        이름 하나마다 세 작업이 있다:
          ① summary  — "이번 대화에서 이놈에 대해 알게 된 것" 한 문단.
                       그래프를 보지 않고 대화(n=4 창)만 본다.
          ② resolve  — "그래프에 이미 있는 놈인가?" 판정. 후보 검색 +
                       LLM 판정 (아래 _resolve_entity_name).
          ③ merge    — ②가 duplicate일 때만: ①의 새 요약과 기존 노드에
                       '저장된' 과거 요약을 결합해 그 노드에 덮어쓴다.
                       노드는 안 늘고 아는 것만 는다.

        ①과 ②는 서로의 출력을 안 쓰므로(②는 새 summary를 보지 않는다 —
        과병합 교정 경위는 llm_ops.resolve_entity) 동시 실행이 합법이고,
        ③이 둘을 기다리는 유일한 합류 지점이다. fan-out 구간은 판정만
        하고 그래프에 아무것도 안 쓴다 — 임베딩은 메인 스레드, 저장은
        판정이 다 끝난 뒤 순차.
        """
        # entity extraction + reflexion("놓친 이름?" 재확인) — 프롬프트 6.1.1
        names = self._extract.extract_entities(episode, previous)
        vecs = {name: _floats(self._embed(name)) for name in names}  # name 임베딩 = cosine 검색 필드 (차원 이탈은 schema.py)

        with ThreadPoolExecutor(LLM_PARALLEL) as pool:
            # 원본처럼 summary와 resolution은 서로 독립 — 동시 실행
            # (node_operations.py:277의 semaphore_gather와 동일 구도)
            summary_futures = {
                n: pool.submit(self._extract.summarize_entity, n, episode, previous)
                for n in names
            }
            resolve_futures = {
                n: pool.submit(self._resolve_entity_name, n, vecs[n], episode, previous)
                for n in names
            }
            summaries = {n: f.result() for n, f in summary_futures.items()}
            resolutions = {n: f.result() for n, f in resolve_futures.items()}
            # ③ merge — ①(새 요약)과 ②(판정)가 만나는 유일한 합류 지점.
            #    match.summary는 과거 발화들이 쌓아 그래프에 저장돼 있던 요약
            merge_futures = {
                n: pool.submit(
                    self._community_ops.summarize, [summaries[n], match.summary]
                )
                for n, (match, _) in resolutions.items()
                if match is not None
            }
            merged = {n: f.result() for n, f in merge_futures.items()}

        resolved: dict[str, EntityNode] = {}  # uuid → node (이름 변형 중복 제거)
        for name in names:
            match, best_name = resolutions[name]
            if match is None:
                # resolution 판정 "신규" → 새 노드 (Sec 2.2.1)
                node = EntityNode(
                    name=best_name, summary=summaries[name],
                    name_embedding=vecs[name], created_at=_now(),
                )
            else:
                # resolution 판정 "같은 놈" → 기존 노드에 병합: best name + summary 결합
                node = replace(
                    match,
                    name=best_name,
                    summary=merged[name],
                    name_embedding=match.name_embedding if best_name == match.name
                    else _floats(self._embed(best_name)),
                )
            self.graph.semantic.save_entity(node)
            resolved[node.uuid] = node
        return list(resolved.values())

    def _resolve_entity_name(
        self, name: str, vec: tuple[float, ...],
        episode: EpisodeNode, previous: list[EpisodeNode],
    ) -> tuple[EntityNode | None, str]:
        """② resolve = 후보 검색(그물) + LLM 판정, 두 단계.

        그래프 전체를 판정 프롬프트에 넣을 수 없으니 "그럴듯한 것 몇 개"를
        먼저 검색한다. 질의는 새 이름 하나뿐 — 이번 발화에서 생성 중인
        새 summary는 여기 안 쓰인다 (그래서 ①과 동시 실행 가능).
        """
        candidates = self.graph.semantic.entity_candidates(
            vec, name, self.config.candidate_limit
        )
        return self._extract.resolve_entity(name, candidates, episode, previous)

    def _ingest_facts(
        self,
        episode: EpisodeNode,
        previous: list[EpisodeNode],
        entities: list[EntityNode],
    ) -> list[str]:
        """Sec 2.2.2~2.2.3 — fact 추출·dedup·temporal·invalidation.

        triple = (주어 entity, 술어, 목적어 entity) + 원문 fact 문장 —
        지식 그래프의 최소 문장 단위. entity가 "누가 있는지"라면
        fact/triple은 "그들 사이에 무슨 일이 있는지"고, 시간축(valid/
        invalid)이 붙는 쪽은 이쪽이다.

        triple 하나마다 서로 독립인 판정 2개를 동시에 던진다:
          resolve       dedup(중복인가) + invalidation 선별(모순 용의자
                        지목)을 LLM 1콜로 — 판결은 _invalidate_edges의
                        날짜 논리 (병합 경위는 llm_ops.resolve_fact)
          temporal      시작/끝 날짜 — "yesterday"를 t_ref로 절대화 (6.1.5)
        그 전에 fast path: 같은 방향 endpoint에 fact 원문이 자구 동일한
        기존 엣지가 있으면 LLM 없이 dedup 확정 (upstream HEAD 526dcad7
        edge_operations.py:684 차용, 2026-07-11).
        entity 쪽과 같은 구도: 판정은 병렬, 저장은 판정이 다 끝난 뒤 순차.
        같은 episode의 triple끼리는 서로의 결과를 못 본다 (원본도 전
        엣지를 동시에 처리하므로 동일 semantics, edge_operations.py:195).
        """
        by_name = {node.name.casefold(): node for node in entities}
        triples = [
            (t, by_name[t.source_entity_name.casefold()], by_name[t.target_entity_name.casefold()])
            for t in self._extract.extract_facts(
                episode, previous, [node.name for node in entities]
            )
            if t.source_entity_name.casefold() in by_name
            and t.target_entity_name.casefold() in by_name
        ]
        vecs = [_floats(self._embed(t.fact)) for t, _, _ in triples]

        def judge(item):
            (triple, source, target), vec = item
            # fact를 semantic edge로 실체화 — 다중 entity fact는 pair마다
            # 별도 엣지가 되고 같은 fact 문장을 나눠 갖는다 (hyper-edge, Sec 2.2.2)
            extracted = SemanticEdge(
                source_uuid=source.uuid,
                target_uuid=target.uuid,
                relation=triple.relation_type,
                fact=triple.fact,
                fact_embedding=vec,
                episode_uuids=(episode.uuid,),
                created_at=_now(),
            )
            candidates = self.graph.semantic.edges_between(
                source.uuid, target.uuid, self.config.candidate_limit
            )
            # fast path — 같은 방향 endpoint + fact 원문 자구 동일이면 LLM
            # 없이 dedup 확정. 자구 동일한 재언급엔 새 날짜 정보가 없으므로
            # temporal도 생략 — "dedup돼도 temporal은 수행" 결정(모듈
            # docstring)의 유일한 예외 (원본도 fast path에선 전부 생략)
            for cand in candidates:
                if (
                    cand.source_uuid == source.uuid
                    and cand.target_uuid == target.uuid
                    and _normalized(cand.fact) == _normalized(extracted.fact)
                ):
                    return (
                        replace(cand, episode_uuids=(*cand.episode_uuids, episode.uuid)),
                        [],
                    )
            pool = self._invalidation_pool(source.uuid, target.uuid, extracted.uuid)
            # 원본처럼 판정과 temporal을 동시 실행 — 선별이 새 엣지의
            # 날짜를 못 보는 것도 원본과 동일 (edge_operations.py:272)
            with ThreadPoolExecutor(2) as inner:
                resolve_f = inner.submit(
                    self._extract.resolve_fact, extracted, candidates, pool
                )
                dates_f = inner.submit(
                    self._extract.extract_temporal, extracted.fact, episode,
                    previous, episode.t_ref,
                )
                duplicate, contradicted = resolve_f.result()
                valid_at, invalid_at = dates_f.result()

            # base = dedup 판정 확정본 — 중복이면 기존 엣지 (출처 episode 누적)
            base = (
                replace(duplicate, episode_uuids=(*duplicate.episode_uuids, episode.uuid))
                if duplicate is not None
                else extracted
            )
            # resolved = temporal 스탬프까지 얹은 최종본 — '시작'(t_valid)은
            # t_ref 기준 절대화, '끝'(t_invalid)은 본문이 종료를 말했을 때만 (Sec 2.2.3)
            resolved = replace(
                base,
                valid_at=valid_at or base.valid_at,
                invalid_at=invalid_at or base.invalid_at,
                expired_at=_now() if invalid_at and not base.expired_at else base.expired_at,
            )
            # dedup을 거치면 resolved.uuid는 기존 엣지의 것 — pool에 자기
            # 자신이 남아 자기 무효화가 가능해 사후 배제 (동시 실행의 대가)
            contradicted = [c for c in contradicted if c.uuid != resolved.uuid]
            return _invalidate_edges(resolved, contradicted)

        with ThreadPoolExecutor(LLM_PARALLEL) as pool:
            results = list(pool.map(judge, zip(triples, vecs)))

        edge_uuids: list[str] = []
        for edge, losers in results:
            self.graph.semantic.save_edge(edge)
            for loser in losers:
                self.graph.semantic.save_edge(loser)
            edge_uuids.append(edge.uuid)
        return edge_uuids

    def _invalidation_pool(
        self, source_uuid: str, target_uuid: str, exclude_uuid: str
    ) -> list[SemanticEdge]:
        """모순 용의자 풀 = 새 fact와 endpoint(양쪽 entity)를 공유하는 엣지들.

        모순은 등장인물이 겹치는 fact 사이에서만 성립한다 — "Alice가 Rex를
        보냈다"가 "Bob은 파리에 산다"와 모순일 수 없다. 그래서 풀을 새
        fact의 두 끝점(source·target) 중 하나라도 건드리는 엣지로 좁힌다
        (Sec 2.2.3; 전역 유사도 검색은 무관 entity의 유사 문장을 끌어와
        오판만 키운다 — graph.py 결정). 방향 무시, 끝점당 최신순
        invalidation_limit개, 자기 자신 제외.

        invalidation 전체는 깔때기다: 그래프 전체 → 끝점 공유(여기)
        → LLM 용의자 지목(resolve_fact의 병합 판정) → 날짜
        판결(_invalidate_edges).
        """
        pool = {
            e.uuid: e
            for uuid in (source_uuid, target_uuid)
            for e in self.graph.semantic.edges_touching(
                uuid, self.config.invalidation_limit
            )
        }
        pool.pop(exclude_uuid, None)
        return list(pool.values())

    # ── answer: READ 전용 (Sec 3 → Sec 4의 chat agent) ────────────────

    def answer(self, question: str) -> str:
        context = self.retrieval.retrieve(question)  # β = χ(ρ(φ(α))) — LLM 호출 없음 (Sec 3)
        return self._answerer.generate(  # β + 질문 → chat agent 답변 (Sec 4)
            context, question, self.speaker_a, self.speaker_b
        )


def _invalidate_edges(
    edge: SemanticEdge, candidates: list[SemanticEdge]
) -> tuple[SemanticEdge, list[SemanticEdge]]:
    """edge invalidation의 판결·집행 — (새 엣지, invalidate된 기존 엣지들).

    resolve_fact(LLM)가 "내용상 모순"인 용의자를 지목하면,
    여기서 날짜 논리로 판결한다 (원본 edge_operations.py:233-296 이식).
    LLM은 모순인지는 알지만 누가 이기는지는 모른다 — 승패와 invalid_at
    값은 valid_at 비교라는 결정론 산수고, 소형 모델에 맡기면 틀린다.

    판결 세 갈래 (용의자마다):
      유효 구간 안 겹침         → 무죄 — 모순이 아니라 순차 사실 (지목 기각)
      용의자가 새 fact보다 과거 → 용의자 invalid_at = 새 fact의 valid_at
      용의자가 새 fact보다 최신 → 반전: 새 엣지가 태어나자마자 invalid
                                  (회상 발화 케이스, "expire new edge")
    """
    now = _now()
    # 1) 더 최신 fact가 이미 있으면 — 새 엣지 쪽이 태어나자마자 invalid ("expire new edge")
    if edge.expired_at is None:
        for candidate in sorted(
            candidates, key=lambda c: (c.valid_at is None, c.valid_at or datetime.min)
        ):
            if candidate.valid_at and edge.valid_at and candidate.valid_at > edge.valid_at:
                edge = replace(edge, invalid_at=candidate.valid_at, expired_at=now)
                break

    # 2) 새 fact보다 과거인 모순 후보들 — invalid_at에 새 fact의 valid_at을 찍는다.
    #    삭제가 아니다: 엣지는 남고 유효 구간만 닫힌다 (Sec 2.2.3의 요점)
    losers = []
    for candidate in candidates:
        disjoint = (
            candidate.invalid_at and edge.valid_at
            and candidate.invalid_at <= edge.valid_at
        ) or (
            candidate.valid_at and edge.invalid_at
            and edge.invalid_at <= candidate.valid_at
        )
        if disjoint:
            continue
        if candidate.valid_at and edge.valid_at and candidate.valid_at < edge.valid_at:
            losers.append(
                replace(
                    candidate,
                    invalid_at=edge.valid_at,
                    expired_at=candidate.expired_at or now,
                )
            )
    return edge, losers


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalized(fact: str) -> str:
    # fast path의 "자구 동일" — 공백 접기 + casefold (원본 _normalize_string_exact 상응)
    return " ".join(fact.split()).casefold()


def _floats(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(float(x) for x in vector)
