"""Nemori 메모리의 스키마 — 논문 Sec 3의 형식화를 코드로 옮긴 것.

논문 형식화와 대응 (arXiv 2508.03341v4):

    논문 (Sec 3)                              여기
    ─────────────────────────────────────    ──────────────────
    M_j = (c_j, N_j, P_j, v_j)  (§3.2.2)     EpisodicMemory
      c_j  episodic cue                        .cue
      N_j  narrative episode                   .narrative
      P_j  raw episode (메시지 부분수열)         .raw
      v_j = f_emb(c_j ∥ N_j)                   .embedding
    k_q ∈ K_in, u_q = f_emb(k_q)  (§3.3.3)   SemanticInsight

raw episode P는 Utterance 그대로 보관한다 — §3.2.2 Discussion의
dual-mode retrieval(서사는 효율, 원문은 정밀) 근거이고, 답변 시
상위 r=2 episode에 원문을 첨부하는 프로토콜(§4.1)이 소비한다.
Associative Memory Integration(§3.2.3)의 병합도 P_k ∥ P_j 이어붙이기.

논문이 침묵해 정한 것:
- occurred_at(episode 시각)은 형식화 M=(c,N,P,v)에 없지만 Pnar 프롬프트가
  추출을 지시하고(부록 D.1.2), Psel의 병합 금지 기준(">1시간 갭", D.1.3)과
  답변 컨텍스트의 "- [timestamp] narrative" 라인이 소비한다 — 프롬프트가
  요구하는 실질 필드라 M에 포함한다. 논문에 없는 발명 필드라 이름도 논문
  기호가 아니라 프롬프트 자구("when this episode occurred")에서 땄다.
  Utterance.timestamp(세션 date_time 문자열)와의 타입 혼동도 피한다.
- occurred_at 파싱 실패 시 fallback은 윈도우 첫 발화의 세션 timestamp. 원본 코드는
  벽시계(datetime.now())인데, 2023년 대화에 2026년 앵커가 찍혀 temporal
  답변과 병합 판정을 오염시키고 run 결정성도 깨므로 배제 (2026-07-17 합의).
- embedding은 하네스 공용 all-MiniLM-L6-v2 384-dim — 논문은
  text-embedding-3-small(1536-dim, §4.1)이지만 메소드 간(MemoryOS/Zep)
  비교 조건을 맞추는 게 재현보다 우선 (zep/schema.py와 동일 결정).
- source_episode_uuid는 논문 비형식화 — 원본 코드(SemanticMemory
  .source_episode_id) 차용. 가이드 노트북에서 insight의 출처 episode를
  추적하는 용도뿐이다.
- 원본 코드의 confidence(항상 1.0 미사용)·memory_type(키워드 분류기,
  논문에 없음)은 제외.
- partition이 만드는 topic(원본의 boundary_reason)은 Pnar 입력으로만
  쓰고 버린다 — cue가 검색·예측용 표제 역할을 대체하므로 저장 안 함.

생성은 전부 원자적이다 — zep의 episode↔edge 상호 참조 같은
"만들고 나중에 채우기"가 없다. Pnar 1콜이 (cue, narrative, occurred_at)를 한꺼번에
반환하고 embedding은 그 직후 계산되므로 EpisodicMemory는 완성 상태로만
태어나며, 병합은 수정이 아니라 삭제 후 재생성(§3.2.3 "superseding"),
insight는 integrate 확정 이후 distill이라 항상 최종 episode를 참조한다.
단 하나의 느슨한 끝: 이미 저장된 insight의 source_episode_uuid는 그 출처
episode가 **나중** 병합으로 삭제되면 dangling이 된다 — 원본 구현과 동일
동작이고 검색에 안 쓰여 무해하나, 노트북 추적 시 유의.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from memlab.methods.base import Utterance


def new_uuid() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class EpisodicMemory:
    """M = (c, N, P, v) + occurred_at — episodic DB D_e의 엔트리 (§3.2.2).

    병합(§3.2.3) 시 이 객체는 삭제되고 (c_ν, N_ν, P_k ∥ P_j) 재생성으로
    대체된다 — frozen인 이유. 진실은 항상 method가 든 D_e 리스트 쪽이다.
    """

    cue: str  # c — 검색·예측 단서가 되는 한 줄 표제 (10-20 단어)
    narrative: str  # N — 3인칭 서사 (절대 날짜 병기, Pnar 요건)
    raw: tuple[Utterance, ...]  # P — 원문 발화 부분수열 (non-lossy)
    embedding: tuple[float, ...]  # v = f_emb(c ∥ N)
    occurred_at: datetime  # episode 발생 시각 (Pnar 추출, 실패 시 세션 timestamp)
    uuid: str = field(default_factory=new_uuid)


@dataclass(frozen=True)
class SemanticInsight:
    """k ∈ K — semantic DB D_s의 엔트리 (§3.3.2-3.3.3).

    prediction error에서 distill된 자립적(atomic·self-contained) fact 진술.
    consolidation은 append뿐이다 — 논문 §3.3.3의 new/merge/conflict 판정은
    ablation(Table 5)에서 naive append 대비 ~0.4pt 차이로 실질 가치가 없고
    (LoCoMo에 knowledge update가 드묾), 공개 코드도 미구현이라 제외
    (2026-07-17 합의). forgetting 변형 실험 시 개입 지점으로 재고.
    """

    statement: str  # k — fact 진술 (현재시제, 문맥 독립; Pant의 "Knowledge Statements")
    embedding: tuple[float, ...]  # u = f_emb(k)
    source_episode_uuid: str  # distill 출처 episode (provenance, 노트북 추적용)
    uuid: str = field(default_factory=new_uuid)
