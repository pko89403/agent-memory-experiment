"""Nemori 메소드 조립 — 소켓(ingest/answer)에 두 cascading module을 연결.

ingest 파이프라인 (Algorithm 1) — buffer가 w에 차면:

    partition (§3.2.1)
    그룹마다:
      _episodic_integration (§3.2.2-3.2.3):
          narrate → f_emb(c ∥ N) → Ke 후보 → select_target
          → merge면 integrate + P_k ∥ P_j + supersede, 아니면 add
      _semantic_distillation (§3.3):          ← cascade: 확정된 episode만
          evoke(τ, Ks) → distill (비면 direct_distill)
          → statement마다 f_emb(k) → consolidate
    buffer reset

같은 윈도우 안에서도 앞 episode의 distillation이 끝난 뒤 다음 episode가
시작된다 — 뒤 episode의 Evoke가 방금 distill된 지식을 본다 (Algorithm 1의
루프 순서 그대로. 원본도 동일하나 async buffer 경쟁에 가려져 있었다).

answer (Algorithm 2)는 READ 전용: embed(Q) → episodic top-k +
semantic top-m → Pans (상위 r은 원문 첨부).

논문이 침묵해 정한 것:
- 잔여(<w) 발화의 flush는 end_ingest() — 러너가 ingest 루프 직후 부르는
  소켓 훅 (base.py 검증 리뷰 N1 참고). 처음엔 첫 answer() 직전 flush로
  설계했으나, flush 실패가 QA 단위 에러 격리에 삼켜져 all-error
  checkpoint로 굳는 결함이 리뷰에서 확정되어 훅으로 옮겼다 (2026-07-17).
  answer()는 READ 전용으로 복귀.
- cold start: evoke가 비면 direct_distill(D.2)로 분기 — 논문 Algorithm 1은
  항상 predict지만 빈 지식의 예측은 무의미한 2배 비용. 원본 배선 차용
  (semantic.py:51-54, 2026-07-17 합의).
- occurred_at: narrate의 timestamp 파싱 실패 시 윈도우 첫 발화의 세션
  timestamp (schema.py 결정 — 벽시계 금지). 세션 timestamp 파싱 실패는
  ValueError로 죽인다 — 데이터 결함은 드러낸다 (형식 상수는 데이터셋
  속성이라 memlab.data.DATE_TIME_FORMAT, 검증 리뷰 N6).
- f_emb(c ∥ N)의 ∥는 원본(episode.py:60)대로 공백 결합 "{cue} {narrative}".
- 병합 raw는 target.raw + new.raw — P_k ∥ P_j 표기 순서 (원본 merger.py:133).
- select_target은 후보가 있을 때만 호출 — 빈 후보 나열은 낭비 콜
  (원본 check_and_merge도 후보 없으면 조기 반환).
- speaker_a/b를 받지 않는다 — nemori는 화자 접기(fold)가 없고 Utterance의
  speaker가 그대로 프롬프트에 실린다 (원본 eval도 한 대화 = 한 메모리 공간,
  role = 화자명).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from memlab.data import DATE_TIME_FORMAT
from memlab.embedding import embed as default_embed
from memlab.llm import LLMProvider
from memlab.methods.base import MemoryMethod, Utterance
from memlab.methods.nemori.llm_ops import EpisodicOps, SemanticOps, generate_answer
from memlab.methods.nemori.memory import EpisodicStore, SemanticStore
from memlab.methods.nemori.schema import EpisodicMemory, SemanticInsight


@dataclass(frozen=True)
class NemoriConfig:
    """하이퍼파라미터 — 전부 논문 §4.1 Implementation Details의 기호·값.

    consolidation 후보 수 K_m은 없다 — append-only 결정(schema.py)으로
    소비처가 없는 값이라 만들지 않는다.
    """

    w: int = 20  # observation window (§3.2.1; Table 11: 5~40 전 구간 안정)
    # Evoke threshold τ (§3.3.1). 논문 값 0.70은 text-embedding-3-small 공간
    # 기준 — all-MiniLM 384-dim 실측(검증 리뷰 N2, 2026-07-17: 관련 fact
    # 0.545~0.892, 무관 ≤0.505)에서 0.70은 어휘가 갈리는 관련 fact를 버려
    # 후속 episode의 ~1/3이 direct_distill로 강등된다. 관련 전부 유지·무관
    # 전부 차단하는 실측 경계값 0.55로 재캘리브레이션 (임베딩 교체와 같은
    # 등급의 공간 적응 — schema.py embedding 결정 참고).
    tau: float = 0.55
    ke: int = 5  # 병합 후보 수 K_e (§3.2.3)
    ks: int = 10  # Evoke 회수량 K_s (§3.3.1)
    k: int = 10  # 답변 시 episodic top-k (§3.4)
    m: int = 20  # 답변 시 semantic top-m (§3.4; m = 2k)
    r: int = 2  # 원문을 첨부하는 상위 episode 수 (§4.1)


class NemoriMethod(MemoryMethod):
    def __init__(
        self,
        llm: LLMProvider,
        embed: Callable[[str], np.ndarray] = default_embed,
        config: NemoriConfig = NemoriConfig(),
    ):
        self._config = config
        self._embed = embed
        self._llm = llm
        self._episodic_ops = EpisodicOps(llm)
        self._semantic_ops = SemanticOps(llm)
        self._d_e = EpisodicStore()  # D_e
        self._d_s = SemanticStore()  # D_s
        self._buffer: list[Utterance] = []  # B (§3.2.1)

    def ingest(self, utterance: Utterance) -> None:
        self._buffer.append(utterance)
        if len(self._buffer) >= self._config.w:  # |B| = w 트리거
            self._process_window(tuple(self._buffer))
            self._buffer.clear()  # 논문: buffer reset

    def end_ingest(self) -> None:
        """잔여(<w) 발화 flush — 러너가 ingest 루프 직후 호출 (검증 리뷰 N1)."""
        if self._buffer:
            self._process_window(tuple(self._buffer))
            self._buffer.clear()

    def answer(self, question: str) -> str:
        query = self._embed(question)
        return generate_answer(
            self._llm, question,
            self._d_e.search(query, self._config.k),
            self._d_s.search(query, self._config.m),
            include_raw_top=self._config.r,
        )

    # ── Algorithm 1 ──────────────────────────────────────────────────

    def _process_window(self, window: tuple[Utterance, ...]) -> None:
        # 잔여(<w) 윈도우는 partition 없이 통짜 한 그룹 — 원본의 batch_threshold
        # 게이트(memory_system.py:106-113) 재현이고, 발화 1개짜리 flush에 유일한
        # 합법 출력을 물으러 LLM 콜을 태우는 낭비도 막는다 (검증 리뷰 N12)
        if len(window) < self._config.w:
            groups = [(window, "conversation")]
        else:
            groups = self._episodic_ops.partition(window)
        for raw, topic in groups:
            episode = self._episodic_integration(raw, topic)  # §3.2
            self._semantic_distillation(episode)  # §3.3 — cascade

    def _episodic_integration(
        self, raw: tuple[Utterance, ...], topic: str
    ) -> EpisodicMemory:
        cue, narrative, t = self._episodic_ops.narrate(raw, topic)
        occurred_at = t or datetime.strptime(raw[0].timestamp, DATE_TIME_FORMAT)
        v = self._embed(f"{cue} {narrative}")  # f_emb(c ∥ N)
        episode = EpisodicMemory(
            cue=cue, narrative=narrative, raw=raw,
            embedding=tuple(v), occurred_at=occurred_at,
        )

        candidates = self._d_e.search(v, self._config.ke)
        if candidates:
            # 유사도 게이트 없이 매번 판정 콜 — §3.2.3의 plain top-Ke 충실
            # 재현. 상당수가 자명한 'new' 판정이라 비용 절감 후보지만 게이트
            # 추가는 논문 이탈이므로 baseline 유지, 변형 실험 재료로 기록
            # (검증 리뷰 N13)
            target = self._episodic_ops.select_target(
                narrative, occurred_at, len(raw), candidates
            )
            if target is not None:
                merged = self._integrate_into(target, episode)
                if merged is not None:
                    return merged
        self._d_e.add(episode)
        return episode

    def _integrate_into(
        self, target: EpisodicMemory, episode: EpisodicMemory
    ) -> EpisodicMemory | None:
        fields = self._episodic_ops.integrate(
            target, episode.cue, episode.narrative, episode.occurred_at, len(episode.raw)
        )
        if fields is None:  # 병합 포기 → 호출자가 별도 저장
            return None
        cue, narrative, occurred_at = fields
        merged = EpisodicMemory(
            cue=cue, narrative=narrative,
            raw=target.raw + episode.raw,  # P_k ∥ P_j
            embedding=tuple(self._embed(f"{cue} {narrative}")),
            occurred_at=occurred_at,
        )
        self._d_e.supersede(target, merged)
        # merged.raw에는 이미 distill된 target 메시지가 포함된다 — 이어질
        # semantic distillation에서 재추출 억제는 전적으로 evoke recall에
        # 걸려 있다 (원본 동일 동작, 검증 리뷰 N14)
        return merged

    def _semantic_distillation(self, episode: EpisodicMemory) -> None:
        evoked = self._d_s.evoke(
            np.asarray(episode.embedding), self._config.ks, self._config.tau
        )
        statements = (
            self._semantic_ops.distill(episode, evoked)
            if evoked
            else self._semantic_ops.direct_distill(episode)
        )
        self._d_s.consolidate([
            SemanticInsight(
                statement=s, embedding=tuple(self._embed(s)),
                source_episode_uuid=episode.uuid,
            )
            for s in statements
        ])
