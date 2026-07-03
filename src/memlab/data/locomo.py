"""LoCoMo-10 loader — the dataset exactly as it is on disk, typed.

설계 원칙: 로더는 데이터를 **있는 그대로** 표현한다. 메소드별 전처리
(예: MemoryOS의 speaker_a/b → user/assistant 쌍 접기, blip_caption을
text에 합치기)는 각 메소드 구현의 몫이다.

데이터 전수 조사에서 확인된 특이점 (2026-07-03, 노트북 01 참고):
- ``session_N_date_time``만 있고 ``session_N`` 본문이 없는 유령 세션 16개
  → 본문이 있는 세션만 파싱한다 (원본 eval도 동일하게 스킵).
- 턴 5,882개 중 1,226개는 이미지 공유 턴 (``blip_caption`` 보유).
  ``img_url``/``query``/``re-download``는 이미지 재수집용 메타데이터라 버린다 —
  memory 실험에서 모델이 볼 수 있는 건 캡션뿐이다.
- ``answer``가 int인 QA 6개 (연도 등) → str로 정규화.
- category 5(adversarial) 446개 중 444개는 ``answer`` 키 자체가 없고
  ``adversarial_answer``만 있다 (2개는 둘 다 보유).

버리는 것 (전체 목록 — tests/test_locomo.py의 회계 테스트가 강제):
- 턴: ``img_url``, ``query``, ``re-download`` (이미지 재수집용 메타데이터)
- 샘플: ``event_summary``, ``observation``, ``session_summary``
  (LoCoMo가 제공하는 부가 어노테이션. MemoryOS eval은 사용하지 않지만,
  요약 기반 baseline은 쓸 수 있으므로 필요해지면 로더에 추가할 것.)
- 본문 없는 세션의 ``session_N_date_time`` 16개.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from memlab.config import LOCOMO_PATH

_SESSION_KEY = re.compile(r"^session_(\d+)$")


class Category(IntEnum):
    """QA 카테고리. 코드→이름 매핑은 노트북 01에서 데이터로 검증했다:

    cat1은 평균 evidence 3.13개(98%가 2개 이상) → multi-hop,
    cat2는 답변의 80%가 날짜 → temporal,
    cat4는 evidence ~1개에 최대 분포(841개) → single-hop,
    cat5는 444/446이 정답 없음 → adversarial (답할 수 없어야 정답).
    """

    MULTI_HOP = 1
    TEMPORAL = 2
    OPEN_DOMAIN = 3
    SINGLE_HOP = 4
    ADVERSARIAL = 5


@dataclass(frozen=True)
class Turn:
    speaker: str
    dia_id: str  # "D3:7" = 세션 3의 7번째 발화. QA.evidence가 이 id를 참조한다.
    text: str
    blip_caption: str | None = None  # 이미지 공유 턴이면 캡션, 아니면 None

    @property
    def has_image(self) -> bool:
        return self.blip_caption is not None


@dataclass(frozen=True)
class Session:
    index: int  # session_N의 N (유령 세션 때문에 연속이 아닐 수 있음)
    date_time: str  # 원문 그대로, 예: "1:56 pm on 8 May, 2023"
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class QA:
    question: str
    answer: str | None  # int였던 6개는 str로 정규화. adversarial 대부분 None.
    evidence: tuple[str, ...]  # 근거 발화의 dia_id들
    category: Category
    adversarial_answer: str | None = None


@dataclass(frozen=True)
class Sample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: tuple[Session, ...]  # index 오름차순
    qa: tuple[QA, ...]

    @property
    def num_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)


def _parse_turn(raw: dict) -> Turn:
    return Turn(
        speaker=raw["speaker"],
        dia_id=raw["dia_id"],
        text=raw["text"],
        blip_caption=raw.get("blip_caption") or None,
    )


def _parse_qa(raw: dict) -> QA:
    answer = raw.get("answer")
    return QA(
        question=raw["question"],
        answer=str(answer) if answer is not None else None,
        evidence=tuple(raw.get("evidence", [])),
        category=Category(raw["category"]),
        adversarial_answer=raw.get("adversarial_answer"),
    )


def _parse_sample(raw: dict) -> Sample:
    conv = raw["conversation"]
    sessions = []
    for key, value in conv.items():
        m = _SESSION_KEY.match(key)
        if not m:
            continue  # speaker_a/b, session_N_date_time, 유령 date_time 등
        index = int(m.group(1))
        sessions.append(
            Session(
                index=index,
                date_time=conv.get(f"session_{index}_date_time", ""),
                turns=tuple(_parse_turn(t) for t in value),
            )
        )
    sessions.sort(key=lambda s: s.index)
    return Sample(
        sample_id=raw["sample_id"],
        speaker_a=conv["speaker_a"],
        speaker_b=conv["speaker_b"],
        sessions=tuple(sessions),
        qa=tuple(_parse_qa(q) for q in raw["qa"]),
    )


def load_locomo(
    path: str | Path = LOCOMO_PATH, limit: int | None = None
) -> list[Sample]:
    """LoCoMo-10을 로드한다.

    Args:
        path: locomo10.json 경로 (기본: 원 출처 snap-research/locomo에서
            fetch_data.py가 받아 체크섬 검증한 external/locomo10.json).
        limit: 앞에서부터 N개 샘플만 (스모크 테스트용 — 핸드오프 문서의
            LOCO_LIMIT에 해당하며, env가 아니라 명시적 인자로 받는다).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [_parse_sample(s) for s in raw]
    return samples[:limit] if limit is not None else samples
