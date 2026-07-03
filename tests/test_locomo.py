"""LoCoMo 로더의 데이터 불변량 테스트.

이 수치들은 2026-07-03 전수 조사에서 확정한 값이다 (노트북 01 참고).
데이터 파일이나 로더가 바뀌어 이 테스트가 깨지면, 점수 비교 가능성이
깨졌다는 뜻이므로 원인을 반드시 규명할 것.
"""
from collections import Counter

import pytest

from memlab.data import Category, load_locomo


@pytest.fixture(scope="module")
def samples():
    return load_locomo()


def test_top_level_counts(samples):
    assert len(samples) == 10
    assert sum(s.num_turns for s in samples) == 5882
    assert sum(len(s.qa) for s in samples) == 1986


def test_category_distribution(samples):
    dist = Counter(q.category for s in samples for q in s.qa)
    assert dist == {
        Category.MULTI_HOP: 282,
        Category.TEMPORAL: 321,
        Category.OPEN_DOMAIN: 96,
        Category.SINGLE_HOP: 841,
        Category.ADVERSARIAL: 446,
    }


def test_image_turns(samples):
    n = sum(1 for s in samples for sess in s.sessions for t in sess.turns if t.has_image)
    assert n == 1226


def test_answers_are_str_or_none(samples):
    for s in samples:
        for q in s.qa:
            assert q.answer is None or isinstance(q.answer, str)


def test_adversarial_mostly_unanswerable(samples):
    adv = [q for s in samples for q in s.qa if q.category == Category.ADVERSARIAL]
    assert len(adv) == 446
    # 444개는 answer 없음(답하지 않는 것이 정답), 2개는 데이터 예외로 answer 보유
    assert sum(1 for q in adv if q.answer is None) == 444
    assert all(q.adversarial_answer is not None for q in adv)


def test_sessions_sorted_and_nonempty(samples):
    for s in samples:
        indices = [sess.index for sess in s.sessions]
        assert indices == sorted(indices)
        # 유령 세션(date_time만 있고 본문 없음)은 로더가 생성하지 않는다
        assert all(len(sess.turns) > 0 for sess in s.sessions)


def test_evidence_dirt_is_exactly_nine(samples):
    """evidence 2,815개 중 9개는 벤치마크 자체의 어노테이션 노이즈다.

    (한 문자열에 여러 id, 깨진 형식 'D'/'D:11:26', 0 패딩 'D30:05' 등.)
    로더는 원문을 보존하므로 이 9개는 그대로 남아야 한다 — 이 숫자가
    변하면 로더가 데이터를 조용히 '고치기' 시작했다는 신호다.
    """
    unresolved = 0
    for s in samples:
        ids = {t.dia_id for sess in s.sessions for t in sess.turns}
        unresolved += sum(
            1 for q in s.qa for e in q.evidence if e not in ids
        )
    assert unresolved == 9


def test_limit(samples):
    assert len(load_locomo(limit=1)) == 1
    assert load_locomo(limit=1)[0].sample_id == samples[0].sample_id


def test_two_speakers_strict_alternation(samples):
    """모든 샘플은 정확히 두 화자, 세션 안에서는 완벽히 번갈아 말한다.

    MemoryOS 재구현의 쌍 접기 로직이 이 불변량에 의존한다. 단, 세션의
    45%(124/272)는 speaker_b가 먼저 말한다 — 원본 process_conversation은
    이 경우 직전 세션의 발화를 덮어쓰거나(61턴 유실) 세션 경계를 넘어
    잘못 짝짓는다(57턴). baseline은 이 동작까지 복제한다.
    """
    starts_with_b = 0
    for s in samples:
        declared = {s.speaker_a, s.speaker_b}
        for sess in s.sessions:
            assert {t.speaker for t in sess.turns} <= declared
            assert all(
                prev.speaker != cur.speaker
                for prev, cur in zip(sess.turns, sess.turns[1:])
            )
            if sess.turns[0].speaker == s.speaker_b:
                starts_with_b += 1
    assert starts_with_b == 124


# ──────────────────────────── 무결성 증명 ────────────────────────────
# 위 테스트들은 "개수"만 본다. 아래 두 테스트가 "내용"을 증명한다:
# ① 충실성 — 보관하는 모든 필드는 raw JSON과 완전히 같아야 한다.
# ② 회계 — raw JSON의 모든 키는 보관 목록 아니면 문서화된 버림 목록에
#    속해야 한다. 모르는 키를 조용히 버리는 것은 불가능하다.

import json
import re

from memlab.config import LOCOMO_PATH

_SESSION_KEY = re.compile(r"^session_(\d+)$")

# locomo.py docstring의 "버리는 것" 목록과 일치해야 한다
DROPPED_TURN_KEYS = {"img_url", "query", "re-download"}
DROPPED_SAMPLE_KEYS = {"event_summary", "observation", "session_summary"}
KEPT_TURN_KEYS = {"speaker", "dia_id", "text", "blip_caption"}
KEPT_QA_KEYS = {"question", "answer", "evidence", "category", "adversarial_answer"}


@pytest.fixture(scope="module")
def raw():
    return json.loads(LOCOMO_PATH.read_text(encoding="utf-8"))


def test_fidelity_every_kept_field_matches_raw(raw, samples):
    """턴 5,882개·QA 1,986개 전부를 raw와 1:1 대조한다."""
    assert len(raw) == len(samples)
    for raw_s, s in zip(raw, samples):
        assert raw_s["sample_id"] == s.sample_id
        conv = raw_s["conversation"]
        assert conv["speaker_a"] == s.speaker_a
        assert conv["speaker_b"] == s.speaker_b

        # 본문이 있는 세션은 하나도 빠짐없이, 순서는 index 오름차순으로
        raw_sessions = {
            int(m.group(1)): v
            for k, v in conv.items()
            if (m := _SESSION_KEY.match(k))
        }
        assert sorted(raw_sessions) == [sess.index for sess in s.sessions]

        for sess in s.sessions:
            assert conv.get(f"session_{sess.index}_date_time", "") == sess.date_time
            raw_turns = raw_sessions[sess.index]
            assert len(raw_turns) == len(sess.turns)
            for rt, t in zip(raw_turns, sess.turns):
                assert rt["speaker"] == t.speaker
                assert rt["dia_id"] == t.dia_id
                assert rt["text"] == t.text
                assert (rt.get("blip_caption") or None) == t.blip_caption

        assert len(raw_s["qa"]) == len(s.qa)
        for rq, q in zip(raw_s["qa"], s.qa):
            assert rq["question"] == q.question
            raw_ans = rq.get("answer")
            assert q.answer == (str(raw_ans) if raw_ans is not None else None)
            assert list(q.evidence) == rq.get("evidence", [])
            assert int(q.category) == rq["category"]
            assert q.adversarial_answer == rq.get("adversarial_answer")


def test_accounting_no_silently_ignored_keys(raw):
    """raw의 모든 키 = 보관 목록 ∪ 문서화된 버림 목록. 잉여도 누락도 없어야 한다."""
    sample_keys, conv_keys, turn_keys, qa_keys = set(), set(), set(), set()
    ghost_date_times = 0
    for raw_s in raw:
        sample_keys |= raw_s.keys()
        conv = raw_s["conversation"]
        conv_keys |= conv.keys()
        session_indices = {
            int(m.group(1)) for k in conv if (m := _SESSION_KEY.match(k))
        }
        for k, v in conv.items():
            if _SESSION_KEY.match(k):
                for t in v:
                    turn_keys |= t.keys()
            elif k.endswith("_date_time"):
                if int(k.split("_")[1]) not in session_indices:
                    ghost_date_times += 1
        for q in raw_s["qa"]:
            qa_keys |= q.keys()

    assert sample_keys == {"sample_id", "conversation", "qa"} | DROPPED_SAMPLE_KEYS
    assert turn_keys == KEPT_TURN_KEYS | DROPPED_TURN_KEYS
    assert qa_keys == KEPT_QA_KEYS
    # conversation 레벨: speaker 둘 + session_N + session_N_date_time 외엔 없어야 함
    leftovers = {
        k for k in conv_keys
        if k not in ("speaker_a", "speaker_b")
        and not _SESSION_KEY.match(k)
        and not re.match(r"^session_\d+_date_time$", k)
    }
    assert leftovers == set()
    assert ghost_date_times == 16  # 유령 세션 — 로더가 버리는 유일한 date_time
