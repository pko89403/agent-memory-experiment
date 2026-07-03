# memlab — Agent Memory 실험 하네스 & 스터디 가이드

LoCoMo 벤치마크 위에서 agent memory 시스템을 **재구현하고, 검증하고, 비교하는** 프로젝트.
MemoryOS(arXiv:2506.06326)를 직접 재구현해 baseline을 확보한 뒤, 새로운 memory 메소드
(cause-aware forgetting)를 같은 조건에서 실험하는 것이 최종 목표다.

이 저장소는 **코드이자 가이드**다:

> **로직은 패키지에, 이야기는 노트북에.**

- `src/memlab/` — 정답이 하나여야 하는 코드 (로더, 메트릭, 메소드, 러너). 실험은 CLI로 실행해 재현 가능하게.
- `notebooks/` — 배움의 서사 (데이터셋 탐구, 메트릭 손계산, 아키텍처 해부). 값싸고 빠른 것만 실행.

## 디렉토리 구조

```
├── src/memlab/
│   ├── config.py                 # 모든 경로 + MEMORYOS_SHA 핀 (레퍼런스의 유일한 커밋 흔적)
│   ├── data/                     # LoCoMo 로더
│   ├── methods/                  # 공통 인터페이스: ingest(turn) / answer(question)
│   │   └── memoryos/             # MemoryOS 재구현 (method #1, baseline)
│   ├── evaluation/               # set-F1 / 표준 F1 / BLEU-1, 카테고리별 리포트
│   └── run.py                    # 실험 러너 CLI (예정)
├── notebooks/                    # 가이드 챕터 01~05
├── scripts/fetch_reference.py    # 레퍼런스 repo를 고정 SHA로 클론/검증
├── external/MemoryOS/            # 원본 repo (gitignore — 커밋되지 않음)
├── runs/                         # 실험 결과 + config 스냅샷 (gitignore)
└── tests/                        # 차분 테스트 (내 재구현 vs 원본)
```

## 설계 결정과 이유

**1. 원본 MemoryOS는 저장소에 포함하지 않는다.**
타인의 repo를 통째로 커밋하면 라이선스·이력이 오염된다. 대신 `config.py`의
`MEMORYOS_SHA` 한 줄만 커밋하고, `scripts/fetch_reference.py`가 그 SHA로
클론/검증한다. 누가 언제 받아도 정확히 같은 코드 = baseline이 어떤 코드에서
나온 수치인지 항상 답할 수 있다.

**1-1. 벤치마크 데이터는 원 출처에서 받는다.**
LoCoMo-10의 원 출처는 snap-research/locomo (Maharana et al., ACL 2024)다.
`scripts/fetch_data.py`가 고정 커밋(`LOCOMO_SHA`)에서 내려받고 SHA-256
체크섬(`LOCOMO_SHA256`)으로 내용을 검증한다 — 시점과 내용의 이중 고정.
MemoryOS repo에도 사본이 vendored돼 있지만, 그건 "한 논문의 스냅샷"이지
벤치마크가 아니다. (2026-07-03 검증: 두 파일은 바이트 단위로 동일하므로
점수 비교 가능성에는 영향 없음.)

**2. MemoryOS를 어댑터로 감싸지 않고 재구현한다.**
STM/MTM/LPM, heat 계산, eviction을 직접 짜야 진짜 이해가 되고, 이후
cause-aware forgetting 변형을 붙일 때 남의 연구 코드가 아니라 내 코드를
수정하게 된다. 원본은 ① 읽기 교재 ② **차분 테스트 오라클**로 쓴다 —
같은 입력에 대해 내 `compute_heat()`와 원본 `compute_segment_heat()`의
출력을 비교하는 식. API 호출 없이 무료로 동치성을 검증할 수 있다.

**3. 재구현은 먼저 원본 eval 버전을 충실히 복제한다.**
원본 eval 설정(STM capacity=1, θ=0.6, top-k=10, 동일 프롬프트)을 그대로
따른다. 개선하고 싶은 게 보여도 baseline 확정 *후*에 바꾼다 — 안 그러면
"변형이 좋아진 건지, 재구현하다 달라진 건지" 구분할 수 없다.

**4. 메소드는 공통 인터페이스 뒤에 둔다.**
`ingest(turn)` / `answer(question)`만 구현하면 어떤 memory 시스템이든
같은 러너·같은 채점기로 평가된다. baseline과 변형이 **완전히 같은 조건**에서
비교되는 것이 이 하네스의 존재 이유다.

**5. 사소하지만: `eval/`이 아니라 `evaluation/`.**
`eval`은 Python 내장 함수라 모듈명으로 쓰면 shadowing 경고가 난다.
원본 repo는 `eval/`을 쓰지만 우리는 우리 규칙을 따른다.

## 시작하기

```bash
uv sync                              # Python 3.12 + 의존성 (uv.lock으로 고정)
uv run scripts/fetch_data.py         # LoCoMo-10을 원 출처에서 (SHA-256 검증)
uv run scripts/fetch_reference.py    # external/MemoryOS를 고정 SHA로 준비
```

실험 실행에는 `OPENAI_API_KEY` 환경변수가 필요하다 (LLM: gpt-4o-mini).
데이터 탐구·메트릭·차분 테스트는 API 키 없이 전부 가능하다.

## 가이드 로드맵 (notebooks/)

| 챕터 | 주제 | 배우는 것 |
|---|---|---|
| 01 | LoCoMo 데이터셋 | 10 샘플 / 5,882 턴 / 1,986 QA. 카테고리(1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial)를 evidence 개수·답변 형태로 데이터에서 직접 검증 |
| 02 | Memory 검증 방법 | ingest → answer → score 패러다임. repo식 set-F1 vs 표준 F1 vs BLEU-1의 차이를 손계산으로 |
| 03 | MemoryOS 해부 | STM(FIFO) → MTM(heat/evict) → LPM(profile) 구조, 원본 코드 읽기, 재구현 설계도 |
| 04 | 실험 실행 | 스모크 → 전량, 비용 관리, config 스냅샷, temperature 등 재현성 함정 |
| 05 | 새 메소드 만들기 | 인터페이스에 내 메소드 꽂기 — cause-aware forgetting이 여기서 시작 |

## Baseline 재현 시 알아둘 것 (원본 코드의 특이점)

원본 `eval/`을 읽으며 확인한, 논문·pypi 패키지와 다른 지점들:

- `ShortTermMemory(max_capacity=1)` — 논문은 7, pypi 기본값은 10. 매 턴 evict가
  일어나 턴마다 LLM 호출이 발생하는 원인.
- 실제 LLM 클라이언트는 `utils.py`의 모듈 전역 `gpt_client` 하나다.
  `OpenAIClient` 클래스의 key/base_url 설정은 openai 1.x에서 죽은 코드.
- 모델은 `gpt-4o-mini` 하드코딩, **temperature=0.7** — 실행마다 답이 달라질 수
  있다는 뜻. 메소드 비교 실험에서는 이 분산을 통제해야 한다.
- 원본 F1은 set-token 방식(토큰 빈도 무시)이라 표준 F1과 다르다. BLEU-1은
  원본 repo에 아예 없다 → 우리 `evaluation/`이 셋 다 계산한다.
- `get_embedding()`이 호출마다 SentenceTransformer를 새로 로드한다(성능 함정).
- cat5(adversarial) 446문항 중 444개가 empty answer — 채점 시 제외 옵션 필요.
- `process_conversation`의 쌍 접기 로직은 세션이 speaker_b로 시작하는 경우
  (124/272 세션)를 잘못 다룬다: **61턴이 덮어쓰기로 유실**되고 **57턴이
  세션 경계를 넘어 잘못 짝지어진다** (타임스탬프도 이전 세션 것으로 오염).
  영향 측정 결과 **QA evidence 94건**이 유실(4)/오염(90) 턴을 가리키며,
  특히 temporal 질문은 잘못된 타임스탬프로 저장된 발화를 근거로 요구한다.
  → 재구현 전처리에 스위치를 둔다: `pairing="original"`(baseline, 결함 복제)
  vs `pairing="lossless"`(개선 실험). 수정 효과 측정은 baseline 확정 후.
