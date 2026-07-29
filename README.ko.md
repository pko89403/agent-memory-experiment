# memlab — Agent Memory 실험 하네스 & 스터디 가이드

*[English](README.md) · 한국어*

LoCoMo 벤치마크 위에서 agent memory 논문들을 **재구현하고, 검증하고, 비교하는** 프로젝트.
메소드는 계속 추가된다 — MemoryOS(arXiv:2506.06326), Zep(arXiv:2501.13956,
temporal knowledge graph), Nemori(arXiv:2508.03341, adaptive memory
distillation)를 재구현했고, 여기에 부품 하나만 갈아 끼워 질문 하나를
분리해낸 합성판(Nemori × A-MEM, arXiv:2502.12110)이 더해졌다.
cause-aware forgetting 변형을 같은 조건에서 실험하는 것이 최종 목표다.

이 저장소는 **코드이자 가이드**다:

> **로직은 패키지에, 이야기는 노트북에.**

- `src/memlab/` — 정답이 하나여야 하는 코드 (로더, 메트릭, 메소드, 러너). 실험은 CLI로 실행해 재현 가능하게.
- `notebooks/` — 배움의 서사 (데이터셋 탐구, 메트릭 손계산, 아키텍처 해부). 값싸고 빠른 것만 실행.

## 디렉토리 구조

```
├── src/memlab/
│   ├── config.py                 # 모든 경로 + 레퍼런스 SHA 핀 (레퍼런스의 유일한 커밋 흔적)
│   ├── data/                     # LoCoMo 로더
│   ├── methods/                  # 공통 인터페이스: ingest(turn) / answer(question)
│   │   ├── memoryos/             # MemoryOS 재구현 (method #1, baseline)
│   │   ├── zep/                  # Zep 재구현 (temporal knowledge graph)
│   │   ├── nemori/               # Nemori 재구현 (adaptive memory distillation)
│   │   ├── amem/                 # A-MEM 부품 (note construct/link/evolve) — 단독 메소드 아님
│   │   └── nemori_amem/          # 합성: Nemori distillation + A-MEM note management
│   ├── evaluation/               # set-F1 / 표준 F1 / BLEU-1, 카테고리별 리포트
│   └── run.py                    # 실험 러너 CLI (체크포인트·에러 격리)
├── notebooks/                    # 가이드 챕터 01~06
├── scripts/                      # fetch_data.py(데이터), fetch_reference.py(레퍼런스)
├── external/                     # 원본 repo들 (gitignore — SHA 핀으로만 커밋)
├── runs/                         # 실험 결과 + config 스냅샷 (gitignore)
└── tests/                        # 로더 무결성·채점 함수 테스트
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
수정하게 된다. 원본은 읽기 교재이자, 논문이 침묵하는 상수·프롬프트의 출처로 쓴다.

**3. 재구현의 명세서는 논문이다. 코드는 참고자료.**
원본 eval 코드는 논문과 다르고(LFU 삭제, 체인 통째 이관, 죽은 recency)
버그도 있다(발화 유실, 엉뚱한 세그먼트 heat 상승). 어차피 LLM도 다르므로
(qwen3.5-9b-mlx vs gpt-4o-mini) 코드 버그까지 복제할 이유가 없다 —
**논문 서술대로 구현**하고, 논문이 침묵하는 상수(α·β·γ 등)만 코드에서 차용한다.
따라서 우리 baseline은 "논문 명세의 MemoryOS + qwen3.5-9b-mlx(로컬 LM Studio)"이며,
논문 표의 수치와 직접 비교하지 않는다. 변형(forgetting) 실험의 기준선으로만 쓴다.

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

LLM은 로컬 LM Studio다 (`localhost:1234`, `qwen3.5-9b-mlx`) — API 키가
필요 없다. 서버에 모델을 로드할 때 두 가지가 필수: **context length 16384**,
**thinking 끄기** (Prompt Template(Jinja) 최상단에
`{%- set enable_thinking = false %}`). Groq free API는 폴백 전용이다
(TPM 6K 한도로 실용 불가 실측 — `config.py` 주석 참고). 원 논문은
gpt-4o-mini였으므로 논문 수치와의 직접 비교는 포기하고, **같은 모델로 잰
자체 baseline vs 변형**의 비교에 집중한다. 데이터 탐구·메트릭·차분 테스트는
LLM 없이 전부 가능하다.

## 현재 상태 — 전량 baseline (LoCoMo 10편, 1,540문항)

전량 10편을 qwen3.5-9b-mlx로 완주한 메소드별 set_f1. 마지막 열은 네 번째
논문이 아니라 합성판이다 (아래 참고). 아티팩트: `runs/`(커밋 안 됨):

| category | n | MemoryOS | Zep | Nemori | Nemori×A-MEM |
|---|---|---|---|---|---|
| MULTI_HOP | 282 | 0.257 | 0.337 | 0.340 | **0.342** |
| TEMPORAL | 321 | 0.282 | 0.177 | **0.434** | 0.414 |
| OPEN_DOMAIN | 96 | 0.190 | 0.135 | **0.190** | 0.186 |
| SINGLE_HOP | 841 | 0.348 | **0.500** | 0.462 | 0.456 |
| ADVERSARIAL ↓ | 446 | 0.287 | 0.286 | 0.236 | **0.214** |
| **OVERALL (1~4)** | 1540 | 0.307 | 0.380 | **0.417** | 0.410 |

ADVERSARIAL은 함정 오답 기준이라 낮을수록 좋다(↓). 대화당 비용: Zep
~1만 콜/~36h(message 단위 처리), MemoryOS ~1.7천 콜/~3h, Nemori
~370 콜(episode 단위) — 논문 §4.3의 효율 주장대로 자릿수가 다르다.
MemoryOS 전량은 conv-26 스모크값보다 낮게 나왔다(0.322 → 0.307) —
논문 3종의 순위는 Nemori > Zep > MemoryOS.

패턴: **Nemori의 temporal 0.434는 Zep(0.177)의 2.4배** — episode 서사가
상대 시점("yesterday")을 절대 날짜로 앵커링하는 설계(논문 §3.2.2)가 그대로
점수가 됐다. adversarial도 최저(=최선)로, 함정 질문에 기억을 지어내는
빈도가 가장 낮다. 반대로 single-hop은 Zep(0.500)이 앞선다 — 단순 사실
회수는 knowledge graph의 정밀 검색이 유리. 논문이 주장한 temporal 우위와
전체 우위(Table 2)가 로컬 9B에서도 방향 그대로 재현된다.

### 합성판 — memory management가 naive append를 이기는가

Nemori는 fact를 distill해서 납작한 리스트에 append한다. A-MEM
(arXiv:2502.12110)은 반대로, 받은 것을 **관리**한다 — note마다 LLM이
keywords·tags·context를 붙이고, 의미적 이웃과 링크를 잇고, 새 note가
들어올 때마다 이웃을 다시 쓴다. 그래서 부품 하나만 바꿨다: Nemori의
semantic store를 A-MEM note store로 교체하고, 답변 경로와 하이퍼파라미터는
전부 그대로 뒀다. 단일 변인, 단일 질문 — **distill된 fact 위에 management를
얹으면 naive append보다 나은가?**

**대체로 아니다.** overall 0.410 vs 0.417로 오차 범위 안이고, ingest 콜은
약 2배다. 유일한 실질 이득은 adversarial **0.236 → 0.214** — 여기 있는 어떤
메소드보다 낮다. 링크와 evolution이 회수 컨텍스트의 주제 응집을 높여
함정 질문에서 엉뚱한 재료를 덜 끌어오는 것으로 보인다. 반대로 temporal은
−0.020으로 내려갔는데 기전은 아직 규명하지 못했다: 표시되는 텍스트는
statement 원문 그대로라 evolution이 답을 직접 훼손할 수는 없고, 회수
**순위** 변화가 유력하다 — 계측이 남은 숙제다.

구현에서 한 번 값을 치른 대목: A-MEM의 임베딩은 `concat(content, keywords,
tags, context)`인데(Eq.3), 처음엔 Nemori의 evoke 게이트도 그 벡터로
돌렸다. τ=0.55는 statement 공간에서 캘리브레이션한 값이라 concat 공간에서는
유사도 분포가 통째로 내려앉았고, 게이트가 6/7이 아니라 3/14로 닫히면서
distillation cascade가 cold start 경로로 붕괴했다. 해법은 두 공간을 각각
색인하는 것 — evoke는 statement 벡터, search·이웃 검색은 concat 벡터.
**similarity threshold는 상수가 아니라 임베딩 공간의 함수다.**

## 가이드 로드맵 (notebooks/)

| 챕터 | 주제 | 배우는 것 |
|---|---|---|
| 01 | LoCoMo 데이터셋 | 10 샘플 / 5,882 턴 / 1,986 QA. 카테고리(1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial)를 evidence 개수·답변 형태로 데이터에서 직접 검증 |
| 02 | Memory 검증 방법 | ingest → answer → score 패러다임. repo식 set-F1 vs 표준 F1 vs BLEU-1, 배치 채점기 |
| 03 | MemoryOS 관찰 | 실제 대화 조각을 MemoryOS에 먹이고 기억의 형성·회상·승격·forgetting을 지켜본다 (LLM 실호출) |
| 04 | Zep 관찰 | temporal knowledge graph 구축 — entity·fact 추출, bi-temporal 스탬프, community, RRF 검색 |
| 05 | Nemori 관찰 | adaptive memory distillation — partition→서사 episode→병합, predict-calibrate로 semantic distillation |
| 06 | Nemori × A-MEM | 부품 하나 교체 — distill된 fact가 링크되고 스스로 evolve하는 note가 된다. 링크가 생기는 것을 보고, evoke 게이트와 회수 슬롯을 계측한다 |

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
- **MTM `max_capacity=2000` vs 대화당 페이지 최대 ~340개 → 벤치마크에서
  heat 기반 forgetting(삭제)이 한 번도 발동하지 않는다.** 승격(heat>5 → LPM)만
  작동. 또한 recency는 실행 중 사실상 상수(γ=0.0001, 실제 벽시계 기준)라
  heat ≈ 0.8·N_visit + 0.8·L_interaction. forgetting 실험은 용량 압력을
  따로 만들어야 하며, 그 조건의 baseline도 별도 측정 필요.
- `process_conversation`의 pair folding(발화→page 묶기) 로직은 세션이 speaker_b로 시작하는 경우
  (124/272 세션)를 잘못 다룬다: **61턴이 덮어쓰기로 유실**되고 **57턴이
  세션 경계를 넘어 잘못 짝지어진다** (타임스탬프도 이전 세션 것으로 오염).
  영향 측정 결과 **QA evidence 94건**이 유실(4)/오염(90) 턴을 가리키며,
  특히 temporal 질문은 잘못된 타임스탬프로 저장된 발화를 근거로 요구한다.
  → 우리 재구현은 유실 없는 pair folding을 쓴다 (논문 우선 전략 — 버그 비복제).

## 출처 및 라이선스

- **이 저장소의 코드**: MIT ([LICENSE](LICENSE))
- **MemoryOS** ([BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS), Apache-2.0;
  논문 arXiv:2506.06326): 재구현의 참고 구현체.
  `prompt_templates.py`의 프롬프트와 90차원 성격 항목 목록은 해당 repo
  (`eval/`, `memoryos-pypi/`)에서 차용·수정했다. 차용분에는 Apache-2.0이
  적용된다 — 전문은 [licenses/MemoryOS-Apache-2.0.txt](licenses/MemoryOS-Apache-2.0.txt).
- **Zep** ([getzep/graphiti](https://github.com/getzep/graphiti), Apache-2.0;
  논문 arXiv:2501.13956), **Nemori**
  ([nemori-ai/nemori](https://github.com/nemori-ai/nemori), MIT; 논문
  arXiv:2508.03341), **A-MEM**
  ([agiresearch/A-mem](https://github.com/agiresearch/A-mem), MIT; 논문
  arXiv:2502.12110): 참고 구현체. `external/`에 SHA로 고정하고, 논문이
  상수·프롬프트를 명시하지 않은 곳에서만 참고했다. A-MEM의 note 구성·
  evolution 프롬프트는 논문 부록판이 불완전해서 원본 코드판을 차용·수정했다.
- **LoCoMo** ([snap-research/locomo](https://github.com/snap-research/locomo),
  CC BY-NC 4.0; Maharana et al., ACL 2024): 벤치마크 데이터.
  **데이터 파일은 이 저장소에 포함되지 않으며** `scripts/fetch_data.py`가
  원 출처에서 받는다. 노트북 출력에 포함된 대화 발췌는 비상업적 연구
  목적의 인용이다.
