# memlab — Claude Code 작업 지침

LoCoMo 벤치마크에서 agent memory 논문들을 재구현·검증·비교하는 스터디
프로젝트. 논문은 계속 추가된다 — 첫 메소드 MemoryOS(baseline 확보 중),
다음은 Zep(temporal knowledge graph). cause-aware forgetting 변형 실험이
최종 목표. 상세는 README.md.

## 작업 규범 (사용자와 합의된 것 — 어기지 말 것)

- **커밋은 사용자가 요청할 때만.** 변경 → 검증 → 보고까지 하고 멈춘다.
- **테스트 파일을 새로 만들지 않는다.** 동작 검증은 일회용 스크립트로
  실행해 결과를 보여주고 버린다. 기존 tests/는 유지.
- **추정 대신 실측.** 비용·속도·한도는 프로브/드라이런/스냅샷으로 증명한다.
  장시간 작업은 중간 로그를 보여줘야 신뢰받는다.
- **대화하며 단계별로.** 큰 방향은 제안 → 합의 → 실행. 긴 자율 작업 금지.
- **기술 용어는 영어 그대로** (eviction, consolidation, retrieval, heat...).
  번역 조어(되뇌기, 응고)·비유(수업, 학생, N막) 금지. 한국어는 평범한 문장에만.
- **코드 식별자에 한글 금지.** 주석·docstring·출력 라벨은 한국어 OK.
- **YAGNI — 단, 코드의 아름다움은 실효성과 동급 가치다.** 죽은 코드·매직
  넘버·낱개 함수 주입·파라미터 나열을 남기지 말 것.
- 가이드 노트북 스타일·편집 규칙은 `.claude/rules/notebooks.md`
  (notebooks/ 작업 시 자동 로드).

## 아키텍처 규칙

- **논문이 명세서, 원본 코드는 참고자료.** 논문이 침묵하는 상수·프롬프트만
  원본에서 차용하고, 모든 해석·결정은 해당 파일 docstring에 기록한다.
- 소켓은 `MemoryMethod(ingest/answer)` 하나. 메소드 고유 스키마(page,
  segment 등)는 `methods/<이름>/` 안에만 — 공용 계층에 새지 않게.
- 새 논문 추가 패턴: `methods/<이름>/` 구현 + `external/<이름>` SHA 고정
  reference + 가이드 노트북 한 편. 공용 계층(러너·채점·데이터)을 고쳐야
  한다면 설계 신호이므로 먼저 합의.
- 공유 인프라(LLM 프로바이더, 임베딩)는 `memlab/` 최상위.
- 의존성 주입: 행위 묶음은 소비자가 선언한 Protocol로, 단일 순수 함수는
  콜러블로. 하이퍼파라미터는 frozen dataclass Config로.
- 핀·경로·모델 설정의 단일 진실은 `config.py`. 비밀은 `.env`(GROQ_API_KEY)만.
- 러너는 메소드를 모른다 — method factory로 주입.

## 자주 쓰는 명령

```bash
uv run pytest tests/ -q                          # 테스트 (20개)
uv run memlab-run --limit 1 --run-id smoke-local # 스모크 (장시간: caffeinate -i 붙이기)
uv run memlab-run --run-id X --score-only        # 저장된 아티팩트만 재채점
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/NN_*.ipynb
uv run scripts/fetch_data.py                     # LoCoMo (SHA-256 검증)
uv run scripts/fetch_reference.py                # external/MemoryOS (SHA 고정)
```

## 실행 환경과 함정

- **LLM은 로컬 LM Studio** (`localhost:1234`, `qwen3.5-9b-mlx`).
  서버 기동 + 모델 로드 필수 조건: **context length 16384**,
  **thinking 끄기** — Prompt Template(Jinja) 최상단에
  `{%- set enable_thinking = false %}` (API 파라미터로는 불가, 실측 확인).
- Groq 무료 티어는 폴백 전용 — TPM 6K 벽으로 실용 불가 판정 (2026-07-04).
- 로컬 소형 모델은 스키마를 벗어나려 함 — 구조화 출력 Pydantic 모델에
  `extra="forbid"` 필수, 빠듯한 max_tokens 금지.
- 실행 중인 러너가 있을 때 코드 수정은 안전하다 (메모리에 로드된 프로세스).
- macOS 백그라운드 프로세스가 "멈춘 듯"하면: 쿼터 프로브(소비량 변화)와
  `faulthandler.dump_traceback_later`로 물증을 잡는다.
