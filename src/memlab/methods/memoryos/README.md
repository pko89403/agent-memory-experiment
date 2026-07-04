# MemoryOS 재구현

**Memory OS of AI Agent** (BAI-LAB, [arXiv:2506.06326](https://arxiv.org/abs/2506.06326))의
논문 명세 기반 재구현. 원본 저장소의 코드가 아니라 **논문 서술을 명세서로**
삼았다 (원본 eval 코드는 논문과 다르고 버그도 있다 — 루트 README의
"원본 코드의 특이점" 참고).

## 논문이 푸는 문제

LLM은 컨텍스트 길이가 고정이라 수개월에 걸친 대화에서 사용자에 대한
기억(사실, 취향, 과거 대화)을 유지하지 못한다. MemoryOS는 운영체제의
메모리 계층(캐시 → 메인 메모리 → 디스크)과 segmented paging에서 구조를
빌려와, 대화 기억을 3계층으로 관리하는 시스템을 제안한다.
LoCoMo 벤치마크에서 baseline 대비 F1 +49.11% (gpt-4o-mini)를 보고했다.

## 핵심 설계 (논문 3장)

```
STM   dialogue page {Q, R, T} 큐 (7칸)
      page마다 dialogue chain: page_chain = {Q, R, T, meta_chain}   (식 1)
      — LLM이 ① 이어짐 판단 ② chain 전체 재요약
  ↓ 큐가 차면 가장 오래된 page를 FIFO 이관
MTM   주제별 segment (최대 200개)
      편입: F_score = cos(e_s, e_p) + Jaccard(K_s, K_p) > θ=0.6     (식 2, 3)
      Heat = α·N_visit + β·L_interaction + γ·R_recency              (식 4)
      — 검색에 걸리면 N_visit 증가 / 자리가 부족하면 heat 최저 삭제
  ↓ heat > τ=5 승격 (이후 L_interaction ← 0 리셋)
LPM   dual persona (영구 저장)
      User: Profile(90차원 등급 포함) + KB(사실, 100칸 FIFO)
      Agent: Profile + Traits(100칸 FIFO)
```

답변 생성(논문 3.4): STM 전부 + MTM 2단계 검색(top-m=5 segment →
top-k=10 page) + LPM(KB·Traits 각 top-10, Profile 전량)을 하나의
프롬프트로 합쳐 LLM이 답한다.

## 파일 맵

| 파일 | 논문 대응 |
|---|---|
| `schema.py` | page / page_chain / segment 정의 (논문 기호 1:1) |
| `short_term.py` | STM 큐 + dialogue chain (식 1) |
| `mid_term.py` | MTM: F_score 편입(식 2·3), heat(식 4), eviction, 2단계 검색 |
| `long_term.py` | LPM: dual persona 저장·검색 |
| `llm_ops.py` | LLM 연산 6종 — tier가 선언한 Protocol과 1:1 (ChainLlmOps 등) |
| `prompt_templates.py` | 프롬프트 (논문에 없어 원본 repo에서 차용, 출처 명시) |
| `method.py` | 전체 조립 `MemoryOS` + 하이퍼파라미터 `MemoryOSConfig` |

## 논문이 침묵해서 우리가 정한 것

- **heat 상수 α=0.8, β=0.8, γ=0.0001** — 논문은 "계수가 중요도를 결정한다"고만
  하고 값을 안 밝힘 → 원본 eval 코드의 값 차용. γ≈0이라 recency는 사실상
  꺼져 있다 (아래 관찰 참고)
- **세그먼트 요약 갱신 주기** — 페이지가 병합될 때마다 member 전체 재요약
  (논문: "summarized by a LLM based on the related dialogue pages")
- **pair folding** — LoCoMo는 친구 둘의 대화라 speaker_a→Q, speaker_b→R로
  묶는다. 유실 없는 방식 (원본 eval의 발화 유실 버그는 계승하지 않음)
- **원본의 multi_summary(주제 2개 분할)는 미채택** — 논문에 없는 메커니즘이고
  배치 중복 삽입 버그의 원인이었다. 우리는 페이지 단위 이관 + 단일 요약

## 실동작에서 관찰된 것 (notebooks/03)

- 실제 대화에서 세그먼트 병합(F > 0.6)은 드물어 **잘게 파편화**된다
- 승격은 검색(N_visit)과 결합해야 실제로 발동한다
- eviction 희생자는 항상 "작고 한 번도 검색되지 않은 새 세그먼트"다 —
  γ≈0이라 새 기억을 보호할 항이 없기 때문. **무엇을 왜 지우는지 묻지 않는
  삭제**이며, 이 병리가 cause-aware forgetting 연구의 출발점이다
