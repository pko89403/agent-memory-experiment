---
description: 로컬 스모크 런 절차 — LM Studio 점검 → 발사 → 진행 증명 → 결과 보고. 스모크/전량 런을 돌릴 때 사용.
---

# 스모크 런: 점검 → 발사 → 증명 → 보고

실제 사고들(침묵 멈춤 2회, thinking 토큰 잠식, 75발화 JSON 잘림)에서 수렴한
절차. $ARGUMENTS가 있으면 run-id로 쓴다 (기본 `smoke-local`).

## 1. 사전 점검 — 실측으로만 통과

- `curl -s localhost:1234/v1/models` — 서버 기동 + `qwen3.5-9b-mlx` 로드 확인.
- ping 1회 (max_tokens 20, "reply with pong"): 응답이 비면 thinking이 켜진 것
  → 사용자에게 Jinja 템플릿 `{%- set enable_thinking = false %}` 확인 요청.
- context length 16384는 API로 못 본다 — GUI 설정이므로 사용자에게 확인.

## 2. 발사

- `caffeinate -i uv run memlab-run --limit N --run-id <id>` 를 백그라운드로.
- 기존 체크포인트가 있으면 그 대화는 skip된다는 걸 미리 알린다.

## 3. 진행 증명 — 신뢰는 스냅샷으로

- 로그의 `N 발화, M 호출, Ts` 줄로 페이스를 중간 보고한다.
- 로그가 오래 침묵하면: 호출 카운터 변화부터 확인, 그래도 의심되면
  `faulthandler.dump_traceback_later`로 물증을 잡는다.
- 중간 결과 없이 "돌고 있어요"라고만 말하지 않는다.

## 4. 결과 보고

- 러너의 카테고리 리포트 표 + 비용(호출/토큰/시간) + 실패 문항 수.
- 카테고리별 예측 샘플을 뽑아 정성 패턴을 1~2개 짚는다
  (예: temporal 날짜 앵커링 실패, adversarial 함정 응답).
