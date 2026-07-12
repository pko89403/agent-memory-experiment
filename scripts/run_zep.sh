#!/usr/bin/env bash
# zep 런 발사대 — 사전 점검(실측) 후 러너를 caffeinate로 돌리고 로그를 남긴다.
#
# 사용법:
#   scripts/run_zep.sh check            # 사전 점검만
#   scripts/run_zep.sh smoke [run-id]   # 대화 1편 (기본 run-id: zep-smoke)
#   scripts/run_zep.sh full  [run-id] [추가 러너 인자...]
#                                       # 대화 10편 전량 (기본: zep-baseline)
#
# 병렬 워커: 자기 Neo4j를 MEMLAB_NEO4J_URI로 지정하고 --samples로 분담한다.
#   MEMLAB_NEO4J_URI=bolt://localhost:7688 \
#     scripts/run_zep.sh full zep-baseline --samples conv-44,conv-47,...
#   (점검도 그 포트로 한다. 로그는 워커별: console.log / console.7688.log)
#
# - 진행 로그는 터미널 + runs/<run-id>/ 양쪽에 남는다 (tee).
# - 중단 후 같은 run-id로 재실행하면 완료된 대화는 checkpoint로 skip된다.
# - 끝나면 러너가 카테고리별 채점표를 출력한다.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:?usage: run_zep.sh check|smoke|full [run-id]}"

# ── 사전 점검 — 실측으로만 통과 (스모크 절차) ──────────────────────
curl -sf -m 3 localhost:1234/v1/models | grep -q qwen3.5-9b-mlx \
  || { echo "FAIL: LM Studio에 qwen3.5-9b-mlx가 없다 — 서버 기동 + 모델 로드"; exit 1; }
pong=$(curl -sf -m 60 localhost:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b-mlx","messages":[{"role":"user","content":"reply with pong"}],"max_tokens":20}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())')
[ -n "$pong" ] \
  || { echo "FAIL: ping 응답이 비었다 — thinking 켜짐 의심 (Jinja 템플릿에 enable_thinking=false)"; exit 1; }
neo4j_port=$(echo "${MEMLAB_NEO4J_URI:-bolt://localhost:7687}" | sed -E 's/.*:([0-9]+).*/\1/')
nc -z localhost "$neo4j_port" >/dev/null 2>&1 \
  || { echo "FAIL: Neo4j가 없다 (port $neo4j_port) — neo4j start 또는 워커 인스턴스 기동"; exit 1; }
echo "OK: LM Studio(ping='${pong}') + Neo4j(:$neo4j_port)"
echo "주의: context length 16384 / parallel 4는 GUI에서만 확인 가능"

case "$mode" in
  check) exit 0 ;;
  smoke) run_id="${2:-zep-smoke}" ;;
  full)  run_id="${2:-zep-baseline}" ;;
  *) echo "unknown mode: $mode" >&2; exit 1 ;;
esac

cmd=(uv run memlab-run --method zep --run-id "$run_id")
[ "$mode" = smoke ] && cmd+=(--limit 1)
shift 2 2>/dev/null || shift $#   # mode·run-id 뒤 나머지는 러너로 pass-through
cmd+=("$@")

log="console.log"
[ "$neo4j_port" != 7687 ] && log="console.$neo4j_port.log"   # 워커별 로그 분리
mkdir -p "runs/$run_id"
echo "발사: ${cmd[*]}  (로그: runs/$run_id/$log)"
caffeinate -i "${cmd[@]}" 2>&1 | tee -a "runs/$run_id/$log"
