#!/usr/bin/env python3
"""PostToolUse hook — 기계적으로 검사 가능한 규범 두 개를 리마인드한다.

CLAUDE.md 규범 중 판단이 필요 없는 것만 강제 대상:
- notebooks/*.ipynb 수정 → nbconvert 재실행 리마인드
- src/**/*.py 수정 → pytest 리마인드

exit 2 + stderr = 도구는 이미 실행됐고, 메시지가 Claude에게 피드백으로
전달된다 (사용자 차단 아님). 해당 없는 편집은 exit 0으로 조용히 통과.
"""
import json
import os
import sys


def main() -> int:
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input", {})
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", data.get("cwd", ""))
    rel = os.path.relpath(path, project_dir) if path.startswith(project_dir) else path

    if rel.startswith("notebooks/") and rel.endswith(".ipynb"):
        print(
            f"[hook] 노트북 수정됨 — 출력 셀이 있으면 이 턴 안에 재실행할 것: "
            f"uv run jupyter nbconvert --to notebook --execute --inplace {rel} "
            f"(재실행에 LLM 호출이 들어가면 먼저 사용자와 합의)",
            file=sys.stderr,
        )
        return 2

    if rel.startswith("src/") and rel.endswith(".py"):
        print(
            "[hook] src/ 수정됨 — 이 턴을 마치기 전에 "
            "`uv run pytest tests/ -q`를 실행하고 결과를 보고할 것",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
