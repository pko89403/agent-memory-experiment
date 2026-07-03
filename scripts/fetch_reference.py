"""Fetch the pinned MemoryOS reference repo into external/.

Usage:
    uv run scripts/fetch_reference.py

Idempotent: clones if missing, verifies the checked-out commit either way.
The pin (MEMORYOS_SHA) lives in src/memlab/config.py — change it there,
never here.
"""
import subprocess
import sys

from memlab.config import EXTERNAL_DIR, MEMORYOS_DIR, MEMORYOS_REPO_URL, MEMORYOS_SHA


def git(*args: str, cwd=None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    if not MEMORYOS_DIR.exists():
        EXTERNAL_DIR.mkdir(exist_ok=True)
        print(f"cloning {MEMORYOS_REPO_URL} -> {MEMORYOS_DIR}")
        git("clone", MEMORYOS_REPO_URL, str(MEMORYOS_DIR))
        git("-c", "advice.detachedHead=false", "checkout", MEMORYOS_SHA, cwd=MEMORYOS_DIR)

    head = git("rev-parse", "HEAD", cwd=MEMORYOS_DIR)
    if head != MEMORYOS_SHA:
        print(f"ERROR: external/MemoryOS is at {head[:12]}, expected {MEMORYOS_SHA[:12]}.")
        print("Fix with: git -C external/MemoryOS checkout " + MEMORYOS_SHA)
        return 1

    print(f"ok: external/MemoryOS pinned at {head[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
