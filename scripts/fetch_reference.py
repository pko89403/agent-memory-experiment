"""Fetch the pinned reference repos into external/.

Usage:
    uv run scripts/fetch_reference.py

Idempotent: clones what's missing, verifies the checked-out commit either way.
The pins (*_SHA) live in src/memlab/config.py — change them there, never here.
"""
import subprocess
import sys

from memlab.config import (
    EXTERNAL_DIR,
    GRAPHITI_DIR,
    GRAPHITI_REPO_URL,
    GRAPHITI_SHA,
    MEMORYOS_DIR,
    MEMORYOS_REPO_URL,
    MEMORYOS_SHA,
)

PINS = [
    (MEMORYOS_REPO_URL, MEMORYOS_DIR, MEMORYOS_SHA),
    (GRAPHITI_REPO_URL, GRAPHITI_DIR, GRAPHITI_SHA),
]


def git(*args: str, cwd=None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def fetch(repo_url: str, repo_dir, sha: str) -> bool:
    if not repo_dir.exists():
        print(f"cloning {repo_url} -> {repo_dir}")
        git("clone", repo_url, str(repo_dir))
        git("-c", "advice.detachedHead=false", "checkout", sha, cwd=repo_dir)

    head = git("rev-parse", "HEAD", cwd=repo_dir)
    if head != sha:
        print(f"ERROR: external/{repo_dir.name} is at {head[:12]}, expected {sha[:12]}.")
        print(f"Fix with: git -C external/{repo_dir.name} checkout {sha}")
        return False

    print(f"ok: external/{repo_dir.name} pinned at {head[:12]}")
    return True


def main() -> int:
    EXTERNAL_DIR.mkdir(exist_ok=True)
    results = [fetch(*pin) for pin in PINS]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
