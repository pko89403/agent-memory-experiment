"""Central paths & pins for the memlab experiment harness.

The original MemoryOS repo is NOT part of this repository — only this pinned
commit SHA is. Run ``uv run scripts/fetch_reference.py`` to materialize it
under ``external/``. Freezing the reference at one SHA is what makes the
baseline reproducible: anyone cloning this repo evaluates against exactly
the same upstream code we studied.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- external reference (read-only study material + differential-test oracle) ---
EXTERNAL_DIR = PROJECT_ROOT / "external"
MEMORYOS_DIR = EXTERNAL_DIR / "MemoryOS"
MEMORYOS_REPO_URL = "https://github.com/BAI-LAB/MemoryOS.git"
MEMORYOS_SHA = "1d717060350931af33d1d0dc3d4e50a72c125a48"  # upstream main, 2026-04-28

# --- data ---
# LoCoMo-10 ships inside the reference repo's eval/ directory; we read it
# from there rather than vendoring a copy.
LOCOMO_PATH = MEMORYOS_DIR / "eval" / "locomo10.json"

# --- experiment outputs ---
RUNS_DIR = PROJECT_ROOT / "runs"
