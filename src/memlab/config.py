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

# --- data (canonical source, NOT the MemoryOS vendored copy) ---
# LoCoMo-10 originates from snap-research/locomo (Maharana et al., ACL 2024).
# We fetch it from there at a pinned commit and verify its checksum.
# Note: MemoryOS's eval/locomo10.json is byte-identical (verified 2026-07-03),
# so scores stay comparable — but the benchmark's provenance is the original repo.
LOCOMO_REPO = "snap-research/locomo"
LOCOMO_SHA = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"  # upstream main, pinned
LOCOMO_URL = (
    f"https://raw.githubusercontent.com/{LOCOMO_REPO}/{LOCOMO_SHA}/data/locomo10.json"
)
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_PATH = EXTERNAL_DIR / "locomo10.json"

# --- experiment outputs ---
RUNS_DIR = PROJECT_ROOT / "runs"
