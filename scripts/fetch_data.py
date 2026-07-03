"""Fetch LoCoMo-10 from its canonical source (snap-research/locomo) into external/.

Usage:
    uv run scripts/fetch_data.py

Idempotent: downloads if missing, verifies the SHA-256 checksum either way.
The pin (LOCOMO_SHA + LOCOMO_SHA256) lives in src/memlab/config.py.

벤치마크 데이터는 그것을 사용한 논문 repo의 사본이 아니라 **원 출처**에서
받는다. 커밋 SHA로 시점을, 파일 체크섬으로 내용을 이중으로 고정한다.
"""
import hashlib
import sys
import urllib.request

from memlab.config import EXTERNAL_DIR, LOCOMO_PATH, LOCOMO_SHA256, LOCOMO_URL


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not LOCOMO_PATH.exists():
        EXTERNAL_DIR.mkdir(exist_ok=True)
        print(f"downloading {LOCOMO_URL}")
        urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_PATH)

    digest = sha256(LOCOMO_PATH)
    if digest != LOCOMO_SHA256:
        print(f"ERROR: {LOCOMO_PATH} checksum mismatch:")
        print(f"  expected {LOCOMO_SHA256}")
        print(f"  got      {digest}")
        print("Delete the file and re-run to re-download.")
        return 1

    print(f"ok: {LOCOMO_PATH.name} verified ({digest[:12]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
