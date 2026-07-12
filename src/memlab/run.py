"""실험 러너 — 대화를 메소드에 ingest하고, QA를 돌리고, 결과를 runs/에 남긴다.

    uv run python -m memlab.run --limit 1        # 스모크 (대화 1편)
    uv run python -m memlab.run                  # 전량 — 중단해도 재개 가능
    uv run python -m memlab.run --score-only     # 저장된 결과만 다시 채점

러너는 메소드를 모른다: method_factory가 (메소드, LLM 프로바이더)를 만들어
준다. baseline이든 변형이든 팩토리만 갈아 끼우면 같은 러너·같은 채점기로
비교된다 (README 설계 결정 4). CLI 기본 팩토리는 MemoryOS + default_provider().

체크포인트: 대화(sample) 단위. runs/<run_id>/<sample_id>.json이 있으면
건너뛴다 — 일일 한도로 며칠에 걸쳐 끊어 돌려도 이어진다.
에러 격리: 질문 1개의 실패가 런을 죽이지 않는다 — error로 기록하고 계속.
(실패 문항은 채점에서 제외하되 개수를 보고한다.)

아티팩트 = 진실의 원천: meta.json(설정 스냅샷) + 대화별 예측 JSON.
채점은 아티팩트만 읽어 재구성한다 — 지표를 추가해도 재실행이 필요 없다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from memlab.config import EMBEDDING_MODEL, LLM_MODEL, RUNS_DIR
from memlab.data import QA, Category, Sample, load_locomo
from memlab.evaluation import Report, format_report, score
from memlab.llm import LLMProvider
from memlab.methods import MemoryMethod, Utterance

# 대화 하나를 맡을 (메소드, 프로바이더) 한 벌을 만든다.
# 프로바이더를 함께 반환하는 이유: 호출 수·토큰이 실험의 보고 지표라서.
MethodFactory = Callable[[Sample], tuple[MemoryMethod, LLMProvider]]


def run(
    method_factory: MethodFactory,
    meta: dict,
    run_id: str = "baseline",
    limit: int | None = None,
    samples: frozenset[str] | None = None,
) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_meta_once(run_dir, run_id, meta)

    for sample in load_locomo(limit=limit):
        if samples is not None and sample.sample_id not in samples:
            continue  # 병렬 워커 분담 — 남의 몫은 건드리지 않는다
        checkpoint = run_dir / f"{sample.sample_id}.json"
        if checkpoint.exists():
            print(f"[skip] {sample.sample_id} — 체크포인트 있음")
            continue
        try:
            record = _run_sample(sample, method_factory)
        except Exception as error:  # 대화 단위 실패: 기록만 하고 다음 대화로
            print(f"[fail] {sample.sample_id}: {error!r} — 재개 시 재시도됨")
            continue
        checkpoint.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[done] {sample.sample_id} → {checkpoint.name}")

    return run_dir


def score_run(run_dir: Path) -> Report | None:
    """저장된 아티팩트에서 (QA, 예측) 쌍을 재구성해 채점한다."""
    pairs: list[tuple[QA, str]] = []
    n_errors = 0
    for checkpoint in sorted(run_dir.glob("conv-*.json")):
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        for p in record["predictions"]:
            if p.get("error"):
                n_errors += 1
                continue
            qa = QA(
                question=p["question"],
                answer=p["answer"],
                evidence=(),
                category=Category(p["category"]),
                adversarial_answer=p["adversarial_answer"],
            )
            pairs.append((qa, p["prediction"]))

    if not pairs:
        print("채점할 예측이 없다 — 먼저 run을 실행할 것")
        return None
    report = score(pairs)
    print(f"\n채점 대상 {len(pairs)}문항 (실패로 제외: {n_errors})")
    print(format_report(report))
    return report


# ── 내부 ─────────────────────────────────────────────────────────────


def _run_sample(sample: Sample, method_factory: MethodFactory) -> dict:
    method, llm = method_factory(sample)

    started = time.time()
    n_total = sum(len(s.turns) for s in sample.sessions)
    print(f"  [{sample.sample_id}] ingest 시작 ({n_total} 발화)")
    n_utterances = 0
    for session in sample.sessions:  # ① ingest
        for turn in session.turns:
            method.ingest(
                Utterance(turn.speaker, turn.text, session.date_time, turn.blip_caption)
            )
            n_utterances += 1
            if n_utterances % 10 == 0:
                print(
                    f"  [{sample.sample_id}] ingest {n_utterances}/{n_total} 발화, "
                    f"{llm.calls} 호출, {time.time() - started:.0f}s"
                )
    print(
        f"  [{sample.sample_id}] ingest 완료: {n_utterances} 발화 / "
        f"{llm.calls} 호출 / {time.time() - started:.0f}s"
    )

    predictions = []
    for i, qa in enumerate(sample.qa):  # ② QA (문항 단위 에러 격리)
        entry = {
            "question": qa.question,
            "answer": qa.answer,
            "adversarial_answer": qa.adversarial_answer,
            "category": int(qa.category),
        }
        try:
            entry["prediction"] = method.answer(qa.question)
        except Exception as error:
            entry["error"] = repr(error)
            print(f"  [{sample.sample_id}] QA {i} 실패: {error!r}")
        predictions.append(entry)
        if (i + 1) % 20 == 0:
            print(f"  [{sample.sample_id}] QA {i + 1}/{len(sample.qa)}")

    return {
        "sample_id": sample.sample_id,
        "speaker_a": sample.speaker_a,
        "speaker_b": sample.speaker_b,
        "predictions": predictions,
        "usage": {"llm_calls": llm.calls, "total_tokens": llm.total_tokens},
        "duration_s": round(time.time() - started, 1),
    }


def _write_meta_once(run_dir: Path, run_id: str, meta: dict) -> None:
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        return  # 재개 시 최초 설정 스냅샷을 보존한다
    meta = {"run_id": run_id, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ── CLI: 기본 팩토리 = MemoryOS + default_provider() ─────────────────


def memoryos_factory(sample: Sample):
    from memlab.llm import default_provider
    from memlab.methods.memoryos import MemoryOS, MemoryOSConfig

    llm = default_provider()
    method = MemoryOS(llm, sample.speaker_a, sample.speaker_b,
                      config=MemoryOSConfig())
    return method, llm


def zep_run_config():
    """LoCoMo 런의 Zep 설정 — 팩토리와 meta.json이 같은 것을 봐야 한다."""
    from memlab.methods.zep import ZepConfig

    # community 유지보수 제외 (근거는 ZepConfig.update_communities)
    return ZepConfig(update_communities=False)


def zep_factory(sample: Sample):
    from memlab.llm import default_provider
    from memlab.methods.zep import ZepMethod

    llm = default_provider()
    method = ZepMethod(llm, sample.speaker_a, sample.speaker_b,
                       config=zep_run_config())
    return method, llm


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # 백그라운드 실행에서도 로그가 실시간
    parser = argparse.ArgumentParser(description="LoCoMo 벤치마크 러너")
    parser.add_argument("--run-id", default="baseline", help="runs/ 하위 디렉토리 이름")
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개 대화만")
    parser.add_argument("--score-only", action="store_true", help="채점만 다시")
    parser.add_argument("--method", choices=("memoryos", "zep"), default="memoryos")
    parser.add_argument(
        "--samples", default=None,
        help="쉼표로 구분한 sample id만 (병렬 워커 분담용, 예: conv-44,conv-47)",
    )
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    if not args.score_only:
        if args.method == "zep":
            factory, config = zep_factory, zep_run_config()
        else:
            from memlab.methods.memoryos import MemoryOSConfig

            factory, config = memoryos_factory, MemoryOSConfig()
        meta = {
            "method": args.method,
            "llm_model": LLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "config": config.to_dict(),
        }
        run_dir = run(
            factory, meta, run_id=args.run_id, limit=args.limit,
            samples=frozenset(args.samples.split(",")) if args.samples else None,
        )
    score_run(run_dir)


if __name__ == "__main__":
    main()
