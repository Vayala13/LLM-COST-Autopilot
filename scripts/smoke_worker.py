"""Phase 5.3 — offline smoke for the background worker (no Docker required).

Uses a temp ``data/`` tree: empty feedback → retrain skipped; seeded JSONL
lines are counted; audit DB row count via ``init_db`` + ``log_completion``.
Never calls live LLM providers.

Run:
    PYTHONPATH=. python -m scripts.smoke_worker
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from app.audit.store import init_db, log_completion
from app.worker.main import collect_status, tick


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_empty_volume_skips_retrain(tmp: Path) -> None:
    db = tmp / "requests.db"
    init_db(db)
    failure = tmp / "routing_failures.jsonl"
    escalations = tmp / "escalations.jsonl"
    feedback = tmp / "feedback_prompts.jsonl"

    out = tick(
        db_path=db,
        failure_log=failure,
        escalations=escalations,
        feedback_path=feedback,
        retrain_enabled=True,
        min_feedback=1,
    )
    assert out["audit_rows"] == 0
    assert out["routing_failures"] == 0
    assert out["escalations"] == 0
    assert out["feedback_rows"] == 0
    assert str(out["retrain"]).startswith("retrain_skipped_need_")
    print("OK empty volume → retrain skipped")


def test_counts_and_retrain_disabled(tmp: Path) -> None:
    db = tmp / "requests.db"
    init_db(db)
    log_completion(
        prompt_hash="abc123deadbeef00",
        complexity_tier=1,
        routed_model="llama-local",
        cost=0.0,
        latency=0.1,
        verifier_quality_score=None,
        escalated=False,
        db_path=db,
    )
    failure = tmp / "routing_failures.jsonl"
    escalations = tmp / "escalations.jsonl"
    feedback = tmp / "feedback_prompts.jsonl"
    _write_jsonl(failure, [{"prompt_hash": "abc123deadbeef00", "passed": False}])
    _write_jsonl(escalations, [{"prompt_hash": "abc123deadbeef00", "escalated": True}])
    # Feedback present but retrain disabled — must not invoke train.
    _write_jsonl(
        feedback,
        [{"prompt": "Say hi", "tier": 2, "source": "routing_failure"}],
    )

    status = collect_status(
        db_path=db,
        failure_log=failure,
        escalations=escalations,
        feedback_path=feedback,
    )
    assert status["audit_rows"] == 1
    assert status["routing_failures"] == 1
    assert status["escalations"] == 1
    assert status["feedback_rows"] == 1

    out = tick(
        db_path=db,
        failure_log=failure,
        escalations=escalations,
        feedback_path=feedback,
        retrain_enabled=False,
        min_feedback=1,
    )
    assert out["retrain"] == "retrain_skipped_disabled"
    print("OK counts + retrain disabled")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="smoke-worker-") as td:
        tmp = Path(td)
        test_empty_volume_skips_retrain(tmp / "a")
        test_counts_and_retrain_disabled(tmp / "b")
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
