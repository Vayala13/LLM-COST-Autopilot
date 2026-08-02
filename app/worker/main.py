"""Phase 5.3 — background worker process (compose ``worker`` service).

Honest architecture
-------------------
The API's verification path uses an **in-process** ``asyncio`` queue
(``app.quality.queue.enqueue_verification``). That queue cannot cross
container boundaries. Quality verification therefore runs inside the API
process after ``POST /v1/completions`` when ``use_case`` +
``enqueue_verification`` are set.

This worker shares the ``data/`` volume with the API and:

1. Reports counts for routing-failure / escalation / feedback JSONL files
   and the SQLite audit DB (``requests.db``).
2. Optionally runs ``scripts.retrain_from_feedback.retrain`` when enough
   feedback rows have accumulated (classifier flywheel).

It never expects to receive or drain the API's in-memory verify jobs.
Offline-safe by default — retrain needs no LLM API keys.

Env (see ``.env.example``)::

    WORKER_INTERVAL_S   — loop sleep seconds (default 300)
    WORKER_RETRAIN      — ``1``/``0`` enable retrain (default 1)
    WORKER_MIN_FEEDBACK — min feedback rows before retrain (default 1)
    WORKER_ONCE         — if set, run one tick then exit (smoke / CI)

Run::

    PYTHONPATH=. python -m app.worker.main
    WORKER_ONCE=1 PYTHONPATH=. python -m app.worker.main
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

from app.classifier.feedback import DEFAULT_FEEDBACK_PATH, load_feedback
from app.quality.escalation import DEFAULT_ESCALATION_LOG
from app.quality.verifier import DEFAULT_FAILURE_LOG

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _ROOT / "data" / "requests.db"

# Process-local: skip retrain until feedback_prompts.jsonl grows.
_last_retrain_feedback_count: int | None = None


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _count_audit_rows(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM requests").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error as exc:
        logger.warning("audit DB unreadable at %s: %s", db_path, exc)
        return 0


def collect_status(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    failure_log: Path = DEFAULT_FAILURE_LOG,
    escalations: Path = DEFAULT_ESCALATION_LOG,
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
) -> dict[str, int]:
    """Aggregate counts from the shared data volume (no raw prompts)."""
    feedback_rows = len(load_feedback(feedback_path))
    return {
        "audit_rows": _count_audit_rows(db_path),
        "routing_failures": _count_jsonl_lines(failure_log),
        "escalations": _count_jsonl_lines(escalations),
        "feedback_rows": feedback_rows,
    }


def maybe_retrain(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
    min_feedback: int = 1,
    enabled: bool = True,
) -> str:
    """Run classifier retrain when feedback grows past the last run.

    Returns a short status string. Does not retrain on every tick once a
    feedback file already exists — only when the row count increases
    (or on the first successful run in this process).
    """
    global _last_retrain_feedback_count

    if not enabled:
        return "retrain_skipped_disabled"
    feedback = load_feedback(feedback_path)
    n = len(feedback)
    if n < min_feedback:
        return f"retrain_skipped_need_{min_feedback}_have_{n}"
    if (
        _last_retrain_feedback_count is not None
        and n <= _last_retrain_feedback_count
    ):
        return f"retrain_skipped_unchanged_{n}"

    # Import lazily so ``WORKER_RETRAIN=0`` paths stay cheap / offline.
    from scripts.retrain_from_feedback import retrain

    result = retrain(feedback_path=feedback_path, force=False)
    if result is None:
        return "retrain_noop"
    _last_retrain_feedback_count = n
    if isinstance(result, dict) and result.get("dry_run"):
        return "retrain_dry_run"
    return "retrain_ok"


def tick(
    *,
    db_path: Path | None = None,
    feedback_path: Path | None = None,
    failure_log: Path | None = None,
    escalations: Path | None = None,
    retrain_enabled: bool | None = None,
    min_feedback: int | None = None,
) -> dict[str, object]:
    """One worker loop: status snapshot + optional retrain."""
    status = collect_status(
        db_path=db_path or DEFAULT_DB_PATH,
        failure_log=failure_log or DEFAULT_FAILURE_LOG,
        escalations=escalations or DEFAULT_ESCALATION_LOG,
        feedback_path=feedback_path or DEFAULT_FEEDBACK_PATH,
    )
    retrain_msg = maybe_retrain(
        feedback_path=feedback_path or DEFAULT_FEEDBACK_PATH,
        min_feedback=(
            min_feedback
            if min_feedback is not None
            else _env_int("WORKER_MIN_FEEDBACK", 1)
        ),
        enabled=(
            retrain_enabled
            if retrain_enabled is not None
            else _env_bool("WORKER_RETRAIN", True)
        ),
    )
    out: dict[str, object] = {**status, "retrain": retrain_msg}
    logger.info(
        "worker tick audit=%s failures=%s escalations=%s feedback=%s %s",
        status["audit_rows"],
        status["routing_failures"],
        status["escalations"],
        status["feedback_rows"],
        retrain_msg,
    )
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [worker] %(message)s",
    )
    interval = max(1, _env_int("WORKER_INTERVAL_S", 300))
    once = _env_bool("WORKER_ONCE", False)

    logger.info(
        "starting worker interval=%ss retrain=%s min_feedback=%s "
        "(API verify stays in-process — this worker watches data/ only)",
        interval,
        _env_bool("WORKER_RETRAIN", True),
        _env_int("WORKER_MIN_FEEDBACK", 1),
    )

    while True:
        try:
            tick()
        except Exception:  # noqa: BLE001 — keep the loop alive
            logger.exception("worker tick failed")
        if once:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
