"""Phase 3.2 — in-process async queue for verification jobs.

Uses stdlib ``asyncio`` only (no Celery/Redis). After the user-facing
response is ready, call ``enqueue_verification`` so scoring runs in a
background task via ``asyncio.to_thread`` (verify is sync / may call APIs).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Callable

from app.providers.registry import ModelConfig
from app.providers.response import Response
from app.quality.verifier import VerificationResult, verify

logger = logging.getLogger(__name__)

SendRequestFn = Callable[[str, ModelConfig], Response]


def enqueue_verification(
    prompt: str,
    cheap_output: str,
    use_case: str,
    *,
    routed_model: str = "unknown",
    required_fields: list[str] | None = None,
    send_request_fn: SendRequestFn | None = None,
    thresholds_path: str | None = None,
    failure_log_path: Path | str | None = None,
) -> asyncio.Task[VerificationResult]:
    """Schedule ``verify`` on the running event loop; return immediately.

    Requires an active event loop (e.g. FastAPI / ``asyncio.run``). The
    returned task can be awaited later or left to finish in the background.
    """
    loop = asyncio.get_running_loop()

    async def _job() -> VerificationResult:
        return await asyncio.to_thread(
            verify,
            prompt,
            cheap_output,
            use_case,
            routed_model=routed_model,
            required_fields=required_fields,
            send_request_fn=send_request_fn,
            thresholds_path=thresholds_path,
            failure_log_path=failure_log_path,
        )

    task = loop.create_task(_job())
    task.add_done_callback(_on_task_done)
    return task


def _on_task_done(task: asyncio.Task[VerificationResult]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("verification task failed: %s", exc)


async def drain(tasks: Sequence[asyncio.Task[VerificationResult]]) -> list[VerificationResult]:
    """Await a batch of enqueued verification tasks (for tests / shutdown)."""
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))
