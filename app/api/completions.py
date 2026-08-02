"""POST /v1/completions business logic — route, call provider, audit.

Pipeline:
  1. Extract text from prompt | messages
  2. Classifier + routing map → tier, model, rationale
  3. ``send_request`` (unified provider interface only)
  4. ``log_completion`` (prompt_hash; never raw prompt / API keys)
  5. Optionally ``enqueue_verification`` (async, non-blocking)

TODO (escalation hook): sync ``verify`` + ``escalate_if_needed`` on the hot
path would add latency; keep verification async. A future worker (Phase 5.3)
can call ``escalate_if_needed`` after background verify and update the audit
row — do not block this response for escalation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.api.schemas import ChatMessage, CompletionRequest, CompletionResponse
from app.audit.store import log_completion
from app.providers.client import send_request
from app.providers.response import Response
from app.quality.queue import enqueue_verification
from app.quality.verifier import prompt_hash
from app.router.map import rationale_for_tier, route_prompt

logger = logging.getLogger(__name__)

# Soft cap already enforced by Pydantic; used when joining messages.
_MAX_JOINED_CHARS = 100_000


def extract_prompt_text(
    *,
    prompt: str | None,
    messages: list[ChatMessage] | None,
) -> str:
    """Flatten request input into a single string for the classifier/router.

    Messages are joined as ``role: content`` lines (system + user + assistant).
    No URLs are fetched; no shell; no filesystem paths from user input.
    """
    if prompt is not None:
        return prompt
    assert messages is not None  # validated by CompletionRequest
    parts: list[str] = []
    for msg in messages:
        parts.append(f"{msg.role}: {msg.content}")
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("messages produced an empty prompt")
    if len(text) > _MAX_JOINED_CHARS:
        raise ValueError(f"joined messages exceed {_MAX_JOINED_CHARS} characters")
    return text


def run_completion(
    body: CompletionRequest,
    *,
    db_path: str | Path | None = None,
    routing_map_path: str | Path | None = None,
    send_request_fn: Any = None,
    enqueue_verification_fn: Any = None,
) -> CompletionResponse:
    """Execute the full route → provider → audit pipeline.

    ``send_request_fn`` / ``enqueue_verification_fn`` are injectable for
    offline smoke tests (mocked; no live API keys). ``routing_map_path`` is
    operator/smoke injection only (must live under ``configs/``).
    """
    text = extract_prompt_text(prompt=body.prompt, messages=body.messages)
    tier, model_key, model_cfg = route_prompt(text, path=routing_map_path)
    rationale = rationale_for_tier(tier, path=routing_map_path)

    send = send_request_fn or send_request
    # ProviderNotConfigured propagates to the FastAPI layer → HTTP 503.
    response: Response = send(text, model_cfg, max_tokens=body.max_tokens)

    p_hash = prompt_hash(text)
    request_id = log_completion(
        prompt_hash=p_hash,
        complexity_tier=tier,
        routed_model=model_key,
        cost=float(response.cost_usd),
        latency=float(response.latency_s),
        verifier_quality_score=None,  # filled later by async verify (if any)
        escalated=False,
        use_case=body.use_case,
        input_tokens=int(response.input_tokens),
        output_tokens=int(response.output_tokens),
        db_path=db_path,
    )

    verification_enqueued = False
    if body.use_case and body.enqueue_verification:
        enqueue = enqueue_verification_fn or enqueue_verification
        try:
            enqueue(
                text,
                response.output_text,
                body.use_case,
                routed_model=model_key,
                required_fields=body.required_fields,
            )
            verification_enqueued = True
        except RuntimeError as exc:
            # No running event loop (e.g. sync unit call) — skip, do not fail.
            logger.info("verification not enqueued (no event loop): %s", exc)
        except Exception as exc:  # noqa: BLE001 — never fail the user response
            logger.error("verification enqueue failed: %s", exc)

    return CompletionResponse(
        output_text=response.output_text,
        model=model_key,
        model_id=response.model_id,
        provider=response.provider,
        complexity_tier=tier,
        rationale=rationale,
        cost_usd=float(response.cost_usd),
        latency_s=float(response.latency_s),
        input_tokens=int(response.input_tokens),
        output_tokens=int(response.output_tokens),
        request_id=request_id,
        verification_enqueued=verification_enqueued,
    )
