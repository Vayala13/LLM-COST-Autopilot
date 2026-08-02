"""Phase 3.2 — quality verifier: score cheap vs high-tier agreement.

Call ``verify()`` after the user-facing response is ready. Prefer
``enqueue_verification()`` (see ``app.quality.queue``) so verification does
not block the response path.

Scoring uses thresholds from ``configs/quality_thresholds.yaml`` via
``app.quality.thresholds``. High-tier / judge / reference calls go through
``app.providers.client.send_request`` only — no provider SDKs here.

Extraction field coverage: the caller passes ``required_fields``; score is
(fields whose names appear as case-insensitive whole words in the cheap
output) / len(required_fields).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.providers.client import send_request
from app.providers.registry import MODEL_REGISTRY, ModelConfig
from app.providers.response import Response
from app.quality.thresholds import passes_threshold, threshold_for

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FAILURE_LOG = _ROOT / "data" / "routing_failures.jsonl"

SendRequestFn = Callable[[str, ModelConfig], Response]


@dataclass(frozen=True)
class VerificationResult:
    """Structured pass/fail result from one verification job."""

    use_case: str
    metric: str
    score: float
    threshold: float
    comparison: str
    passed: bool
    routing_failure: bool
    routed_model: str
    comparison_model: str | None
    detail: str = ""
    prompt_hash: str = ""


def prompt_hash(prompt: str) -> str:
    """Short SHA-256 of the prompt for logs (never log raw prompt text)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def score_field_coverage(output_text: str, required_fields: list[str]) -> float:
    """Fraction of required field names present in ``output_text``.

    Presence = case-insensitive whole-word match (``\\b``) so short names
    like ``id`` do not match inside ``invalid``. Callers must supply a
    non-empty ``required_fields`` list for extraction jobs.
    """
    if not required_fields:
        raise ValueError("extraction requires a non-empty required_fields list")
    text = output_text.lower()
    present = 0
    for field in required_fields:
        # Word-boundary match avoids substring false positives (security/quality).
        pattern = rf"\b{re.escape(field.lower())}\b"
        if re.search(pattern, text):
            present += 1
    return present / len(required_fields)


def normalize_label(text: str) -> str:
    """Normalize a classification label for exact-agreement comparison."""
    if not text or not text.strip():
        return ""
    first_line = text.strip().splitlines()[0]
    # Strip trailing sentence punctuation so "positive." == "positive".
    cleaned = " ".join(first_line.lower().split()).rstrip(string.punctuation)
    return cleaned


def parse_judge_score(text: str, scale_max: float = 5.0) -> float:
    """Parse a numeric judge score from free-form model output.

    Prefers patterns like ``score: 4`` / ``4/5``; falls back to the *last*
    number in range [1, scale_max] so prose like "scale of 1 to 5 … 4"
    does not latch onto the scale floor.
    """
    if not text or not text.strip():
        raise ValueError("empty judge response; cannot parse score")

    lowered = text.strip().lower()
    patterned = re.search(
        r"(?:score\s*[:=]?\s*)(\d+(?:\.\d+)?)\s*(?:/\s*\d+(?:\.\d+)?)?",
        lowered,
    )
    if patterned:
        value = float(patterned.group(1))
    else:
        slash = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", lowered)
        if slash:
            value = float(slash.group(1))
        else:
            candidates = [
                float(m.group(1))
                for m in re.finditer(r"(\d+(?:\.\d+)?)", lowered)
                if 1.0 <= float(m.group(1)) <= scale_max
            ]
            if not candidates:
                raise ValueError(f"no numeric score in judge output: {text!r:.80}")
            value = candidates[-1]

    if value < 1.0 or value > scale_max:
        raise ValueError(
            f"judge score {value} outside expected range [1, {scale_max}]"
        )
    return value


def _judge_prompt(prompt: str, cheap_output: str, scale_max: float) -> str:
    return (
        "You are an impartial quality judge. Score how well the SUMMARY "
        f"answers the PROMPT on a scale of 1 to {int(scale_max)} "
        f"(1=poor, {int(scale_max)}=excellent).\n"
        f"Reply with ONLY a single number (optionally as N/{int(scale_max)}). "
        "No explanation.\n\n"
        f"PROMPT:\n{prompt}\n\n"
        f"SUMMARY:\n{cheap_output}\n"
    )


def verify(
    prompt: str,
    cheap_output: str,
    use_case: str,
    *,
    routed_model: str = "unknown",
    required_fields: list[str] | None = None,
    send_request_fn: SendRequestFn | None = None,
    thresholds_path: str | None = None,
    failure_log_path: Path | str | None = None,
    feedback_path: Path | str | None = None,
    record_feedback: bool = True,
) -> VerificationResult:
    """Score agreement for a routed response against the use-case threshold.

    For ``summarization`` / ``classification``, calls the configured
    judge/reference model via ``send_request``. For ``extraction``, only
    local field-coverage scoring runs (no API call).

    On ``routing_failure``, also records a classifier feedback example
    (Phase 3.4) while the prompt is still in memory — failure JSONL stays
    hash-only; feedback JSONL holds the labeled prompt (gitignored).
    """
    send = send_request_fn or send_request
    cfg = threshold_for(use_case, path=thresholds_path)
    p_hash = prompt_hash(prompt)
    comparison_model: str | None = None
    detail = ""

    if cfg.metric == "field_coverage":
        fields = required_fields if required_fields is not None else []
        score = score_field_coverage(cheap_output, fields)
        detail = f"present={int(round(score * len(fields)))}/{len(fields)}"
    elif cfg.metric == "llm_judge_score":
        comparison_model = cfg.judge_model
        if not comparison_model:
            raise ValueError(
                f"use_case {use_case!r} metric llm_judge_score requires judge_model"
            )
        scale_max = cfg.scale_max if cfg.scale_max is not None else 5.0
        judge_cfg = MODEL_REGISTRY[comparison_model]
        judge_resp = send(
            _judge_prompt(prompt, cheap_output, scale_max),
            judge_cfg,
        )
        score = parse_judge_score(judge_resp.output_text, scale_max=scale_max)
        # Truncate; never echo the original prompt into logs/detail.
        detail = f"judge_raw={judge_resp.output_text.strip()[:80]!r}"
    elif cfg.metric == "label_agreement":
        comparison_model = cfg.reference_model
        if not comparison_model:
            raise ValueError(
                f"use_case {use_case!r} metric label_agreement requires "
                "reference_model"
            )
        ref_cfg = MODEL_REGISTRY[comparison_model]
        ref_resp = send(prompt, ref_cfg)
        cheap_label = normalize_label(cheap_output)
        ref_label = normalize_label(ref_resp.output_text)
        score = 1.0 if cheap_label == ref_label and cheap_label != "" else 0.0
        detail = f"cheap={cheap_label!r} reference={ref_label!r}"
    else:
        raise ValueError(
            f"unsupported metric {cfg.metric!r} for use_case {use_case!r}"
        )

    passed = passes_threshold(score, cfg)
    result = VerificationResult(
        use_case=cfg.use_case,
        metric=cfg.metric,
        score=score,
        threshold=cfg.threshold,
        comparison=cfg.comparison,
        passed=passed,
        routing_failure=not passed,
        routed_model=routed_model,
        comparison_model=comparison_model,
        detail=detail,
        prompt_hash=p_hash,
    )

    if result.routing_failure:
        _log_routing_failure(result, failure_log_path)
        if record_feedback:
            _record_classifier_feedback(
                prompt,
                routed_model=routed_model,
                use_case=result.use_case,
                prompt_hash=p_hash,
                feedback_path=feedback_path,
            )

    return result


def _record_classifier_feedback(
    prompt: str,
    *,
    routed_model: str,
    use_case: str,
    prompt_hash: str,
    feedback_path: Path | str | None,
) -> None:
    """Best-effort Phase 3.4 feedback write; never raises to callers."""
    # Local import keeps verifier free of a hard classifier dependency cycle.
    try:
        from app.classifier.feedback import record_routing_failure_example
    except ImportError as exc:
        logger.error("classifier feedback import failed: %s", exc)
        return
    try:
        record_routing_failure_example(
            prompt,
            routed_model=routed_model,
            use_case=use_case,
            prompt_hash=prompt_hash,
            feedback_path=feedback_path,
        )
    except Exception as exc:  # noqa: BLE001 — feedback must not break verify
        logger.error("failed to record classifier feedback: %s", exc)


def _log_routing_failure(
    result: VerificationResult,
    failure_log_path: Path | str | None,
) -> None:
    """Append one JSONL record for a failed verification (prompt hashed only)."""
    path = Path(failure_log_path) if failure_log_path else DEFAULT_FAILURE_LOG
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "routing_failure",
        **asdict(result),
    }
    logger.warning(
        "routing_failure use_case=%s metric=%s score=%s %s %s "
        "routed=%s comparison=%s prompt_hash=%s",
        result.use_case,
        result.metric,
        result.score,
        result.comparison,
        result.threshold,
        result.routed_model,
        result.comparison_model,
        result.prompt_hash,
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError as exc:
        logger.error("failed to write routing failure log %s: %s", path, exc)
