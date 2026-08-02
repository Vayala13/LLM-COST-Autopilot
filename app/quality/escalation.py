"""Phase 3.3 — auto-escalate on routing failure.

Call ``escalate_if_needed`` after ``verify()`` when a cheap-model response
may need a higher-tier re-run. Escalation uses
``app.providers.client.send_request`` only — no provider SDKs here.

Rule (see ``configs/escalation.yaml``):
  On ``verification_result.routing_failure``, re-run the original user
  prompt with ``escalation_model`` (default: tier-3 ``claude-sonnet``).
  Skip the re-run (but still log) when latency budget is exceeded, or
  when the routed model is already the escalation target.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.providers.client import send_request
from app.providers.registry import MODEL_REGISTRY
from app.quality.verifier import SendRequestFn, VerificationResult, prompt_hash

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ESCALATION_PATH = _ROOT / "configs" / "escalation.yaml"
DEFAULT_ESCALATION_LOG = _ROOT / "data" / "escalations.jsonl"


@dataclass(frozen=True)
class EscalationConfig:
    """Swappable escalation target + optional latency budget."""

    escalation_model: str
    max_escalation_latency_s: float | None


@dataclass(frozen=True)
class EscalationResult:
    """Structured outcome of an escalation decision (re-run or skip)."""

    escalated: bool
    skipped_reason: str | None
    original_model: str
    escalated_model: str | None
    original_output: str
    escalated_output: str | None
    cost_delta_usd: float | None
    quality_gap: float
    verification_score: float
    verification_threshold: float
    verification_metric: str
    use_case: str
    latency_allowed: bool
    estimated_escalation_latency_s: float | None
    max_escalation_latency_s: float | None
    prompt_hash: str
    detail: str = ""

    @property
    def output_text(self) -> str:
        """Best available text: escalated output when re-run, else original."""
        if self.escalated and self.escalated_output is not None:
            return self.escalated_output
        return self.original_output


def load_escalation_config(path: str | None = None) -> EscalationConfig:
    """Load escalation YAML; validate registry key and latency budget."""
    cfg_path = Path(path) if path else DEFAULT_ESCALATION_PATH
    # Only load project config paths in production callers — path is
    # unconstrained for local smoke/tests; do not pass untrusted paths.
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("escalation.yaml: top level must be a mapping")

    model_key = data.get("escalation_model")
    if not model_key or not isinstance(model_key, str):
        raise ValueError("escalation.yaml: escalation_model must be a non-empty string")
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"escalation.yaml: escalation_model {model_key!r} not in MODEL_REGISTRY"
        )

    raw_max = data.get("max_escalation_latency_s", None)
    max_latency: float | None
    if raw_max is None:
        max_latency = None
    else:
        try:
            max_latency = float(raw_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "escalation.yaml: max_escalation_latency_s must be a number or null"
            ) from exc
        if max_latency <= 0:
            raise ValueError(
                f"escalation.yaml: max_escalation_latency_s must be > 0, got {max_latency}"
            )

    return EscalationConfig(
        escalation_model=model_key,
        max_escalation_latency_s=max_latency,
    )


def quality_gap(verification: VerificationResult) -> float:
    """How far the score sits below the pass bar (positive ⇒ shortfall).

    For failing scores under ``>`` / ``>=`` / ``==`` thresholds this is
    ``threshold - score``. Clamped at 0 when already passing.
    """
    gap = verification.threshold - verification.score
    return gap if gap > 0 else 0.0


def escalate_if_needed(
    prompt: str,
    cheap_output: str,
    use_case: str,
    routed_model: str,
    verification_result: VerificationResult,
    *,
    original_cost_usd: float = 0.0,
    send_request_fn: SendRequestFn | None = None,
    escalation_config_path: str | None = None,
    escalation_log_path: Path | str | None = None,
) -> EscalationResult:
    """Re-run with the higher-tier model when verification failed.

    Returns a structured ``EscalationResult``. On success, ``output_text``
    is the escalated response; otherwise it stays the cheap output.
    Always safe to call after ``verify`` — no-ops when not a routing failure.
    """
    p_hash = verification_result.prompt_hash or prompt_hash(prompt)
    gap = quality_gap(verification_result)
    cfg = load_escalation_config(escalation_config_path)
    target = cfg.escalation_model
    target_avg = MODEL_REGISTRY[target].avg_latency_s
    latency_allowed = (
        True
        if cfg.max_escalation_latency_s is None
        else target_avg <= cfg.max_escalation_latency_s
    )

    base_kwargs: dict[str, Any] = {
        "original_model": routed_model,
        "escalated_model": target,
        "original_output": cheap_output,
        "quality_gap": gap,
        "verification_score": verification_result.score,
        "verification_threshold": verification_result.threshold,
        "verification_metric": verification_result.metric,
        "use_case": use_case,
        "latency_allowed": latency_allowed,
        "estimated_escalation_latency_s": target_avg,
        "max_escalation_latency_s": cfg.max_escalation_latency_s,
        "prompt_hash": p_hash,
    }

    # 1) Pass — no escalation.
    if not verification_result.routing_failure:
        result = EscalationResult(
            escalated=False,
            skipped_reason="no_routing_failure",
            escalated_output=None,
            cost_delta_usd=None,
            detail="verification passed; no escalation",
            **base_kwargs,
        )
        # Do not log passes — escalation log is failure/escalation only.
        return result

    # 2) Already on the escalation target — nowhere higher to go.
    if routed_model == target:
        result = EscalationResult(
            escalated=False,
            skipped_reason="already_highest",
            escalated_output=None,
            cost_delta_usd=None,
            detail=f"routed_model already {target}; skip re-run",
            **base_kwargs,
        )
        _log_escalation(result, escalation_log_path, event="escalation_skipped")
        return result

    # 3) Latency budget exceeded — log would-be, skip re-run.
    if not latency_allowed:
        result = EscalationResult(
            escalated=False,
            skipped_reason="latency_budget",
            escalated_output=None,
            cost_delta_usd=None,
            detail=(
                f"avg_latency_s={target_avg} exceeds "
                f"max_escalation_latency_s={cfg.max_escalation_latency_s}; "
                "skip re-run"
            ),
            **base_kwargs,
        )
        _log_escalation(result, escalation_log_path, event="escalation_skipped")
        return result

    # 4) Re-run original prompt on the higher-tier model.
    send = send_request_fn or send_request
    resp = send(prompt, MODEL_REGISTRY[target])
    cost_delta = float(resp.cost_usd) - float(original_cost_usd)
    result = EscalationResult(
        escalated=True,
        skipped_reason=None,
        escalated_output=resp.output_text,
        cost_delta_usd=cost_delta,
        detail=(
            f"escalated {routed_model} → {target}; "
            f"cost_delta_usd={cost_delta:.6f}; "
            f"quality_gap={gap:.4f}; "
            f"latency_s={resp.latency_s:.3f}"
        ),
        **base_kwargs,
    )
    _log_escalation(result, escalation_log_path, event="escalation")
    return result


def _log_escalation(
    result: EscalationResult,
    escalation_log_path: Path | str | None,
    *,
    event: str,
) -> None:
    """Append one JSONL record (prompt hashed only; never raw prompt / keys)."""
    path = Path(escalation_log_path) if escalation_log_path else DEFAULT_ESCALATION_LOG
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **asdict(result),
    }
    # Drop full output bodies from the on-disk log — keep hashes + metrics.
    # Callers still receive outputs on EscalationResult for the response path.
    record.pop("original_output", None)
    record.pop("escalated_output", None)

    logger.warning(
        "%s use_case=%s original=%s escalated_model=%s escalated=%s "
        "skipped=%s cost_delta=%s quality_gap=%s score=%s threshold=%s "
        "latency_allowed=%s prompt_hash=%s",
        event,
        result.use_case,
        result.original_model,
        result.escalated_model,
        result.escalated,
        result.skipped_reason,
        result.cost_delta_usd,
        result.quality_gap,
        result.verification_score,
        result.verification_threshold,
        result.latency_allowed,
        result.prompt_hash,
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError as exc:
        logger.error("failed to write escalation log %s: %s", path, exc)
