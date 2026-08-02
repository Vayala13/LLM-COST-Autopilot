"""Phase 3.3 — smoke auto-escalation (offline by default).

Covers: escalate-on-failure, skip-when-pass, latency-budget skip.
Uses a mocked ``send_request`` so no API keys are required.

Run:
    python -m scripts.smoke_escalation
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

from app.providers import MODEL_REGISTRY
from app.providers.response import Response
from app.quality.escalation import (
    escalate_if_needed,
    load_escalation_config,
    quality_gap,
)
from app.quality.verifier import VerificationResult, verify


def _fake_response(text: str, model_key: str = "claude-sonnet", cost: float = 0.01) -> Response:
    cfg = MODEL_REGISTRY[model_key]
    return Response(
        model_id=cfg.model_id,
        provider=cfg.provider,
        output_text=text,
        input_tokens=10,
        output_tokens=5,
        latency_s=0.02,
        cost_usd=cost,
    )


_ESCALATED_TEXT = "escalated high-tier answer"


def _mock_send(prompt: str, model_config) -> Response:
    """Deterministic stand-in for send_request (no network)."""
    # Escalation re-runs the *user* prompt (not a judge prompt).
    if "impartial quality judge" in prompt.lower() or "score how well" in prompt.lower():
        return _fake_response("3", "claude-sonnet", cost=0.002)
    return _fake_response(_ESCALATED_TEXT, "claude-sonnet", cost=0.012)


def _failing_extraction(failure_log: Path) -> VerificationResult:
    return verify(
        prompt="Extract name and age from: Ada is 36.",
        cheap_output="name: Ada",
        use_case="extraction",
        routed_model="llama-local",
        required_fields=["name", "age"],
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )


def test_config_loads() -> None:
    cfg = load_escalation_config()
    assert cfg.escalation_model == "claude-sonnet"
    assert cfg.escalation_model in MODEL_REGISTRY
    assert cfg.max_escalation_latency_s == 3.0
    print(
        f"OK escalation config model={cfg.escalation_model} "
        f"max_latency={cfg.max_escalation_latency_s}"
    )


def test_skip_when_pass(tmp: Path) -> None:
    failure_log = tmp / "routing_failures.jsonl"
    esc_log = tmp / "escalations.jsonl"
    ok = verify(
        prompt="Extract name and age",
        cheap_output="name: Ada\nage: 36",
        use_case="extraction",
        routed_model="llama-local",
        required_fields=["name", "age"],
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    assert ok.passed and not ok.routing_failure

    result = escalate_if_needed(
        prompt="Extract name and age",
        cheap_output="name: Ada\nage: 36",
        use_case="extraction",
        routed_model="llama-local",
        verification_result=ok,
        original_cost_usd=0.0,
        send_request_fn=_mock_send,
        escalation_log_path=esc_log,
    )
    assert not result.escalated
    assert result.skipped_reason == "no_routing_failure"
    assert result.output_text == "name: Ada\nage: 36"
    assert result.cost_delta_usd is None
    assert not esc_log.exists()  # passes are not logged
    print("OK skip-when-pass (no JSONL)")


def test_escalate_on_failure(tmp: Path) -> None:
    failure_log = tmp / "routing_failures.jsonl"
    esc_log = tmp / "escalations.jsonl"
    fail = _failing_extraction(failure_log)
    assert fail.routing_failure
    assert quality_gap(fail) > 0

    result = escalate_if_needed(
        prompt="Extract name and age from: Ada is 36.",
        cheap_output="name: Ada",
        use_case="extraction",
        routed_model="llama-local",
        verification_result=fail,
        original_cost_usd=0.001,
        send_request_fn=_mock_send,
        escalation_log_path=esc_log,
    )
    assert result.escalated
    assert result.skipped_reason is None
    assert result.original_model == "llama-local"
    assert result.escalated_model == "claude-sonnet"
    assert result.escalated_output == _ESCALATED_TEXT
    assert result.output_text == _ESCALATED_TEXT
    assert result.cost_delta_usd is not None
    # mock escalate cost 0.012 - original 0.001
    assert abs(result.cost_delta_usd - 0.011) < 1e-9
    assert result.quality_gap == quality_gap(fail)
    assert result.latency_allowed is True

    text = esc_log.read_text(encoding="utf-8")
    assert '"event": "escalation"' in text
    assert "llama-local" in text
    assert "claude-sonnet" in text
    # Prompt text and raw outputs must not appear in the log.
    assert "Extract name" not in text
    assert _ESCALATED_TEXT not in text
    assert "name: Ada" not in text
    print(
        f"OK escalate-on-failure cost_delta={result.cost_delta_usd} "
        f"gap={result.quality_gap} hash={result.prompt_hash}"
    )


def test_latency_skip(tmp: Path) -> None:
    """Tight latency budget → skip re-run but still log would-be escalation."""
    failure_log = tmp / "routing_failures.jsonl"
    esc_log = tmp / "escalations.jsonl"
    cfg_path = tmp / "escalation.yaml"
    # Sonnet avg_latency_s is 2.0 — budget 0.5 forces a skip.
    cfg_path.write_text(
        yaml.dump(
            {
                "version": 1,
                "escalation_model": "claude-sonnet",
                "max_escalation_latency_s": 0.5,
            }
        ),
        encoding="utf-8",
    )

    fail = _failing_extraction(failure_log)
    calls: list[str] = []

    def counting_send(prompt: str, model_config) -> Response:
        calls.append(model_config.model_id)
        return _mock_send(prompt, model_config)

    result = escalate_if_needed(
        prompt="Extract name and age from: Ada is 36.",
        cheap_output="name: Ada",
        use_case="extraction",
        routed_model="llama-local",
        verification_result=fail,
        original_cost_usd=0.0,
        send_request_fn=counting_send,
        escalation_config_path=str(cfg_path),
        escalation_log_path=esc_log,
    )
    assert not result.escalated
    assert result.skipped_reason == "latency_budget"
    assert result.latency_allowed is False
    assert result.output_text == "name: Ada"
    assert result.cost_delta_usd is None
    assert calls == []  # no send_request for escalation

    text = esc_log.read_text(encoding="utf-8")
    assert '"event": "escalation_skipped"' in text
    assert "latency_budget" in text
    assert "Extract name" not in text
    print("OK latency-budget skip (logged, no re-run)")


def test_already_highest(tmp: Path) -> None:
    failure_log = tmp / "routing_failures.jsonl"
    esc_log = tmp / "escalations.jsonl"
    fail = verify(
        prompt="Extract name and age",
        cheap_output="name: Ada",
        use_case="extraction",
        routed_model="claude-sonnet",
        required_fields=["name", "age"],
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    result = escalate_if_needed(
        prompt="Extract name and age",
        cheap_output="name: Ada",
        use_case="extraction",
        routed_model="claude-sonnet",
        verification_result=fail,
        send_request_fn=_mock_send,
        escalation_log_path=esc_log,
    )
    assert not result.escalated
    assert result.skipped_reason == "already_highest"
    assert '"event": "escalation_skipped"' in esc_log.read_text(encoding="utf-8")
    print("OK already-highest skip")


def main() -> int:
    test_config_loads()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        test_skip_when_pass(tmp)
        test_escalate_on_failure(tmp)
        test_latency_skip(tmp)
        test_already_highest(tmp)

    print("\nsmoke_escalation: all offline checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
