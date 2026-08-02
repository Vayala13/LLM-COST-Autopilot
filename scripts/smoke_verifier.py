"""Phase 3.2 — smoke the async verifier (offline by default).

Runs scoring helpers + verify/enqueue with a mocked ``send_request`` so no
API keys are required. Optional ``--live`` attempts a real summarization
judge call and skips gracefully when the provider is unconfigured.

Run:
    python -m scripts.smoke_verifier
    python -m scripts.smoke_verifier --live
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

from app.providers import MODEL_REGISTRY, ProviderNotConfigured, send_request
from app.providers.response import Response
from app.quality import (
    drain,
    enqueue_verification,
    passes_threshold,
    threshold_for,
    verify,
)
from app.quality.verifier import (
    normalize_label,
    parse_judge_score,
    score_field_coverage,
)


def _fake_response(text: str, model_key: str = "claude-sonnet") -> Response:
    cfg = MODEL_REGISTRY[model_key]
    return Response(
        model_id=cfg.model_id,
        provider=cfg.provider,
        output_text=text,
        input_tokens=10,
        output_tokens=5,
        latency_s=0.01,
        cost_usd=0.0,
    )


_MOCK_JUDGE_SCORE = "5"


def _mock_send(prompt: str, model_config) -> Response:
    """Deterministic stand-in for send_request (no network)."""
    # Judge prompts ask for a score; classification / other get a label.
    if "impartial quality judge" in prompt.lower() or "score how well" in prompt.lower():
        return _fake_response(_MOCK_JUDGE_SCORE, "claude-sonnet")
    return _fake_response("positive", "claude-sonnet")


def test_helpers() -> None:
    assert score_field_coverage("name: Ada\nage: 36", ["name", "age"]) == 1.0
    assert score_field_coverage("name: Ada", ["name", "age"]) == 0.5
    # Whole-word: "id" must not match inside "invalid"
    assert score_field_coverage("status: invalid", ["id"]) == 0.0
    assert score_field_coverage("id: 7\nstatus: invalid", ["id"]) == 1.0
    assert normalize_label("  Positive \nextra") == "positive"
    assert normalize_label("positive.") == "positive"
    assert parse_judge_score("Score: 5") == 5.0
    assert parse_judge_score("4/5") == 4.0
    # Last in-range number wins over scale prose ("1 to 5")
    assert parse_judge_score("on a scale of 1 to 5 I rate this 4") == 4.0

    t_sum = threshold_for("summarization")
    assert passes_threshold(5.0, t_sum) is True
    assert passes_threshold(4.0, t_sum) is False  # comparison is >

    t_ext = threshold_for("extraction")
    assert passes_threshold(1.0, t_ext) is True
    assert passes_threshold(0.99, t_ext) is False

    t_cls = threshold_for("classification")
    assert passes_threshold(1.0, t_cls) is True
    assert passes_threshold(0.0, t_cls) is False
    print("OK helpers + passes_threshold")


def test_verify_offline(failure_log: Path) -> None:
    # Extraction: full coverage → pass
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
    assert ok.score == 1.0
    print(f"OK extraction pass score={ok.score} hash={ok.prompt_hash}")

    # Extraction: missing field → routing failure + JSONL log
    fail = verify(
        prompt="Extract name and age",
        cheap_output="name: Ada",
        use_case="extraction",
        routed_model="llama-local",
        required_fields=["name", "age"],
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    assert not fail.passed and fail.routing_failure
    assert fail.score == 0.5
    assert failure_log.is_file()
    lines = failure_log.read_text(encoding="utf-8").strip().splitlines()
    assert any('"event": "routing_failure"' in line for line in lines)
    assert "Extract name" not in failure_log.read_text(encoding="utf-8")
    print(f"OK extraction fail logged score={fail.score} lines={len(lines)}")

    # Summarization: mocked judge returns 5 → pass (> 4)
    global _MOCK_JUDGE_SCORE
    _MOCK_JUDGE_SCORE = "5"
    summ = verify(
        prompt="Summarize the article",
        cheap_output="A short accurate summary.",
        use_case="summarization",
        routed_model="gemini-flash",
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    assert summ.passed and summ.score == 5.0
    assert summ.comparison_model == "claude-sonnet"
    print(f"OK summarization pass score={summ.score} judge={summ.comparison_model}")

    # Summarization: mocked judge returns 3 → routing failure
    _MOCK_JUDGE_SCORE = "3"
    summ_fail = verify(
        prompt="Summarize the article",
        cheap_output="meh",
        use_case="summarization",
        routed_model="gemini-flash",
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    assert not summ_fail.passed and summ_fail.routing_failure
    assert summ_fail.score == 3.0
    _MOCK_JUDGE_SCORE = "5"
    print(f"OK summarization fail score={summ_fail.score}")

    # Classification: cheap matches mocked reference → pass
    cls = verify(
        prompt="Classify sentiment: I love this.",
        cheap_output="positive",
        use_case="classification",
        routed_model="gemini-flash",
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    assert cls.passed and cls.score == 1.0
    print(f"OK classification pass score={cls.score} ref={cls.comparison_model}")

    # Classification: mismatch → routing failure
    cls_fail = verify(
        prompt="Classify sentiment: I love this.",
        cheap_output="negative",
        use_case="classification",
        routed_model="gemini-flash",
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    assert not cls_fail.passed and cls_fail.routing_failure
    assert cls_fail.score == 0.0
    print(f"OK classification fail score={cls_fail.score}")


async def test_enqueue(failure_log: Path) -> None:
    task = enqueue_verification(
        prompt="Extract fields",
        cheap_output="email: a@b.com",
        use_case="extraction",
        routed_model="llama-local",
        required_fields=["email"],
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        record_feedback=False,
    )
    results = await drain([task])
    assert len(results) == 1 and results[0].passed
    print(f"OK enqueue+drain passed={results[0].passed}")


def test_live_optional() -> None:
    """Optional live judge call; skip if Anthropic (or judge provider) missing."""
    judge_key = threshold_for("summarization").judge_model
    assert judge_key is not None
    cfg = MODEL_REGISTRY[judge_key]
    try:
        result = verify(
            prompt="Summarize in one sentence: Cats sleep most of the day.",
            cheap_output="Cats spend most of the day sleeping.",
            use_case="summarization",
            routed_model="gemini-flash",
            send_request_fn=send_request,
            record_feedback=False,
        )
    except ProviderNotConfigured as exc:
        print(f"SKIP live: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - smoke: report and continue
        print(f"SKIP live: provider error: {exc}")
        return
    print(
        f"OK live summarization passed={result.passed} score={result.score} "
        f"judge={cfg.model_id} failure={result.routing_failure}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="also attempt a live judge call (skips if unconfigured)",
    )
    args = parser.parse_args()

    test_helpers()
    with tempfile.TemporaryDirectory() as tmp:
        failure_log = Path(tmp) / "routing_failures.jsonl"
        test_verify_offline(failure_log)
        asyncio.run(test_enqueue(failure_log))

    if args.live:
        test_live_optional()
    else:
        print("SKIP live (pass --live to attempt)")

    print("\nsmoke_verifier: all offline checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
