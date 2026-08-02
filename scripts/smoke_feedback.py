"""Phase 3.4 — smoke classifier feedback + offline retrain (no APIs).

Creates fake routing-failure feedback examples, runs retrain into a temp
dir, and checks that verify() records feedback when a failure is caught.

Run:
    python -m scripts.smoke_feedback
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from app.classifier.feedback import (
    RELABEL_RULE,
    corrected_tier,
    load_feedback,
    merge_training_rows,
    record_routing_failure_example,
)
from app.classifier.model import load_classifier, predict_tier
from app.providers import MODEL_REGISTRY
from app.providers.response import Response
from app.quality.verifier import prompt_hash, verify
from scripts.retrain_from_feedback import retrain


def _fake_response(text: str, model_key: str = "claude-sonnet") -> Response:
    cfg = MODEL_REGISTRY[model_key]
    return Response(
        model_id=cfg.model_id,
        provider=cfg.provider,
        output_text=text,
        input_tokens=10,
        output_tokens=5,
        latency_s=0.02,
        cost_usd=0.001,
    )


def _mock_send(prompt: str, model_config) -> Response:
    if "impartial quality judge" in prompt.lower() or "score how well" in prompt.lower():
        return _fake_response("3", "claude-sonnet")
    return _fake_response("reference-label", "claude-sonnet")


def test_relabel_rule() -> None:
    inferred, corrected = corrected_tier(
        routed_model="llama-local", use_case="extraction"
    )
    assert inferred == 1 and corrected == 2
    inferred, corrected = corrected_tier(
        routed_model="gemini-flash", use_case="summarization"
    )
    assert inferred == 2 and corrected == 3
    inferred, corrected = corrected_tier(
        routed_model="claude-sonnet", use_case="classification"
    )
    assert inferred == 3 and corrected == 3  # cap
    # Unknown model → use_case base then bump.
    inferred, corrected = corrected_tier(
        routed_model="unknown", use_case="extraction"
    )
    assert inferred == 1 and corrected == 2
    print(f"OK relabel rule ({RELABEL_RULE})")


def test_record_and_dedup(tmp: Path) -> None:
    feedback = tmp / "feedback_dedup.jsonl"
    prompt = "Summarize the quarterly report in two sentences with risks."
    ph = prompt_hash(prompt)
    rec = record_routing_failure_example(
        prompt,
        routed_model="llama-local",
        use_case="summarization",
        prompt_hash=ph,
        feedback_path=feedback,
    )
    assert rec is not None
    assert rec["tier"] == 2
    assert rec["prompt"] == prompt
    assert rec["prompt_hash"] == ph
    assert rec["relabel_rule"] == RELABEL_RULE

    # Duplicate hash skipped.
    again = record_routing_failure_example(
        prompt,
        routed_model="llama-local",
        use_case="summarization",
        prompt_hash=ph,
        feedback_path=feedback,
    )
    assert again is None
    rows = load_feedback(feedback)
    assert len(rows) == 1
    print(f"OK record+dedup hash={ph}")


def test_verify_writes_feedback(tmp: Path) -> None:
    failure_log = tmp / "routing_failures.jsonl"
    feedback = tmp / "feedback_from_verify.jsonl"
    prompt = "Extract name and age from: Ada is 36."
    result = verify(
        prompt=prompt,
        cheap_output="name: Ada",
        use_case="extraction",
        routed_model="llama-local",
        required_fields=["name", "age"],
        send_request_fn=_mock_send,
        failure_log_path=failure_log,
        feedback_path=feedback,
    )
    assert result.routing_failure
    rows = load_feedback(feedback)
    assert len(rows) == 1
    assert rows[0]["prompt"] == prompt
    assert rows[0]["tier"] == 2
    # Failure log stays hash-only (no raw prompt).
    fail_text = failure_log.read_text(encoding="utf-8")
    assert prompt_hash(prompt) in fail_text
    assert "Ada is 36" not in fail_text
    print("OK verify→feedback (failure log still hashed)")


def test_merge_feedback_overrides() -> None:
    base = [{"prompt": "hello", "tier": 1}, {"prompt": "world", "tier": 2}]
    fb = [{"prompt": "hello", "tier": 3, "source": "routing_failure"}]
    merged = merge_training_rows(base, fb)
    by_p = {r["prompt"]: r["tier"] for r in merged}
    assert by_p["hello"] == 3
    assert by_p["world"] == 2
    print("OK merge (feedback overrides base)")


def test_retrain_offline(tmp: Path) -> None:
    """Write a few feedback rows and retrain into temp paths (uses real base)."""
    feedback = tmp / "feedback_prompts.jsonl"
    prompts = [
        (
            "Write a multi-step plan to migrate a monolith to services, "
            "with trade-offs and a risk register.",
            "llama-local",
            "summarization",
        ),
        (
            "Argue both sides of remote work for a 200-person eng org "
            "and recommend a policy with justification.",
            "gemini-flash",
            "classification",
        ),
        (
            "Synthesize these conflicting stakeholder notes into an "
            "executive decision memo with open questions.",
            "llama-local",
            "extraction",
        ),
    ]
    for prompt, model, use_case in prompts:
        record_routing_failure_example(
            prompt,
            routed_model=model,
            use_case=use_case,
            prompt_hash=prompt_hash(prompt),
            feedback_path=feedback,
        )
    assert len(load_feedback(feedback)) == 3

    model_out = tmp / "complexity_classifier.joblib"
    metrics_out = tmp / "classifier_metrics.json"
    features_out = tmp / "prompt_features.json"

    # Dry-run first.
    dry = retrain(
        feedback_path=feedback,
        features_path=features_out,
        model_path=model_out,
        metrics_path=metrics_out,
        dry_run=True,
    )
    assert dry is not None and dry.get("dry_run")
    assert not model_out.exists()

    metrics = retrain(
        feedback_path=feedback,
        features_path=features_out,
        model_path=model_out,
        metrics_path=metrics_out,
        dry_run=False,
    )
    assert metrics is not None
    assert metrics["n_feedback"] == 3
    assert metrics["n_examples"] >= 201
    assert model_out.exists()
    assert metrics_out.exists()

    bundle = load_classifier(model_out)
    assert bundle["feature_names"]
    # Smoke: predict_tier runs without error on a feedback prompt.
    tier = predict_tier(prompts[0][0], bundle=bundle)
    assert tier in (1, 2, 3)

    payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert payload["n_feedback"] == 3
    print(
        f"OK retrain offline winner={metrics['winner']} "
        f"acc={metrics['winner_accuracy']:.1%} n={metrics['n_examples']}"
    )


def main() -> int:
    test_relabel_rule()
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        test_record_and_dedup(tmp)
        test_verify_writes_feedback(tmp)
        test_merge_feedback_overrides()
        test_retrain_offline(tmp)

    print("\nsmoke_feedback: all offline checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
