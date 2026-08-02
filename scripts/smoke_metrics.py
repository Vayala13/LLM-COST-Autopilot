"""Phase 4.2 — smoke cost metrics + demo seed (offline, no browser / APIs).

Builds a temp audit DB, seeds synthetic rows, and checks summary / series
aggregates (actual vs GPT-4o, routing, quality, escalation).

Run:
    PYTHONPATH=. python -m scripts.smoke_metrics
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from app.audit.store import count_requests, init_db, log_completion
from app.metrics.cost import (
    FALLBACK_INPUT_TOKENS,
    FALLBACK_OUTPUT_TOKENS,
    GPT4O_COUNTERFACTUAL_NOTE,
    compute_summary,
    cost_by_day,
    cost_by_week,
    escalation_rate_by_day,
    gpt4o_cost_for_tokens,
    quality_score_distribution,
    routing_distribution,
    seed_demo_requests,
)
from app.providers.registry import MODEL_REGISTRY


def test_gpt4o_token_math() -> None:
    cfg = MODEL_REGISTRY["gpt-4o"]
    expected = 1000 * cfg.cost_per_input_token + 500 * cfg.cost_per_output_token
    got = gpt4o_cost_for_tokens(1000, 500)
    assert abs(got - expected) < 1e-12
    print(f"OK gpt4o_cost_for_tokens → {got:.6f}")


def test_seed_and_summary(tmp: Path) -> None:
    db = tmp / "requests.db"
    init_db(db)
    assert count_requests(db_path=db) == 0

    n = seed_demo_requests(db_path=db, days=7, per_day=4)
    assert n == 28, f"expected 28 demo rows, got {n}"
    assert count_requests(db_path=db) == 28

    # Second seed is a no-op when DB non-empty.
    assert seed_demo_requests(db_path=db) == 0

    summary = compute_summary(db_path=db)
    assert summary.request_count == 28
    assert summary.actual_cost_usd >= 0
    assert summary.gpt4o_cost_usd > summary.actual_cost_usd
    assert summary.savings_usd > 0
    assert 0 < summary.savings_pct <= 100
    assert summary.tokens_known_count == 28
    assert summary.tokens_fallback_count == 0
    assert summary.escalation_count >= 1
    assert 0 < summary.escalation_rate < 1
    assert summary.mean_quality_score is not None
    assert GPT4O_COUNTERFACTUAL_NOTE in summary.counterfactual_note

    print(
        f"OK summary requests={summary.request_count} "
        f"actual=${summary.actual_cost_usd:.4f} "
        f"gpt4o=${summary.gpt4o_cost_usd:.4f} "
        f"saved=${summary.savings_usd:.4f} ({summary.savings_pct:.1f}%) "
        f"esc_rate={summary.escalation_rate:.2f}"
    )


def test_series(tmp: Path) -> None:
    db = tmp / "requests.db"
    days = cost_by_day(db_path=db)
    weeks = cost_by_week(db_path=db)
    assert len(days) >= 1
    assert all(r.request_count >= 1 for r in days)
    # Per-day actual may exceed GPT-4o when traffic is mostly claude-sonnet
    # (Sonnet list price > GPT-4o). Overall mix should still save.
    assert sum(r.actual_cost_usd for r in days) < sum(r.gpt4o_cost_usd for r in days)
    assert len(weeks) >= 1

    shares = routing_distribution(db_path=db)
    assert shares
    assert abs(sum(s.share for s in shares) - 1.0) < 1e-9
    models = {s.routed_model for s in shares}
    assert "llama-local" in models
    assert "gemini-flash" in models
    assert "claude-sonnet" in models

    quality = quality_score_distribution(db_path=db)
    assert quality
    assert sum(b.count for b in quality) == compute_summary(db_path=db).scored_count

    esc = escalation_rate_by_day(db_path=db)
    assert esc
    assert any(r.escalation_count > 0 for r in esc)

    print(
        f"OK series days={len(days)} weeks={len(weeks)} "
        f"models={len(shares)} quality_bins={len(quality)} esc_days={len(esc)}"
    )


def test_token_fallback(tmp: Path) -> None:
    db = tmp / "fallback.db"
    init_db(db)
    # Row without tokens — must use documented fallback size.
    log_completion(
        prompt_hash="deadbeefcafebabe",
        complexity_tier=2,
        routed_model="gemini-flash",
        cost=0.001,
        latency=0.4,
        verifier_quality_score=4.0,
        db_path=db,
    )
    summary = compute_summary(db_path=db)
    assert summary.request_count == 1
    assert summary.tokens_fallback_count == 1
    expected = gpt4o_cost_for_tokens(FALLBACK_INPUT_TOKENS, FALLBACK_OUTPUT_TOKENS)
    assert abs(summary.gpt4o_cost_usd - expected) < 1e-12
    print(f"OK token fallback gpt4o=${summary.gpt4o_cost_usd:.6f}")


def test_no_raw_prompts_in_metrics(tmp: Path) -> None:
    db = tmp / "requests.db"
    summary = compute_summary(db_path=db)
    blob = str(summary)
    assert "Extract" not in blob
    assert "Summarize" not in blob
    shares = routing_distribution(db_path=db)
    assert all(" " not in s.routed_model or s.routed_model.count(" ") < 3 for s in shares)
    print("OK metrics expose aggregates only")


def main() -> int:
    test_gpt4o_token_math()
    with tempfile.TemporaryDirectory(prefix="metrics-smoke-") as tmp_str:
        tmp = Path(tmp_str)
        test_seed_and_summary(tmp)
        test_series(tmp)
        test_token_fallback(tmp)
        test_no_raw_prompts_in_metrics(tmp)
    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
