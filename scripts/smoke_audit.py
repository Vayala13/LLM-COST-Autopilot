"""Phase 4.1 — smoke SQLite request audit trail (offline, no APIs).

Inserts a few synthetic completion rows into a temp DB, reads them back,
and checks required audit fields.

Run:
    python -m scripts.smoke_audit
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from app.audit.store import (
    fetch_requests,
    init_db,
    log_completion,
    log_request,
    row_to_dict,
)
from app.quality.verifier import prompt_hash


def test_init_and_schema(tmp: Path) -> None:
    db = tmp / "requests.db"
    path = init_db(db)
    assert path == db
    assert db.is_file()
    print(f"OK init_db → {db}")


def test_log_completion_roundtrip(tmp: Path) -> None:
    db = tmp / "requests.db"
    init_db(db)

    samples = [
        {
            "prompt": "Extract the name from: Alice works at Acme.",
            "complexity_tier": 1,
            "routed_model": "llama-local",
            "cost": 0.0,
            "latency": 0.12,
            "verifier_quality_score": 1.0,
            "escalated": False,
            "use_case": "extraction",
        },
        {
            "prompt": "Summarize this article in three bullets.",
            "complexity_tier": 2,
            "routed_model": "gemini-flash",
            "cost": 0.0004,
            "latency": 0.45,
            "verifier_quality_score": 4.0,
            "escalated": True,
            "escalation_model": "claude-sonnet",
            "cost_delta": 0.008,
            "use_case": "summarization",
        },
        {
            "prompt": "Argue both sides of a policy tradeoff with nuance.",
            "complexity_tier": 3,
            "routed_model": "claude-sonnet",
            "cost": 0.012,
            "latency": 1.8,
            "verifier_quality_score": None,  # verify not finished yet
            "escalated": False,
            "use_case": "reasoning",
        },
    ]

    ids: list[int] = []
    hashes: list[str] = []
    for s in samples:
        ph = prompt_hash(s["prompt"])
        hashes.append(ph)
        # Call-site helper — same API future FastAPI will use.
        row_id = log_completion(
            prompt_hash=ph,
            complexity_tier=s["complexity_tier"],
            routed_model=s["routed_model"],
            cost=s["cost"],
            latency=s["latency"],
            verifier_quality_score=s["verifier_quality_score"],
            escalated=s["escalated"],
            escalation_model=s.get("escalation_model"),
            cost_delta=s.get("cost_delta"),
            use_case=s.get("use_case"),
            db_path=db,
        )
        assert isinstance(row_id, int) and row_id >= 1
        ids.append(row_id)

    assert len(set(ids)) == 3, "each insert should get a distinct id"

    rows = fetch_requests(limit=10, db_path=db)
    assert len(rows) == 3

    # Newest first.
    by_hash = {r.prompt_hash: r for r in rows}
    assert set(by_hash) == set(hashes)

    r1 = by_hash[hashes[0]]
    assert r1.complexity_tier == 1
    assert r1.routed_model == "llama-local"
    assert r1.cost == 0.0
    assert r1.latency == 0.12
    assert r1.verifier_quality_score == 1.0
    assert r1.escalated is False
    assert r1.escalation_model is None
    assert r1.use_case == "extraction"
    assert r1.timestamp  # non-empty ISO timestamp

    r2 = by_hash[hashes[1]]
    assert r2.escalated is True
    assert r2.escalation_model == "claude-sonnet"
    assert r2.cost_delta == 0.008
    assert r2.verifier_quality_score == 4.0

    r3 = by_hash[hashes[2]]
    assert r3.verifier_quality_score is None
    assert r3.complexity_tier == 3

    # No raw prompt text in the DB payload.
    for r in rows:
        blob = str(row_to_dict(r))
        assert "Extract the name" not in blob
        assert "Summarize this article" not in blob
        assert "Argue both sides" not in blob

    print(f"OK log_completion roundtrip n={len(rows)} ids={ids}")
    for r in rows:
        print(
            f"  id={r.id} tier={r.complexity_tier} model={r.routed_model} "
            f"cost={r.cost} score={r.verifier_quality_score} "
            f"escalated={r.escalated} hash={r.prompt_hash}"
        )


def test_log_request_alias(tmp: Path) -> None:
    db = tmp / "alias.db"
    rid = log_request(
        prompt_hash=prompt_hash("alias check"),
        complexity_tier=2,
        routed_model="gemini-flash",
        cost=0.001,
        latency=0.2,
        db_path=db,
    )
    rows = fetch_requests(limit=1, db_path=db)
    assert len(rows) == 1 and rows[0].id == rid
    print(f"OK log_request alias id={rid}")


def test_rejects_bad_inputs(tmp: Path) -> None:
    db = tmp / "bad.db"
    init_db(db)
    try:
        log_completion(
            prompt_hash="",
            complexity_tier=1,
            routed_model="llama-local",
            cost=0.0,
            latency=0.1,
            db_path=db,
        )
        raise AssertionError("empty prompt_hash should raise")
    except ValueError:
        pass
    try:
        log_completion(
            prompt_hash="abc",
            complexity_tier=9,
            routed_model="llama-local",
            cost=0.0,
            latency=0.1,
            db_path=db,
        )
        raise AssertionError("bad tier should raise")
    except ValueError:
        pass
    print("OK rejects bad inputs")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="audit-smoke-") as tmp_str:
        tmp = Path(tmp_str)
        test_init_and_schema(tmp)
        test_log_completion_roundtrip(tmp)
        test_log_request_alias(tmp)
        test_rejects_bad_inputs(tmp)
    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
