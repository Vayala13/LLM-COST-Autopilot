"""Phase 5.1 — smoke POST /v1/completions (offline, mocked provider).

Uses FastAPI TestClient + a fake ``send_request`` so no API keys are required.
Asserts: router chooses the model, audit row written, client cannot force model.

Run:
    PYTHONPATH=. python -m scripts.smoke_api
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.audit.store import fetch_requests
from app.providers.registry import MODEL_REGISTRY
from app.providers.response import Response
from app.quality.verifier import prompt_hash
from app.router.map import load_routing_map, rationale_for_tier


def _fake_send(prompt: str, model_config, max_tokens: int = 1024) -> Response:
    """Deterministic provider stub — never hits the network."""
    return Response(
        model_id=model_config.model_id,
        provider=model_config.provider,
        output_text=f"[stub:{model_config.provider}] {prompt[:40]}",
        input_tokens=12,
        output_tokens=8,
        latency_s=0.05,
        cost_usd=0.0001,
    )


def _make_client(tmp: Path, *, enqueue_fn=None) -> TestClient:
    db = tmp / "requests.db"
    app = create_app(
        db_path=db,
        send_request_fn=_fake_send,
        enqueue_verification_fn=enqueue_fn,
    )
    return TestClient(app)


def test_healthz(tmp: Path) -> None:
    with _make_client(tmp) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
    print("OK healthz")


def test_prompt_completion_and_audit(tmp: Path) -> None:
    prompt = "Convert 'March 3, 2026' to YYYY-MM-DD."
    with _make_client(tmp) as client:
        r = client.post("/v1/completions", json={"prompt": prompt})
        assert r.status_code == 200, r.text
        data = r.json()

        assert "output_text" in data and data["output_text"].startswith("[stub:")
        assert data["complexity_tier"] in (1, 2, 3)
        mapping = load_routing_map()
        expected_model = mapping[data["complexity_tier"]]
        assert data["model"] == expected_model
        assert data["model_id"] == MODEL_REGISTRY[expected_model].model_id
        assert data["provider"] == MODEL_REGISTRY[expected_model].provider
        assert data["rationale"] == rationale_for_tier(data["complexity_tier"])
        assert data["cost_usd"] == 0.0001
        assert data["latency_s"] == 0.05
        assert data["input_tokens"] == 12
        assert data["output_tokens"] == 8
        assert data["request_id"] >= 1
        assert data["verification_enqueued"] is False

        rows = fetch_requests(limit=5, db_path=tmp / "requests.db")
        assert len(rows) == 1
        row = rows[0]
        assert row.id == data["request_id"]
        assert row.prompt_hash == prompt_hash(prompt)
        assert row.complexity_tier == data["complexity_tier"]
        assert row.routed_model == data["model"]
        assert row.cost == data["cost_usd"]
        assert row.latency == data["latency_s"]
        assert row.input_tokens == 12
        assert row.output_tokens == 8
        # Never store raw prompt in audit payload.
        assert prompt not in str(row)
        assert "March 3" not in str(row)

    print(
        f"OK prompt completion → T{data['complexity_tier']} "
        f"{data['model']} request_id={data['request_id']}"
    )


def test_messages_schema(tmp: Path) -> None:
    with _make_client(tmp) as client:
        r = client.post(
            "/v1/completions",
            json={
                "messages": [
                    {"role": "system", "content": "You extract dates."},
                    {"role": "user", "content": "Convert March 3, 2026 to ISO."},
                ]
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["model"] in MODEL_REGISTRY
        assert data["complexity_tier"] in (1, 2, 3)
    print(f"OK messages schema → {data['model']}")


def test_client_cannot_force_model(tmp: Path) -> None:
    with _make_client(tmp) as client:
        r = client.post(
            "/v1/completions",
            json={"prompt": "hello world", "model": "claude-sonnet"},
        )
        # extra=forbid → 422; router owns model selection.
        assert r.status_code == 422, r.text
    print("OK client cannot force model (422 on model field)")


def test_rejects_empty_and_both_inputs(tmp: Path) -> None:
    with _make_client(tmp) as client:
        r1 = client.post("/v1/completions", json={})
        assert r1.status_code == 422
        r2 = client.post(
            "/v1/completions",
            json={
                "prompt": "hi",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r2.status_code == 422
        r3 = client.post("/v1/completions", json={"prompt": "   "})
        assert r3.status_code == 422
    print("OK rejects empty / dual input")


def test_verification_enqueue(tmp: Path) -> None:
    enqueue = MagicMock(return_value=MagicMock())
    with _make_client(tmp, enqueue_fn=enqueue) as client:
        r = client.post(
            "/v1/completions",
            json={
                "prompt": "Summarize: batteries are good, cameras are slow.",
                "use_case": "summarization",
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["verification_enqueued"] is True
        assert enqueue.call_count == 1
        args, kwargs = enqueue.call_args
        assert args[2] == "summarization"
        assert kwargs.get("routed_model") == data["model"]
    print("OK verification enqueue (mocked)")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="api-smoke-") as tmp_str:
        tmp = Path(tmp_str)
        test_healthz(tmp)
        test_prompt_completion_and_audit(tmp)
        test_messages_schema(tmp)
        test_client_cannot_force_model(tmp)
        test_rejects_empty_and_both_inputs(tmp)
        test_verification_enqueue(tmp)
    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
