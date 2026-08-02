"""Phase 5.1–5.2 — smoke FastAPI endpoints (offline, mocked provider).

Uses FastAPI TestClient + a fake ``send_request`` so no API keys are required.
Asserts: router chooses the model, audit row written, client cannot force model,
config endpoints (models / stats / routing-config) work with a temp map under
``configs/`` and a temp audit DB.

Run:
    PYTHONPATH=. python -m scripts.smoke_api
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.audit.store import fetch_requests, log_completion
from app.providers.registry import MODEL_REGISTRY
from app.providers.response import Response
from app.quality.verifier import prompt_hash
from app.router.map import (
    DEFAULT_MAP_PATH,
    load_routing_map,
    rationale_for_tier,
)

_ROOT = Path(__file__).resolve().parent.parent
_SMOKE_MAP = _ROOT / "configs" / "_smoke_routing_map.yaml"


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


def _make_client(
    tmp: Path,
    *,
    enqueue_fn=None,
    routing_map_path: Path | None = None,
    allow_routing_config_write: bool | None = True,
) -> TestClient:
    db = tmp / "requests.db"
    app = create_app(
        db_path=db,
        routing_map_path=routing_map_path,
        allow_routing_config_write=allow_routing_config_write,
        send_request_fn=_fake_send,
        enqueue_verification_fn=enqueue_fn,
    )
    return TestClient(app)


def _prepare_smoke_map() -> Path:
    """Copy the real routing map into a configs/_smoke_*.yaml file for PUT tests."""
    if _SMOKE_MAP.exists():
        _SMOKE_MAP.unlink()
    shutil.copyfile(DEFAULT_MAP_PATH, _SMOKE_MAP)
    return _SMOKE_MAP


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


def test_get_models(tmp: Path) -> None:
    with _make_client(tmp) as client:
        r = client.get("/v1/models")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "models" in data
        keys = {m["key"] for m in data["models"]}
        assert keys == set(MODEL_REGISTRY)
        for m in data["models"]:
            cfg = MODEL_REGISTRY[m["key"]]
            assert m["provider"] == cfg.provider
            assert m["model_id"] == cfg.model_id
            assert m["cost_per_input_token"] == cfg.cost_per_input_token
            assert m["cost_per_output_token"] == cfg.cost_per_output_token
            assert m["avg_latency_s"] == cfg.avg_latency_s
            assert m["quality_tier"] == cfg.quality_tier
            # No secrets in response.
            blob = str(m).lower()
            assert "api_key" not in blob
            assert "sk-" not in blob
    print(f"OK GET /v1/models ({len(data['models'])} models)")


def test_get_stats_empty_and_populated(tmp: Path) -> None:
    # Fresh DB — do not reuse the completions audit DB from earlier tests.
    stats_tmp = tmp / "stats-only"
    stats_tmp.mkdir()
    with _make_client(stats_tmp) as client:
        r = client.get("/v1/stats")
        assert r.status_code == 200, r.text
        empty = r.json()
        assert empty["empty"] is True
        assert empty["request_count"] == 0
        assert empty["actual_cost_usd"] == 0.0
        assert empty["savings_usd"] == 0.0
        assert empty["cost_reduction_pct"] == 0.0
        assert "prompt" not in empty

        log_completion(
            prompt_hash="abc123",
            complexity_tier=1,
            routed_model="llama-local",
            cost=0.0,
            latency=0.1,
            verifier_quality_score=None,
            escalated=False,
            input_tokens=100,
            output_tokens=50,
            db_path=stats_tmp / "requests.db",
        )

        r2 = client.get("/v1/stats")
        assert r2.status_code == 200, r2.text
        populated = r2.json()
        assert populated["empty"] is False
        assert populated["request_count"] == 1
        assert populated["actual_cost_usd"] == 0.0
        assert populated["gpt4o_cost_usd"] > 0
        assert populated["savings_usd"] > 0
        assert populated["cost_reduction_pct"] > 0
        # Aggregates only — no raw prompt text fields.
        for forbidden in ("prompt", "output_text", "messages"):
            assert forbidden not in populated
    print(
        f"OK GET /v1/stats empty→populated "
        f"reduction={populated['cost_reduction_pct']:.1f}%"
    )


def test_routing_config_get_put(tmp: Path) -> None:
    smoke_map = _prepare_smoke_map()
    try:
        with _make_client(tmp, routing_map_path=smoke_map) as client:
            r = client.get("/v1/routing-config")
            assert r.status_code == 200, r.text
            before = r.json()
            assert set(before["routing"]) == {"1", "2", "3"}
            for tier in ("1", "2", "3"):
                assert before["routing"][tier]["model"] in MODEL_REGISTRY

            # Preserve tier-1 rationale; change tier-1 model to claude-haiku.
            original_t1_rationale = before["routing"]["1"].get("rationale")
            put_body = {
                "routing": {
                    "1": {
                        "model": "claude-haiku",
                        # omit rationale → preserve existing on disk
                    },
                    "2": before["routing"]["2"]["model"],
                    "3": {
                        "model": before["routing"]["3"]["model"],
                        "rationale": "Smoke override for tier 3.",
                    },
                }
            }
            r_put = client.put("/v1/routing-config", json=put_body)
            assert r_put.status_code == 200, r_put.text
            after = r_put.json()
            assert after["routing"]["1"]["model"] == "claude-haiku"
            if original_t1_rationale:
                assert after["routing"]["1"]["rationale"] == original_t1_rationale
            assert after["routing"]["3"]["rationale"] == "Smoke override for tier 3."

            # File re-read without restart.
            disk = load_routing_map(smoke_map)
            assert disk[1] == "claude-haiku"
            assert disk[2] == before["routing"]["2"]["model"]
            assert disk[3] == before["routing"]["3"]["model"]

            # Invalid registry key → 422
            bad = client.put(
                "/v1/routing-config",
                json={
                    "routing": {
                        "1": "not-a-real-model",
                        "2": "gemini-flash",
                        "3": "claude-sonnet",
                    }
                },
            )
            assert bad.status_code == 422, bad.text

            # Missing tier → 422
            missing = client.put(
                "/v1/routing-config",
                json={"routing": {"1": "llama-local", "2": "gemini-flash"}},
            )
            assert missing.status_code == 422, missing.text

            # Extra field → 422
            extra = client.put(
                "/v1/routing-config",
                json={
                    "routing": {
                        "1": "llama-local",
                        "2": "gemini-flash",
                        "3": "claude-sonnet",
                    },
                    "path": "/etc/passwd",
                },
            )
            assert extra.status_code == 422, extra.text
    finally:
        if smoke_map.exists():
            smoke_map.unlink()
    print("OK GET/PUT /v1/routing-config (validate + persist + preserve rationale)")


def test_routing_config_write_gate(tmp: Path) -> None:
    smoke_map = _prepare_smoke_map()
    try:
        with _make_client(
            tmp,
            routing_map_path=smoke_map,
            allow_routing_config_write=False,
        ) as client:
            r = client.put(
                "/v1/routing-config",
                json={
                    "routing": {
                        "1": "llama-local",
                        "2": "gemini-flash",
                        "3": "claude-sonnet",
                    }
                },
            )
            assert r.status_code == 403, r.text
            # GET still works when writes are disabled.
            g = client.get("/v1/routing-config")
            assert g.status_code == 200
    finally:
        if smoke_map.exists():
            smoke_map.unlink()
    print("OK PUT /v1/routing-config gated by ALLOW_ROUTING_CONFIG_WRITE")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="api-smoke-") as tmp_str:
        tmp = Path(tmp_str)
        test_healthz(tmp)
        test_prompt_completion_and_audit(tmp)
        test_messages_schema(tmp)
        test_client_cannot_force_model(tmp)
        test_rejects_empty_and_both_inputs(tmp)
        test_verification_enqueue(tmp)
        test_get_models(tmp)
        test_get_stats_empty_and_populated(tmp)
        test_routing_config_get_put(tmp)
        test_routing_config_write_gate(tmp)
    print("ALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
