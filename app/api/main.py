"""FastAPI application entry — Phase 5.1 completions + Phase 5.2 config.

Run locally (bind localhost; no reload/debug in production defaults)::

    PYTHONPATH=. uvicorn app.api.main:app --host 127.0.0.1 --port 8000

OpenAPI docs: http://127.0.0.1:8000/docs  (OK for local/portfolio demos).

Auth: unauthenticated for local portfolio use. Do not deploy publicly
without adding real authentication. No CORS ``allow_origins=["*"]`` with
credentials — CORS is omitted so browsers only hit same-origin / tools.

PUT /v1/routing-config is gated by ``ALLOW_ROUTING_CONFIG_WRITE`` (default on
for local demos). That flag is not auth — disable writes in non-local deploys.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.completions import run_completion
from app.api.config_routes import register_config_routes
from app.api.schemas import CompletionRequest, CompletionResponse
from app.audit.store import init_db
from app.providers.client import ProviderNotConfigured

logger = logging.getLogger(__name__)


def create_app(
    *,
    db_path: str | Path | None = None,
    routing_map_path: str | Path | None = None,
    allow_routing_config_write: bool | None = None,
    send_request_fn: Any = None,
    enqueue_verification_fn: Any = None,
) -> FastAPI:
    """Build the FastAPI app.

    Defaults are safe for portfolio demos: docs enabled, debug/reload off,
    no wildcard CORS. ``db_path`` / ``routing_map_path`` / injectable callables
    support offline smoke (routing map path must stay under ``configs/``).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ensure audit schema exists; never log secrets here.
        path = init_db(app.state.db_path)
        logger.info("audit DB ready at %s", path)
        yield

    application = FastAPI(
        title="LLM Cost AutoPilot",
        description=(
            "Intelligent LLM routing layer. "
            "`POST /v1/completions` — the router chooses the model; "
            "clients do not. "
            "`GET /v1/models`, `GET /v1/stats`, `GET|PUT /v1/routing-config` "
            "for registry, savings aggregates, and live routing map updates. "
            "Local/portfolio API is unauthenticated."
        ),
        version="0.5.2",
        docs_url="/docs",
        redoc_url="/redoc",
        # debug defaults False — do not enable reload here (CLI flag for local only).
        lifespan=lifespan,
    )
    application.state.db_path = db_path
    application.state.routing_map_path = routing_map_path
    application.state.allow_routing_config_write = allow_routing_config_write
    application.state.send_request_fn = send_request_fn
    application.state.enqueue_verification_fn = enqueue_verification_fn

    @application.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/v1/completions",
        response_model=CompletionResponse,
        summary="Routed chat/text completion",
        response_description="Model output plus routing and cost metadata",
    )
    def create_completion(
        body: CompletionRequest,
        request: Request,
    ) -> CompletionResponse:
        """Route a completion request; client cannot force a model.

        Accepts ``prompt`` **or** ``messages`` (see ``CompletionRequest``).
        Returns output text, selected model, complexity tier, rationale,
        cost, latency, and token counts. Writes one audit row (prompt_hash).
        """
        try:
            return run_completion(
                body,
                db_path=request.app.state.db_path,
                routing_map_path=request.app.state.routing_map_path,
                send_request_fn=request.app.state.send_request_fn,
                enqueue_verification_fn=request.app.state.enqueue_verification_fn,
            )
        except ProviderNotConfigured as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Provider not configured: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("completions failed")
            raise HTTPException(
                status_code=500,
                detail="Internal error processing completion",
            ) from exc

    @application.exception_handler(ProviderNotConfigured)
    async def _provider_unconfigured(
        _request: Request, exc: ProviderNotConfigured
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": f"Provider not configured: {exc}"},
        )

    register_config_routes(application)
    return application


# Module-level app for ``uvicorn app.api.main:app``
app = create_app()
