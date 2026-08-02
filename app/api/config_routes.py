"""Phase 5.2 — config endpoints: models, stats, routing-config.

- GET  /v1/models          — registry models + costs (no secrets)
- GET  /v1/stats           — cost savings aggregates from audit DB
- GET  /v1/routing-config  — current tier→model map (+ rationales)
- PUT  /v1/routing-config  — update map; persist under configs/ only

Auth: local/portfolio API remains unauthenticated. PUT is gated by
``ALLOW_ROUTING_CONFIG_WRITE`` (default on). Not a substitute for real auth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    ModelInfo,
    ModelsResponse,
    RoutingConfigResponse,
    RoutingConfigUpdate,
    RoutingTierEntry,
    StatsResponse,
)
from app.metrics.cost import compute_summary
from app.providers.registry import MODEL_REGISTRY
from app.router.map import (
    DEFAULT_MAP_PATH,
    load_routing_entries,
    resolve_routing_map_path,
    save_routing_map,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


def _map_path(request: Request) -> Path:
    """Operator-injected path or default configs/routing_map.yaml — never from client."""
    injected = getattr(request.app.state, "routing_map_path", None)
    if injected is None:
        return DEFAULT_MAP_PATH
    return resolve_routing_map_path(injected)


def _writes_allowed(request: Request) -> bool:
    override = getattr(request.app.state, "allow_routing_config_write", None)
    if override is not None:
        return bool(override)
    import config as app_config

    return bool(app_config.ALLOW_ROUTING_CONFIG_WRITE)


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    summary="List available models and costs",
)
def list_models() -> ModelsResponse:
    """Return MODEL_REGISTRY entries (pricing, latency, quality tier).

    No API keys or secrets — public registry metadata only.
    """
    models = [
        ModelInfo(
            key=key,
            provider=cfg.provider,
            model_id=cfg.model_id,
            cost_per_input_token=cfg.cost_per_input_token,
            cost_per_output_token=cfg.cost_per_output_token,
            avg_latency_s=cfg.avg_latency_s,
            quality_tier=cfg.quality_tier,
        )
        for key, cfg in MODEL_REGISTRY.items()
    ]
    return ModelsResponse(models=models)


@router.get(
    "/v1/stats",
    response_model=StatsResponse,
    summary="Cost savings summary (aggregates only)",
)
def get_stats(request: Request) -> StatsResponse:
    """Aggregate savings vs all-GPT-4o from the audit DB.

    Empty DB → zeros / empty=true. Never returns raw prompts.
    """
    db_path = getattr(request.app.state, "db_path", None)
    summary = compute_summary(db_path=db_path)
    empty = summary.request_count == 0
    return StatsResponse(
        request_count=summary.request_count,
        actual_cost_usd=summary.actual_cost_usd,
        gpt4o_cost_usd=summary.gpt4o_cost_usd,
        savings_usd=summary.savings_usd,
        savings_pct=summary.savings_pct,
        cost_reduction_pct=summary.cost_reduction_pct,
        escalation_count=summary.escalation_count,
        escalation_rate=summary.escalation_rate,
        scored_count=summary.scored_count,
        mean_quality_score=summary.mean_quality_score,
        baseline_label=summary.baseline_label,
        empty=empty,
        counterfactual_note=summary.counterfactual_note,
    )


@router.get(
    "/v1/routing-config",
    response_model=RoutingConfigResponse,
    summary="Current tier→model routing map",
)
def get_routing_config(request: Request) -> RoutingConfigResponse:
    """Read the live routing YAML (re-read every call; no restart needed)."""
    path = _map_path(request)
    try:
        entries = load_routing_entries(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="routing map not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    routing = {
        str(tier): RoutingTierEntry(
            model=entries[tier]["model"],
            rationale=entries[tier]["rationale"] or None,
        )
        for tier in (1, 2, 3)
    }
    return RoutingConfigResponse(version=1, routing=routing)


@router.put(
    "/v1/routing-config",
    response_model=RoutingConfigResponse,
    summary="Update tier→model routing map",
)
def put_routing_config(
    body: RoutingConfigUpdate,
    request: Request,
) -> RoutingConfigResponse:
    """Validate registry keys and persist to configs/routing_map.yaml.

    Unauthenticated portfolio API — disabled when
    ``ALLOW_ROUTING_CONFIG_WRITE`` is false. Clients cannot choose the write
    path; only the project configs/ map (or injected smoke path under configs/).
    """
    if not _writes_allowed(request):
        raise HTTPException(
            status_code=403,
            detail=(
                "Routing config writes disabled "
                "(set ALLOW_ROUTING_CONFIG_WRITE=1 for local portfolio demos)"
            ),
        )

    path = _map_path(request)
    mapping, rationales = body.to_mapping()
    try:
        save_routing_map(mapping, path=path, rationales=rationales)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("failed to write routing map")
        raise HTTPException(
            status_code=500, detail="Failed to persist routing config"
        ) from exc

    logger.info(
        "routing-config updated path=%s mapping=%s",
        path,
        {t: mapping[t] for t in (1, 2, 3)},
    )
    return get_routing_config(request)


def register_config_routes(app: Any) -> None:
    """Attach config routes to a FastAPI app."""
    app.include_router(router)
