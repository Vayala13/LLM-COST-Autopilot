"""Phase 2.4 — load the YAML routing map and resolve a tier to a ModelConfig.

The map lives in configs/routing_map.yaml so tier→model swaps need no code
changes. Registry keys are validated on load.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.providers.registry import MODEL_REGISTRY, ModelConfig

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MAP_PATH = _ROOT / "configs" / "routing_map.yaml"


def _load_routing_raw(path: str | None = None) -> dict[int, dict[str, str]]:
    """Parse routing YAML into {tier: {model, rationale}}.

    Only load project config paths in production callers — path is unconstrained
    for local smoke/tests; do not pass untrusted user paths here.
    """
    map_path = Path(path) if path else DEFAULT_MAP_PATH
    data = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    routing = data.get("routing") or {}
    resolved: dict[int, dict[str, str]] = {}
    for tier_key, entry in routing.items():
        tier = int(tier_key)
        if isinstance(entry, dict):
            model_key = entry["model"]
            rationale = str(entry.get("rationale") or "").strip()
        else:
            model_key = entry
            rationale = ""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"routing_map tier {tier} → {model_key!r} not in MODEL_REGISTRY"
            )
        resolved[tier] = {"model": model_key, "rationale": rationale}
    if set(resolved) != {1, 2, 3}:
        raise ValueError(f"routing_map must define tiers 1/2/3, got {sorted(resolved)}")
    return resolved


def load_routing_map(path: str | None = None) -> dict[int, str]:
    """Return {tier_id: MODEL_REGISTRY key} from the routing YAML.

    Always re-reads the file so YAML edits apply without a process restart
    (Phase 5.2 PUT /v1/routing-config can rely on this).
    """
    return {tier: entry["model"] for tier, entry in _load_routing_raw(path).items()}


def rationale_for_tier(tier: int, path: str | None = None) -> str:
    """Return the human-readable routing rationale for a complexity tier."""
    entries = _load_routing_raw(path)
    if tier not in entries:
        raise ValueError(f"No routing entry for tier {tier}")
    return entries[tier]["rationale"] or f"Routed to tier {tier} model per routing map."


def model_for_tier(tier: int, path: str | None = None) -> ModelConfig:
    """Look up the ModelConfig configured for a complexity tier."""
    mapping = load_routing_map(path)
    if tier not in mapping:
        raise ValueError(f"No routing entry for tier {tier}")
    return MODEL_REGISTRY[mapping[tier]]


def route_prompt(prompt: str, path: str | None = None) -> tuple[int, str, ModelConfig]:
    """Classify a prompt and return (tier, model_key, ModelConfig)."""
    from app.classifier import predict_tier

    tier = predict_tier(prompt)
    mapping = load_routing_map(path)
    model_key = mapping[tier]
    return tier, model_key, MODEL_REGISTRY[model_key]
