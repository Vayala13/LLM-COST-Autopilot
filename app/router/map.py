"""Phase 2.4 — load the YAML routing map and resolve a tier to a ModelConfig.

The map lives in configs/routing_map.yaml so tier→model swaps need no code
changes. Registry keys are validated on load.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from app.providers.registry import MODEL_REGISTRY, ModelConfig

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MAP_PATH = _ROOT / "configs" / "routing_map.yaml"


@lru_cache(maxsize=1)
def load_routing_map(path: str | None = None) -> dict[int, str]:
    """Return {tier_id: MODEL_REGISTRY key} from the routing YAML."""
    map_path = Path(path) if path else DEFAULT_MAP_PATH
    data = yaml.safe_load(map_path.read_text())
    routing = data.get("routing") or {}
    resolved: dict[int, str] = {}
    for tier_key, entry in routing.items():
        tier = int(tier_key)
        model_key = entry["model"] if isinstance(entry, dict) else entry
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"routing_map tier {tier} → {model_key!r} not in MODEL_REGISTRY"
            )
        resolved[tier] = model_key
    if set(resolved) != {1, 2, 3}:
        raise ValueError(f"routing_map must define tiers 1/2/3, got {sorted(resolved)}")
    return resolved


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
