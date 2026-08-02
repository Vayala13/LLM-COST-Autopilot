"""Phase 2.4 / 5.2 — load/save the YAML routing map and resolve tiers.

The map lives in configs/routing_map.yaml so tier→model swaps need no code
changes. Registry keys are validated on load and on save.

Phase 5.2 PUT /v1/routing-config persists here. Clients never supply a file
path — callers use ``DEFAULT_MAP_PATH`` or an operator-injected path (smoke).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from app.providers.registry import MODEL_REGISTRY, ModelConfig

_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = (_ROOT / "configs").resolve()
DEFAULT_MAP_PATH = (_CONFIGS_DIR / "routing_map.yaml").resolve()

_HEADER = (
    "# Phase 2.4 — complexity tier → model routing map.\n"
    "#\n"
    "# Values are keys in app.providers.registry.MODEL_REGISTRY. Swap a model here\n"
    "# without changing code. Updated via PUT /v1/routing-config (Phase 5.2) or\n"
    "# by editing this file. load_routing_map() re-reads on every call.\n"
)


def resolve_routing_map_path(path: str | Path | None = None) -> Path:
    """Resolve a routing-map path.

    Production default is ``configs/routing_map.yaml``. When a path is given
    (operator / smoke injection only — never from client input), it must be a
    ``.yaml``/``.yml`` file under the project ``configs/`` directory.
    """
    if path is None:
        return DEFAULT_MAP_PATH
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("routing map path must be a .yaml or .yml file")
    try:
        p.relative_to(_CONFIGS_DIR)
    except ValueError as exc:
        raise ValueError(
            f"routing map path must be under {_CONFIGS_DIR}"
        ) from exc
    return p


def _load_routing_raw(path: str | Path | None = None) -> dict[int, dict[str, str]]:
    """Parse routing YAML into {tier: {model, rationale}}.

    Path is unconstrained only when already resolved by the caller for
    smoke/tests under configs/; do not pass untrusted user paths here.
    """
    map_path = Path(path) if path is not None else DEFAULT_MAP_PATH
    data = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
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


def load_routing_entries(
    path: str | Path | None = None,
) -> dict[int, dict[str, str]]:
    """Return {tier: {model, rationale}} (always re-reads the YAML file)."""
    return _load_routing_raw(path)


def load_routing_map(path: str | Path | None = None) -> dict[int, str]:
    """Return {tier_id: MODEL_REGISTRY key} from the routing YAML.

    Always re-reads the file so YAML edits apply without a process restart
    (Phase 5.2 PUT /v1/routing-config can rely on this).
    """
    return {tier: entry["model"] for tier, entry in _load_routing_raw(path).items()}


def rationale_for_tier(tier: int, path: str | Path | None = None) -> str:
    """Return the human-readable routing rationale for a complexity tier."""
    entries = _load_routing_raw(path)
    if tier not in entries:
        raise ValueError(f"No routing entry for tier {tier}")
    return entries[tier]["rationale"] or f"Routed to tier {tier} model per routing map."


def save_routing_map(
    mapping: Mapping[int, str],
    *,
    path: str | Path | None = None,
    rationales: Mapping[int, str] | None = None,
    version: int = 1,
) -> Path:
    """Persist tier→model mapping to YAML under ``configs/``.

    Validates every model key against ``MODEL_REGISTRY``. Preserves existing
    rationales when the caller omits one for a tier. Never accepts a
    client-supplied path — ``path`` is operator/smoke injection only and must
    resolve under ``configs/``.
    """
    map_path = resolve_routing_map_path(path)
    if set(mapping) != {1, 2, 3}:
        raise ValueError(f"routing map must define tiers 1/2/3, got {sorted(mapping)}")

    existing: dict[int, dict[str, str]] = {}
    if map_path.is_file():
        try:
            existing = _load_routing_raw(map_path)
        except (ValueError, KeyError, TypeError, OSError):
            existing = {}

    routing_out: dict[int, dict[str, str]] = {}
    for tier in (1, 2, 3):
        model_key = mapping[tier]
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"routing_map tier {tier} → {model_key!r} not in MODEL_REGISTRY"
            )
        if rationales is not None and tier in rationales and rationales[tier].strip():
            rationale = rationales[tier].strip()
        elif tier in existing and existing[tier].get("rationale"):
            rationale = existing[tier]["rationale"]
        else:
            rationale = f"Routed to tier {tier} model per routing map."
        routing_out[tier] = {"model": model_key, "rationale": rationale}

    payload = {
        "version": int(version),
        "routing": {
            tier: {"model": routing_out[tier]["model"], "rationale": routing_out[tier]["rationale"]}
            for tier in (1, 2, 3)
        },
    }
    body = _HEADER + yaml.safe_dump(
        payload,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    map_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish replace within configs/ only.
    tmp = map_path.with_suffix(map_path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(map_path)
    return map_path


def model_for_tier(tier: int, path: str | Path | None = None) -> ModelConfig:
    """Look up the ModelConfig configured for a complexity tier."""
    mapping = load_routing_map(path)
    if tier not in mapping:
        raise ValueError(f"No routing entry for tier {tier}")
    return MODEL_REGISTRY[mapping[tier]]


def route_prompt(
    prompt: str, path: str | Path | None = None
) -> tuple[int, str, ModelConfig]:
    """Classify a prompt and return (tier, model_key, ModelConfig)."""
    from app.classifier import predict_tier

    tier = predict_tier(prompt)
    mapping = load_routing_map(path)
    model_key = mapping[tier]
    return tier, model_key, MODEL_REGISTRY[model_key]
