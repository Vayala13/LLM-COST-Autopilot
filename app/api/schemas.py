"""Pydantic request/response models for the FastAPI surface.

Completions (Phase 5.1): send either ``prompt`` **or** ``messages``. The
client does **not** choose a model — the complexity router selects one.
A ``model`` field is rejected (extra=forbid).

Config (Phase 5.2): models list, stats aggregates, routing-config get/put.
PUT bodies are validated against ``MODEL_REGISTRY`` keys.

Auth: this portfolio API is local/unauthenticated for now. Do not invent
insecure shared API keys; real auth lands later if needed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.providers.registry import MODEL_REGISTRY


class ChatMessage(BaseModel):
    """One OpenAI-style chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=100_000)


class CompletionRequest(BaseModel):
    """Body for ``POST /v1/completions``.

    Provide **exactly one** of:
    - ``prompt``: plain user text
    - ``messages``: non-empty chat list (system/user/assistant)

    Optional ``use_case`` enables async quality verification enqueue
    (``extraction`` | ``summarization`` | ``classification``). Verification
    does not block the response.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(
        default=None,
        description="Plain prompt text (mutually exclusive with messages).",
        max_length=100_000,
    )
    messages: list[ChatMessage] | None = Field(
        default=None,
        description="OpenAI-style chat messages (mutually exclusive with prompt).",
        max_length=50,
    )
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    use_case: str | None = Field(
        default=None,
        description=(
            "Optional verification use case: extraction | summarization | classification. "
            "When set, verification is enqueued in the background (non-blocking)."
        ),
    )
    required_fields: list[str] | None = Field(
        default=None,
        description="For use_case=extraction: field names the verifier should check.",
        max_length=50,
    )
    enqueue_verification: bool = Field(
        default=True,
        description="When use_case is set, enqueue async verification (default true).",
    )

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, v: str | None) -> str | None:
        if v is None:
            return None
        text = v.strip()
        if not text:
            raise ValueError("prompt must be non-empty after strip")
        return text

    @field_validator("use_case")
    @classmethod
    def _normalize_use_case(cls, v: str | None) -> str | None:
        if v is None:
            return None
        key = v.strip().lower()
        allowed = {"extraction", "summarization", "classification"}
        if key not in allowed:
            raise ValueError(f"use_case must be one of {sorted(allowed)}")
        return key

    @field_validator("required_fields")
    @classmethod
    def _validate_required_fields(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned = [f.strip() for f in v if isinstance(f, str) and f.strip()]
        if not cleaned:
            raise ValueError("required_fields must be a non-empty list of strings")
        if len(cleaned) > 50:
            raise ValueError("required_fields: at most 50 names")
        for name in cleaned:
            if len(name) > 128:
                raise ValueError("required_fields entries must be <= 128 chars")
        return cleaned

    @model_validator(mode="after")
    def _exactly_one_input(self) -> CompletionRequest:
        has_prompt = self.prompt is not None
        has_messages = self.messages is not None and len(self.messages) > 0
        if has_prompt == has_messages:
            raise ValueError("Provide exactly one of: prompt, messages")
        if self.use_case == "extraction" and self.enqueue_verification:
            if not self.required_fields:
                raise ValueError(
                    "use_case=extraction requires required_fields when verification is enabled"
                )
        return self


class CompletionResponse(BaseModel):
    """Routed completion plus routing/cost metadata for the portfolio audit trail."""

    model_config = ConfigDict(extra="forbid")

    output_text: str
    model: str = Field(description="Registry key chosen by the router (not client).")
    model_id: str = Field(description="Provider model ID actually called.")
    provider: str
    complexity_tier: int = Field(ge=1, le=3)
    rationale: str = Field(description="Why this model was selected (from routing map).")
    cost_usd: float = Field(ge=0)
    latency_s: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    request_id: int = Field(description="SQLite audit row id (prompt_hash only in DB).")
    verification_enqueued: bool = False


# ---------------------------------------------------------------------------
# Phase 5.2 — config endpoints
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """One entry from MODEL_REGISTRY (no secrets)."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Internal registry key used in routing_map.yaml.")
    provider: str
    model_id: str
    cost_per_input_token: float = Field(ge=0)
    cost_per_output_token: float = Field(ge=0)
    avg_latency_s: float = Field(ge=0)
    quality_tier: Literal["high", "medium", "low"]


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelInfo]


class StatsResponse(BaseModel):
    """Cost savings aggregates — never raw prompts."""

    model_config = ConfigDict(extra="forbid")

    request_count: int = Field(ge=0)
    actual_cost_usd: float
    gpt4o_cost_usd: float = Field(ge=0)
    savings_usd: float
    savings_pct: float
    cost_reduction_pct: float = Field(
        description="Portfolio money-shot: % reduction vs all GPT-4o."
    )
    escalation_count: int = Field(ge=0)
    escalation_rate: float = Field(ge=0)
    scored_count: int = Field(ge=0)
    mean_quality_score: float | None = None
    baseline_label: str
    empty: bool = Field(description="True when the audit DB has zero requests.")
    counterfactual_note: str


class RoutingTierEntry(BaseModel):
    """One tier mapping: registry model key + optional rationale."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="MODEL_REGISTRY key (not a provider SDK model id).",
    )
    rationale: str | None = Field(default=None, max_length=500)

    @field_validator("model")
    @classmethod
    def _model_in_registry(cls, v: str) -> str:
        key = v.strip()
        if key not in MODEL_REGISTRY:
            raise ValueError(
                f"model {key!r} not in MODEL_REGISTRY; "
                f"known: {sorted(MODEL_REGISTRY)}"
            )
        return key

    @field_validator("rationale")
    @classmethod
    def _strip_rationale(cls, v: str | None) -> str | None:
        if v is None:
            return None
        text = v.strip()
        return text or None


class RoutingConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    routing: dict[str, RoutingTierEntry] = Field(
        description='Keys are tier strings "1", "2", "3".'
    )


class RoutingConfigUpdate(BaseModel):
    """Body for ``PUT /v1/routing-config``.

    Shape::

        {"routing": {"1": "llama-local", "2": "gemini-flash", "3": "claude-sonnet"}}

    or with optional rationales::

        {"routing": {"1": {"model": "claude-haiku", "rationale": "..."}}}

    Tiers 1/2/3 are required. Model keys must exist in ``MODEL_REGISTRY``.
    Extra fields forbidden. Clients cannot supply a write path.
    """

    model_config = ConfigDict(extra="forbid")

    routing: dict[str, Any] = Field(
        ...,
        description='Map of tier "1"|"2"|"3" → registry key or {model, rationale}.',
    )

    @model_validator(mode="after")
    def _validate_routing(self) -> RoutingConfigUpdate:
        if set(self.routing) != {"1", "2", "3"}:
            raise ValueError('routing must define exactly tiers "1", "2", and "3"')
        normalized: dict[str, RoutingTierEntry] = {}
        for tier_key, raw in self.routing.items():
            if isinstance(raw, str):
                entry = RoutingTierEntry(model=raw)
            elif isinstance(raw, dict):
                entry = RoutingTierEntry.model_validate(raw)
            elif isinstance(raw, RoutingTierEntry):
                entry = raw
            else:
                raise ValueError(
                    f"routing[{tier_key!r}] must be a model key string or object"
                )
            normalized[tier_key] = entry
        # Store normalized entries for handlers.
        object.__setattr__(self, "routing", normalized)
        return self

    def to_mapping(self) -> tuple[dict[int, str], dict[int, str]]:
        """Return ({tier: model_key}, {tier: rationale}) for tiers with rationale."""
        mapping: dict[int, str] = {}
        rationales: dict[int, str] = {}
        for tier_key, entry in self.routing.items():
            assert isinstance(entry, RoutingTierEntry)
            tier = int(tier_key)
            mapping[tier] = entry.model
            if entry.rationale:
                rationales[tier] = entry.rationale
        return mapping, rationales
