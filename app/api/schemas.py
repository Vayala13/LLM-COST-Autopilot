"""Pydantic request/response models for POST /v1/completions.

Schema (documented): send either ``prompt`` (plain string) **or** ``messages``
(OpenAI-style chat list). The client does **not** choose a model — the
complexity router selects one. A ``model`` field is rejected (extra=forbid).

Auth: this portfolio API is local/unauthenticated for now. Do not invent
insecure shared API keys; real auth lands later if needed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
