"""Model registry: one ModelConfig per model we can route to.

Pricing is stored per single token (USD). Public list prices are quoted per
1M tokens, so we convert with `_per_million` to keep the numbers readable.
Update these as provider pricing changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "openai" | "anthropic" | "ollama" | "gemini"
    model_id: str  # exact model string passed to the provider SDK
    cost_per_input_token: float
    cost_per_output_token: float
    avg_latency_s: float  # rough baseline; refined by the test harness
    quality_tier: str  # "high" | "medium" | "low"


def _per_million(dollars_per_million: float) -> float:
    return dollars_per_million / 1_000_000


# Keys are the names WE use internally (provider-agnostic labels).
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        cost_per_input_token=_per_million(2.50),
        cost_per_output_token=_per_million(10.00),
        avg_latency_s=1.8,
        quality_tier="high",
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_input_token=_per_million(0.15),
        cost_per_output_token=_per_million(0.60),
        avg_latency_s=1.0,
        quality_tier="medium",
    ),
    # Pricing below is approximate — verify against current Anthropic pricing.
    "claude-sonnet": ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-5",
        cost_per_input_token=_per_million(3.00),
        cost_per_output_token=_per_million(15.00),
        avg_latency_s=2.0,
        quality_tier="high",
    ),
    "claude-haiku": ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        cost_per_input_token=_per_million(0.80),
        cost_per_output_token=_per_million(4.00),
        avg_latency_s=1.0,
        quality_tier="medium",
    ),
    # Gemini pricing is approximate list price — verify at ai.google.dev/pricing.
    # On the free student tier the real API cost is $0; list prices are kept here
    # so the cost-savings math reflects a realistic paid scenario.
    "gemini-flash": ModelConfig(
        provider="gemini",
        model_id="gemini-flash-latest",
        cost_per_input_token=_per_million(0.30),
        cost_per_output_token=_per_million(2.50),
        avg_latency_s=1.2,
        quality_tier="medium",
    ),
    # gemini-pro omitted: free/student tier has a 0-request quota on the Pro
    # model (429 RESOURCE_EXHAUSTED). Claude Sonnet covers the high tier.
    "llama-local": ModelConfig(
        provider="ollama",
        model_id="llama3.2",
        cost_per_input_token=0.0,  # local model: no per-token API cost
        cost_per_output_token=0.0,
        avg_latency_s=1.5,
        quality_tier="low",
    ),
}
