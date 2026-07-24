"""Standardized response returned by every provider call."""

from dataclasses import dataclass


@dataclass
class Response:
    model_id: str
    provider: str
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
