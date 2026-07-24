"""Unified interface for talking to any provider.

Every model in the registry is reached through a single function:
    send_request(prompt, model_config) -> Response

Provider SDKs are imported lazily inside each handler so a missing package
(e.g. no `ollama` installed) only breaks that one provider, not the whole app.
"""

import time

import config

from .registry import ModelConfig
from .response import Response


class ProviderNotConfigured(Exception):
    """Raised when a provider is missing its API key or SDK."""


def send_request(prompt: str, model_config: ModelConfig, max_tokens: int = 1024) -> Response:
    handler = _HANDLERS.get(model_config.provider)
    if handler is None:
        raise ValueError(f"Unknown provider: {model_config.provider!r}")

    start = time.perf_counter()
    output_text, input_tokens, output_tokens = handler(prompt, model_config, max_tokens)
    latency_s = time.perf_counter() - start

    cost_usd = (
        input_tokens * model_config.cost_per_input_token
        + output_tokens * model_config.cost_per_output_token
    )

    return Response(
        model_id=model_config.model_id,
        provider=model_config.provider,
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
        cost_usd=cost_usd,
    )


def _call_anthropic(prompt: str, cfg: ModelConfig, max_tokens: int):
    if not config.ANTHROPIC_API_KEY:
        raise ProviderNotConfigured("ANTHROPIC_API_KEY is not set in .env")
    try:
        import anthropic
    except ImportError as e:
        raise ProviderNotConfigured("anthropic SDK not installed (pip install anthropic)") from e

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=cfg.model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_openai(prompt: str, cfg: ModelConfig, max_tokens: int):
    if not config.OPENAI_API_KEY:
        raise ProviderNotConfigured("OPENAI_API_KEY is not set in .env")
    try:
        import openai
    except ImportError as e:
        raise ProviderNotConfigured("openai SDK not installed (pip install openai)") from e

    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=cfg.model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_gemini(prompt: str, cfg: ModelConfig, max_tokens: int):
    if not config.GEMINI_API_KEY:
        raise ProviderNotConfigured("GEMINI_API_KEY is not set in .env")
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ProviderNotConfigured("google-genai SDK not installed (pip install google-genai)") from e

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=cfg.model_id,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    text = resp.text or ""
    usage = resp.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    return text, input_tokens, output_tokens


def _call_ollama(prompt: str, cfg: ModelConfig, max_tokens: int):
    try:
        import ollama
    except ImportError as e:
        raise ProviderNotConfigured("ollama SDK not installed (pip install ollama)") from e

    try:
        resp = ollama.chat(
            model=cfg.model_id,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": max_tokens},
        )
    except Exception as e:  # connection refused => daemon not running
        raise ProviderNotConfigured(f"Ollama not reachable: {e}") from e

    text = resp["message"]["content"]
    # Ollama reports token counts as prompt_eval_count / eval_count.
    return text, resp.get("prompt_eval_count", 0), resp.get("eval_count", 0)


_HANDLERS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
    "ollama": _call_ollama,
}
