import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Phase 5.2 — gate PUT /v1/routing-config writes. Default on for local/portfolio
# demos (API is unauthenticated). Set to 0/false/off to disable YAML writes in
# non-local deploys. This is not authentication — do not invent weak API keys.
_ALLOW_ROUTING_RAW = os.environ.get("ALLOW_ROUTING_CONFIG_WRITE", "1").strip().lower()
ALLOW_ROUTING_CONFIG_WRITE = _ALLOW_ROUTING_RAW in {"1", "true", "yes", "on"}
