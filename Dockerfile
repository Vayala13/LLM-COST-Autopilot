# LLM Cost AutoPilot — API / worker image (Phase 5.3)
# No secrets baked in. Pass keys via env / compose env_file (.env, gitignored).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Non-root runtime user (no privileged mode in compose).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + configs + classifier artifact. Runtime SQLite / JSONL live on the
# shared ``data/`` volume (see docker-compose.yml) — not copied secrets.
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser dashboard ./dashboard
COPY --chown=appuser:appuser config.py ./
COPY --chown=appuser:appuser data/labeled_prompts.jsonl ./data/labeled_prompts.jsonl
COPY --chown=appuser:appuser data/classifier_metrics.json ./data/classifier_metrics.json
COPY --chown=appuser:appuser data/baseline_results.json ./data/baseline_results.json

RUN mkdir -p /app/data && chown -R appuser:appuser /app/data /app/models /app/configs

USER appuser

EXPOSE 8000

# Default = API. Compose overrides command for the worker service.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
