# LLM Cost AutoPilot

# Reduced LLM API costs by **30.6%** vs all GPT-4o

Load-test aggregate (Phase 6.1, offline, `n=750`): saved **$0.3656**
(actual $0.8286 vs GPT-4o $1.1942). Full report + dashboard PNG:
[`reports/load_test_savings.md`](./reports/load_test_savings.md).

**Case study (Phase 6.2):** [`CASE_STUDY.md`](./CASE_STUDY.md) — routing logic,
quality loop, and the 30.6% headline framed for portfolio.

```bash
PYTHONPATH=. python -m scripts.load_test          # n=750 → reports/
PYTHONPATH=. python -m scripts.smoke_load_test    # n=50 CI-like check
```

Smaller offline demo seed (sidebar / `--demo`, `n=84`) shows **36.7%** — useful
for empty-DB screenshots, not the load-test headline:

```bash
PYTHONPATH=. python -m scripts.show_savings --demo
```

Dashboard (bind localhost; load-test DB or demo seed):

```bash
# After load_test, copy the gitignored load-test DB for the Streamlit hero:
cp data/load_test_requests.db data/requests.db
streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501
# Or empty data/requests.db + sidebar "Load demo data" (36.7% seed)
```

An intelligent routing layer in front of multiple LLM providers. It scores each
request's complexity, routes it to the cheapest model that can handle it well,
and asynchronously verifies that the routing decision was correct.

![Architecture](./architecture-diagram.png)

---

## Architecture (honest)

```
Client ──► FastAPI (api) ──► classifier → routing map → provider
                │                      │
                │              log_completion → data/requests.db
                │                      │
                └─ in-process asyncio verify (same process)
                         │
                         └─ feedback / failure JSONL → data/

         worker (separate container) ── watches data/ ──► retrain_from_feedback
```

**Verification queue:** `enqueue_verification` is an **in-memory asyncio**
queue inside the API process. It does **not** cross containers. The compose
`worker` shares the `./data` volume (SQLite + JSONL) and runs the classifier
retrain flywheel — it does not drain the API's asyncio queue.

**SQLite:** `data/requests.db` lives on the shared volume (gitignored).

**Auth:** Local/portfolio API is **unauthenticated**. Do not expose publicly
without real auth. No wildcard CORS with credentials.

---

## Quick start — Docker Compose (Phase 5.3)

```bash
# 1. Copy env template (no secrets in the image; fill keys as needed)
cp .env.example .env

# 2. Build + run API + worker (shared ./data volume for SQLite/JSONL)
docker compose up --build

# 3. Health + savings
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/v1/stats
```

Compose publishes the API on **127.0.0.1:8000** only. Containers run as
non-root, with `no-new-privileges`, and **without** `privileged: true`.
Set `ALLOW_ROUTING_CONFIG_WRITE=0` outside local demos (that flag is **not** auth).

| Service | Role |
|---|---|
| `api` | `uvicorn app.api.main:app` — completions + config endpoints; in-process verify |
| `worker` | `python -m app.worker.main` — data/ watch + optional retrain |

Validate the compose file (no daemon required for `config`):

```bash
docker compose config
```

Offline worker smoke (no Docker):

```bash
PYTHONPATH=. python -m scripts.smoke_worker
```

---

## Quick start — local venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add real keys if you want live providers

# Offline API smoke (TestClient + mocked provider)
PYTHONPATH=. python -m scripts.smoke_api

# Run API on localhost
PYTHONPATH=. uvicorn app.api.main:app --host 127.0.0.1 --port 8000
# Docs: http://127.0.0.1:8000/docs
```

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/completions` | Router chooses the model; clients do not |
| `GET` | `/v1/models` | Registry models + costs (no secrets) |
| `GET` | `/v1/stats` | Cost savings aggregates (no raw prompts) |
| `GET`/`PUT` | `/v1/routing-config` | Read/update tier→model map (`configs/routing_map.yaml`) |

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Convert March 3, 2026 to YYYY-MM-DD."}'

curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/stats
curl -s http://127.0.0.1:8000/v1/routing-config
```

---

## Cost dashboard

```bash
PYTHONPATH=. python -m scripts.smoke_metrics
PYTHONPATH=. python -m scripts.show_savings --demo
PYTHONPATH=. python -m scripts.smoke_load_test
streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501
```

If `data/requests.db` is empty, use **Load demo data** in the sidebar
(or copy `data/load_test_requests.db` after a load test). Charts are
aggregates only (no raw prompts).

---

## Project layout

| Path | Purpose |
|---|---|
| `app/api/` | FastAPI completions + config endpoints |
| `app/worker/` | Compose background worker (data/ watch + retrain) |
| `app/providers/` | Unified `send_request` + model registry |
| `app/classifier/` | Features + complexity model + feedback flywheel |
| `app/quality/` | Thresholds, in-process verify queue, escalation |
| `app/audit/` | SQLite request audit trail |
| `app/metrics/` | Savings aggregates (`cost_reduction_pct`) |
| `configs/` | YAML: tiers, routing map, thresholds, escalation |
| `reports/` | Phase 6.1 load-test savings report + dashboard PNG |
| `Dockerfile` / `docker-compose.yml` | API + worker + shared `data/` volume |
| `.env.example` | Env template — copy to `.env` (gitignored) |

See [`AGENTS.md`](./AGENTS.md) for the full 6-phase build plan and current status.

---

## Load test (Phase 6.1)

Offline portfolio run: real classifier + routing map; mocked provider costs from
the registry (no API quota). Writes gitignored `data/load_test_requests.db` and
commits summaries under `reports/`.

| Artifact | Path |
|---|---|
| Savings JSON | [`reports/load_test_savings.json`](./reports/load_test_savings.json) |
| Savings Markdown | [`reports/load_test_savings.md`](./reports/load_test_savings.md) |
| Dashboard PNG | [`reports/load_test_dashboard.png`](./reports/load_test_dashboard.png) |

Headline (n=750): **30.6%** cost reduction vs all GPT-4o.

Phase 6.2 will frame the case study around this number (not started here).

---

## Notes

- **Python:** 3.11+ (Docker image uses 3.11-slim). Local `.venv` may be newer.
- **Secret hygiene:** keys only in `.env` (gitignored). Rotate if exposed.
- **Providers are optional:** missing `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
  `GEMINI_API_KEY` → that provider skips or returns HTTP 503 on live calls.
  Offline smokes never need keys.
