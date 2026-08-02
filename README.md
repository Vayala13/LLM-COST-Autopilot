# LLM Cost AutoPilot

An intelligent routing layer in front of multiple LLM providers. It scores each
request's complexity, routes it to the cheapest model that can handle it well,
and asynchronously verifies that the routing decision was correct.

**Goal:** "Reduced LLM API costs by X% while maintaining Y% quality parity."

**Portfolio headline (Phase 4.3):** print the live cost-reduction % vs all GPT-4o:

```bash
PYTHONPATH=. python -m scripts.show_savings --demo
```

See [`AGENTS.md`](./AGENTS.md) for the full 6-phase build plan and current status.

---

## What's set up so far

This is the **foundation** — secrets, config, and a working Python environment.
No routing or model logic yet; that starts in Phase 1 (see `AGENTS.md`).

| File | Purpose |
|---|---|
| `.env` | Holds real secrets (your `ANTHROPIC_API_KEY`). **Git-ignored — never committed.** |
| `.gitignore` | Keeps `.env`, `.venv/`, and `__pycache__/` out of git. |
| `config.py` | Loads `.env` and exposes `ANTHROPIC_API_KEY` to the rest of the code. |
| `requirements.txt` | Python dependencies (currently just `python-dotenv`). |
| `.venv/` | Isolated Python environment so installs don't touch system Python. |

---

## Setup

```bash
# 1. Create the virtual environment (once)
python3 -m venv .venv

# 2. Activate it (every new terminal)
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your real key to .env
#    ANTHROPIC_API_KEY=sk-ant-...
```

## Verify it works

```bash
python -c "import config; print('key loaded, length', len(config.ANTHROPIC_API_KEY))"
```

Expected output: `key loaded, length 108`

---

## Cost dashboard & money-shot metric (Phase 4.2–4.3)

```bash
# Offline metrics smoke (temp DB, no browser) — includes cost_reduction_pct checks
PYTHONPATH=. python -m scripts.smoke_metrics

# Portfolio headline number (offline; --demo uses a temp DB)
PYTHONPATH=. python -m scripts.show_savings --demo

# Local dashboard — bind to localhost only (hero = cost reduction % vs all GPT-4o)
streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501
```

If `data/requests.db` is empty, use **Load demo data** in the sidebar for portfolio screenshots. Charts are aggregates only (no raw prompts). The dashboard hero and `show_savings` both surface `cost_reduction_pct` from `app/metrics/cost.py`.

---

## How config loading works

`config.py` reads the `.env` file at import time, so any module can do:

```python
import config
client = SomeClient(api_key=config.ANTHROPIC_API_KEY)
```

The key lives only in `.env` (git-ignored), never hardcoded in source.

---

## Notes

- **Python version:** the `.venv` uses your system Python (3.14). `AGENTS.md`
  targets 3.11+, so this is fine.
- **Secret hygiene:** if the key is ever exposed (shared repo, screen share),
  rotate it in the [Anthropic console](https://console.anthropic.com/).
