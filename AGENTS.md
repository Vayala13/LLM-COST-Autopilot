# LLM Cost AutoPilot — Project Guide (AGENTS.md)

> **How to use this file:** This is the persistent context for the coding assistant (Fable) in Cursor.
> At the start of every session, read the **Current Status** section to know where we are.
> At the end of every session, update **Current Status** and the **Session Log**. That is a required step, not optional.

---

## Current Status

- **Phase:** 5 of 6 — Expose as an API
- **Last completed:** 5.1 FastAPI `POST /v1/completions`. Router chooses model (client cannot force). Returns output + model/tier/rationale/cost/latency/tokens; audits via `log_completion` (prompt_hash). Optional async verify enqueue. Smoke `python -m scripts.smoke_api` (TestClient + mocks). Run: `uvicorn app.api.main:app --host 127.0.0.1 --port 8000`.
- **Next action:** Phase 5.2 — Config endpoints (`GET /v1/models`, `GET /v1/stats`, `PUT /v1/routing-config`).
- **Blockers:** None. OpenAI disabled in `.env` (invalid key, 401) — not required; GPT-4o pricing in registry powers the dashboard counterfactual. Classification/summarization judge uses `claude-sonnet` until OpenAI is enabled. Portfolio API is local/unauthenticated.
- **Last updated:** 2026-08-02

> **Update rule:** Whenever a step is finished, change "Last completed", set "Next action" to the next unchecked box, and add a line to the Session Log.

---

## What We're Building

An intelligent routing layer that sits in front of multiple LLM providers. It analyzes each incoming request's complexity, routes it to the cheapest model capable of handling it at acceptable quality, and continuously validates that routing decisions were correct.

**Portfolio headline (the goal):** "Reduced LLM API costs by X% while maintaining Y% quality parity."

---

## Tech Stack

| Component | Tool / Library | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem compatibility |
| LLM Providers | OpenAI, Anthropic, Ollama (local) | Mix of cloud and local models |
| Router | FastAPI | Async-native, production-grade |
| Classifier | Scikit-learn or small fine-tuned model | Lightweight complexity scoring |
| Eval | Custom scoring + LLM-as-judge | Quality verification loop |
| Logging | SQLite + structured JSON logs | Full audit trail per request |
| Dashboard | Streamlit or Grafana | Cost and quality visualization |
| Containerization | Docker + docker-compose | Multi-service orchestration |

---

## Conventions for the Assistant

1. Work **one step at a time**, in phase order. Do not jump ahead.
2. Before writing code, confirm which numbered step you are on against **Current Status**.
3. Keep provider-specific logic behind the unified interface — no leaking OpenAI/Anthropic calls into router or classifier code.
4. Store swappable config (routing map, model registry pricing, quality thresholds) in **YAML**, not hardcoded.
5. Every request must be logged to SQLite — no silent paths.
6. After finishing a step, update **Current Status** and **Session Log** in this file.
7. After each numbered phase step is **merged**, update SIGAI-Pilot docs at `../SIGAI-Pilot/LLM-COST-Autopilot/Viviana-Ayala/{timeline,issues,prompts}.md` (commit + push that repo).

---

## Build Plan

### Phase 1: Unified Model Interface (Day 1–3)

- [ ] **1.1 Model registry** — Define a `ModelConfig` dataclass: provider name, model ID, cost per input token, cost per output token, average latency, quality tier (high/medium/low). Populate with real pricing for GPT-4o, GPT-4o-mini, Claude Sonnet, Claude Haiku, and a local Llama via Ollama.
- [ ] **1.2 Abstraction layer** — Write one `send_request(prompt, model_config)` that handles provider-specific API calls behind a unified interface. Every call returns a standardized `Response` object: output text, tokens used (input + output), latency, cost, model ID.
- [ ] **1.3 Test every provider** — Send the same 10 prompts to every model in the registry. Log outputs, costs, latencies. This is baseline data for routing and validates the abstraction layer.

### Phase 2: Complexity Classifier (Day 3–6)

- [x] **2.1 Define complexity tiers** — Tier 1 (simple): reformatting, extraction, basic Q&A from provided context. Tier 2 (moderate): summarization, classification, structured analysis. Tier 3 (complex): multi-step reasoning, creative generation, nuanced judgment. → `configs/complexity_tiers.yaml`.
- [x] **2.2 Labeled dataset** — 201 hand-labeled prompts in `data/labeled_prompts.jsonl`; 11 features extracted by `app/classifier/features.py`; validated via `scripts/inspect_dataset.py` → `data/prompt_features.json`.
- [x] **2.3 Train the classifier** — Logistic regression vs random forest on `data/prompt_features.json`; both 88.2% held-out. Winner LR → `models/complexity_classifier.joblib`; metrics in `data/classifier_metrics.json`.
- [x] **2.4 Routing map** — `configs/routing_map.yaml`: Tier 1 → `llama-local`, Tier 2 → `gemini-flash`, Tier 3 → `claude-sonnet`. Loader `app/router/map.py`; smoke via `python -m scripts.show_routing`.

### Phase 3: Async Quality Verification Loop (Day 6–9)

- [x] **3.1 Quality thresholds per use case** — Define "good enough" per request type. Extraction: `field_coverage >= 1.0`. Summarization: LLM-as-judge score >4/5 (`claude-sonnet` judge). Classification: label agreement with high-tier reference (`claude-sonnet`; plan named GPT-4o, OpenAI disabled). → `configs/quality_thresholds.yaml`; loader `app/quality/thresholds.py`; smoke `python -m scripts.show_quality_thresholds`.
- [x] **3.2 Async verifier** — After the response returns to the user, queue an async job that sends the same prompt to the highest-tier model and compares outputs. Score agreement. If the cheap model diverges significantly, log a routing failure. → `app/quality/verifier.py` + `app/quality/queue.py`; smoke `python -m scripts.smoke_verifier`.
- [x] **3.3 Auto-escalation** — On a caught failure, automatically re-run with the higher-tier model and return the better result (if latency permits). Log the escalation: original model, escalated model, cost delta, quality gap that triggered it. → `configs/escalation.yaml`; `app/quality/escalation.py` (`escalate_if_needed`); smoke `python -m scripts.smoke_escalation`.
- [x] **3.4 Feedback to classifier** — Every routing failure becomes a new training example. Build a simple loop that retrains the classifier weekly using accumulated failure data. This is the flywheel that makes the system smarter over time. → `app/classifier/feedback.py` + `scripts/retrain_from_feedback.py`; smoke `python -m scripts.smoke_feedback`.

### Phase 4: Logging and Cost Dashboard (Day 9–11)

- [x] **4.1 Log everything** — Every request → one DB row: timestamp, prompt hash, complexity tier, routed model, cost, latency, verifier quality score, escalated flag. This is the audit trail. → `app/audit/store.py` (`log_completion`); DB `data/requests.db` (gitignored); smoke `python -m scripts.smoke_audit`.
- [x] **4.2 Cost dashboard** — Show total cost per day/week vs. what it would have cost using GPT-4o for everything ("you saved $X"), routing distribution (pie chart of model share), quality score distribution, escalation rate over time. → `app/metrics/cost.py` + `dashboard/app.py` (`streamlit run dashboard/app.py --server.address=127.0.0.1`); smoke `python -m scripts.smoke_metrics`.
- [x] **4.3 Money-shot metric** — Calculate and prominently display the cost reduction percentage. If routing saved 60% vs. all-most-expensive, that number is the headline of the portfolio piece. → `cost_reduction_pct` on `DashboardSummary` (vs all GPT-4o); hero in `dashboard/app.py`; CLI `scripts/show_savings.py`; smoke in `scripts/smoke_metrics.py`.

### Phase 5: Expose as an API (Day 11–13)

- [x] **5.1 FastAPI service** — Single `POST /v1/completions` accepting a standard chat completion request. The user does not choose the model — the router does. Return the response with metadata: which model was selected and why. → `app/api/main.py` + schemas/completions; smoke `python -m scripts.smoke_api`.
- [ ] **5.2 Config endpoints** — `GET /v1/models` (available models + costs), `GET /v1/stats` (cost savings summary), `PUT /v1/routing-config` (update tier→model mappings without redeploying).
- [ ] **5.3 Containerize & document** — docker-compose with the API service, a background worker for async verification, and the SQLite DB. README with architecture diagram, setup instructions, and the cost savings number front and center.

### Phase 6: Polish for Portfolio (Day 13–14)

- [ ] **6.1 Realistic load test** — Send 500–1,000 diverse prompts through the system. Generate the final cost savings report. Screenshot the dashboard. These are the portfolio artifacts.
- [ ] **6.2 Case study** — Frame as: "I built a system that reduced LLM API costs by X% while maintaining Y% quality parity." Lead with the number. Explain the routing logic. Show the feedback loop.

---

## Session Log

> Add a dated entry each session. Newest at top.

| Date | Phase/Step | What happened | Next |
|---|---|---|---|
| 2026-08-02 | 5.1 | FastAPI `POST /v1/completions`: `app/api/{main,schemas,completions}.py`. Schema = `prompt` **or** OpenAI-ish `messages` (extra=forbid; no client model). Pipeline: `route_prompt` → `send_request` → `log_completion` → optional async `enqueue_verification`. Response includes model/tier/rationale/cost/latency/tokens. Pinned `fastapi==0.141.1`, `uvicorn==0.52.1`, `httpx==0.28.1`. Smoke `scripts/smoke_api.py` (TestClient + mocks). README uvicorn localhost. Escalation left as TODO hook (async verify; don't block hot path). | Phase 5.2 (config endpoints) |
| 2026-08-02 | 4.3 | Money-shot metric: `cost_reduction_pct` + `PORTFOLIO_BASELINE_LABEL` ("vs all GPT-4o"); `format_portfolio_headline` / `load_portfolio_headline`. Dashboard hero (large % + $ saved). CLI `scripts/show_savings.py` (`--demo` temp DB). README + light portfolio log note. Smoke asserts headline helpers. Phase 4 complete. | Phase 5.1 (FastAPI completions) |
| 2026-08-02 | 4.2 | Cost dashboard: `app/metrics/cost.py` (actual vs GPT-4o via registry + tokens; savings $/%; cost by day/week; routing share; quality buckets; escalation rate by day; `seed_demo_requests`). Streamlit `dashboard/app.py` (savings callout + charts; empty-DB demo button; localhost bind note). Audit schema + nullable `input_tokens`/`output_tokens` (minimal migration). Pinned `streamlit==1.60.0`. Smoke `scripts/smoke_metrics.py` (offline temp DB). | Phase 4.3 (money-shot metric polish) |
| 2026-08-02 | 4.1 | SQLite request audit trail: `app/audit/store.py` (`init_db`, `log_request`, `log_completion`, `fetch_requests`). One row per completion — timestamp, prompt_hash, complexity_tier, routed_model, cost, latency, verifier_quality_score, escalated (+ nullable escalation_model / cost_delta / use_case). Parameterized SQL; DB `data/requests.db` gitignored. Smoke `scripts/smoke_audit.py` (offline temp DB). FastAPI wiring deferred to Phase 5. | Phase 4.2 (cost dashboard) |
| 2026-08-02 | 3.4 | Classifier feedback flywheel: `app/classifier/feedback.py` records labeled examples on routing failure (prompt in memory; relabel = routed tier + 1 capped at 3). Accumulates in gitignored `data/feedback_prompts.jsonl`. `scripts/retrain_from_feedback.py` merges base + feedback, rebuilds features, reuses `train_and_save` (weekly = cron/manual). Hooked from `verify()`. Smoke `scripts/smoke_feedback.py` (offline). Phase 3 complete. | Phase 4.1 (SQLite request logging) |
| 2026-08-02 | 3.3 | Added auto-escalation on routing failure: `configs/escalation.yaml` (`escalation_model` + optional `max_escalation_latency_s`), `app/quality/escalation.py` (`escalate_if_needed`, `EscalationResult`, `load_escalation_config`). Re-runs original prompt via unified `send_request` to tier-3 target; skips on pass / already-highest / latency budget (would-be still logged). JSONL `data/escalations.jsonl` (prompt hash + metrics; no raw prompt/outputs). Smoke `scripts/smoke_escalation.py` (offline mocks). | Phase 3.4 (classifier feedback / weekly retrain) |
| 2026-08-01 | 3.2 | Added async quality verifier: `app/quality/verifier.py` (`verify`, field_coverage / llm_judge_score / label_agreement scoring via unified `send_request`), `passes_threshold` in `thresholds.py`, in-process asyncio queue `app/quality/queue.py` (`enqueue_verification`, `drain`). Routing failures logged (structured warning + JSONL under `data/routing_failures.jsonl` with prompt hash only). Smoke `scripts/smoke_verifier.py` (mocked offline; optional `--live`). | Phase 3.3 (auto-escalation) |
| 2026-08-01 | 3.1 | Added `configs/quality_thresholds.yaml` (extraction field_coverage >= 1.0; summarization llm_judge_score > 4/5 with judge claude-sonnet; classification label_agreement == 1.0 with reference claude-sonnet — GPT-4o unavailable while OpenAI disabled). Implemented `app/quality/thresholds.py` (`load_quality_thresholds`, `threshold_for`) validating required use cases, comparison ops, and registry keys. Smoke script `scripts/show_quality_thresholds.py`. | Phase 3.2 (async verifier) |
| 2026-07-31 | 2.4 | Added `configs/routing_map.yaml` (Tier 1 → llama-local, Tier 2 → gemini-flash, Tier 3 → claude-sonnet). Implemented `app/router/map.py` (`load_routing_map`, `model_for_tier`, `route_prompt`) validating registry keys on load. Smoke script `scripts/show_routing.py`. Phase 2 complete. | Phase 3.1 (quality thresholds) |
| 2026-07-31 | 2.3 | Trained scikit-learn classifiers on 11 features / 201 prompts. Stratified 75/25 split. Logistic regression and random forest both 88.2% held-out accuracy (target >80%). Saved winner (logistic regression) to `models/complexity_classifier.joblib`, metrics to `data/classifier_metrics.json`. Added `app/classifier/model.py` (`load_classifier`, `predict_tier`) and `scripts/train_classifier.py`. Pinned scikit-learn/numpy/joblib in requirements. | Phase 2.4 (routing map YAML) |
| 2026-07-29 | 2.2 | Built the labeled dataset: `data/labeled_prompts.jsonl` (201 hand-labeled prompts, 72/62/67 across tiers 1/2/3). Added `app/classifier/features.py` extracting 11 features (token/char count, per-tier instruction-verb counts loaded from the tiers YAML, constraint count, context-provided, reasoning-required, question marks, has-numbers, output-format complexity). `scripts/inspect_dataset.py` validates tier balance + duplicates and writes `data/prompt_features.json`. Mean feature values separate cleanly by tier. Added PyYAML to requirements. Also cloned the SIGAI-Pilot team docs repo as a sibling folder for documentation. | Phase 2.3 (train scikit-learn classifier, >80% held-out) |
| 2026-07-24 | 2.1 | Defined the 3 complexity tiers in `configs/complexity_tiers.yaml` (summaries, task types, example prompts, provisional targets, and the feature signals the classifier will use). Disabled the invalid OpenAI key in `.env` so baseline runs skip cleanly. | Phase 2.2 (labeled dataset + features) |
| 2026-07-24 | 1.3 | Reinstalled Ollama (empty app bundle was the blocker), pulled `llama3.2`, started daemon. Ran the full 10-prompt baseline across `claude-sonnet`, `claude-haiku`, `llama-local` → 30 records in `data/baseline_results.json`. Cost totals: Sonnet $0.0386, Haiku $0.0061 (−84%), Llama $0.00. Added Gemini provider: `gemini-flash` (`gemini-flash-latest`) works as a Tier 2 model; `gemini-pro` dropped (free tier has 0-request quota → 429). Installed Figma plugin + drafted architecture diagram (pending plan selection). | Phase 2.1 (complexity tiers) |
| 2026-07-23 | 1.1–1.3 | Built `app/providers/` (ModelConfig registry, Response, unified `send_request`), 10 baseline prompts, and `scripts/baseline_test.py`. Live smoke test passed on Anthropic; OpenAI/Ollama skip gracefully. Also set up foundation (.env, config.py, venv) and a portfolio devlog site. | Full 10-prompt baseline once OpenAI/Ollama available, then Phase 2.1 |
| _YYYY-MM-DD_ | 1.1 | _example: created ModelConfig + registry_ | _1.2_ |