# LLM Cost AutoPilot — Project Guide (AGENTS.md)

> **How to use this file:** This is the persistent context for the coding assistant (Fable) in Cursor.
> At the start of every session, read the **Current Status** section to know where we are.
> At the end of every session, update **Current Status** and the **Session Log**. That is a required step, not optional.

---

## Current Status

- **Phase:** 3 of 6 — Async Quality Verification Loop
- **Last completed:** 3.1 quality thresholds. `configs/quality_thresholds.yaml` defines extraction (`field_coverage >= 1.0`), summarization (`llm_judge_score > 4/5`, judge `claude-sonnet`), classification (`label_agreement == 1.0`, reference `claude-sonnet` — GPT-4o unavailable while OpenAI disabled). Loader: `app/quality/thresholds.py` (`load_quality_thresholds`, `threshold_for`). Smoke: `python -m scripts.show_quality_thresholds`.
- **Next action:** Phase 3.2 — async verifier (queue high-tier comparison after response; score agreement; log routing failures).
- **Blockers:** None. OpenAI disabled in `.env` (invalid key, 401) — not required; GPT pricing stays in registry for the Phase 4.3 "vs GPT-4o" cost math. Classification/summarization judge uses `claude-sonnet` until OpenAI is enabled.
- **Last updated:** 2026-08-01

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
- [ ] **3.2 Async verifier** — After the response returns to the user, queue an async job that sends the same prompt to the highest-tier model and compares outputs. Score agreement. If the cheap model diverges significantly, log a routing failure.
- [ ] **3.3 Auto-escalation** — On a caught failure, automatically re-run with the higher-tier model and return the better result (if latency permits). Log the escalation: original model, escalated model, cost delta, quality gap that triggered it.
- [ ] **3.4 Feedback to classifier** — Every routing failure becomes a new training example. Build a simple loop that retrains the classifier weekly using accumulated failure data. This is the flywheel that makes the system smarter over time.

### Phase 4: Logging and Cost Dashboard (Day 9–11)

- [ ] **4.1 Log everything** — Every request → one DB row: timestamp, prompt hash, complexity tier, routed model, cost, latency, verifier quality score, escalated flag. This is the audit trail.
- [ ] **4.2 Cost dashboard** — Show total cost per day/week vs. what it would have cost using GPT-4o for everything ("you saved $X"), routing distribution (pie chart of model share), quality score distribution, escalation rate over time.
- [ ] **4.3 Money-shot metric** — Calculate and prominently display the cost reduction percentage. If routing saved 60% vs. all-most-expensive, that number is the headline of the portfolio piece.

### Phase 5: Expose as an API (Day 11–13)

- [ ] **5.1 FastAPI service** — Single `POST /v1/completions` accepting a standard chat completion request. The user does not choose the model — the router does. Return the response with metadata: which model was selected and why.
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
| 2026-08-01 | 3.1 | Added `configs/quality_thresholds.yaml` (extraction field_coverage >= 1.0; summarization llm_judge_score > 4/5 with judge claude-sonnet; classification label_agreement == 1.0 with reference claude-sonnet — GPT-4o unavailable while OpenAI disabled). Implemented `app/quality/thresholds.py` (`load_quality_thresholds`, `threshold_for`) validating required use cases, comparison ops, and registry keys. Smoke script `scripts/show_quality_thresholds.py`. | Phase 3.2 (async verifier) |
| 2026-07-31 | 2.4 | Added `configs/routing_map.yaml` (Tier 1 → llama-local, Tier 2 → gemini-flash, Tier 3 → claude-sonnet). Implemented `app/router/map.py` (`load_routing_map`, `model_for_tier`, `route_prompt`) validating registry keys on load. Smoke script `scripts/show_routing.py`. Phase 2 complete. | Phase 3.1 (quality thresholds) |
| 2026-07-31 | 2.3 | Trained scikit-learn classifiers on 11 features / 201 prompts. Stratified 75/25 split. Logistic regression and random forest both 88.2% held-out accuracy (target >80%). Saved winner (logistic regression) to `models/complexity_classifier.joblib`, metrics to `data/classifier_metrics.json`. Added `app/classifier/model.py` (`load_classifier`, `predict_tier`) and `scripts/train_classifier.py`. Pinned scikit-learn/numpy/joblib in requirements. | Phase 2.4 (routing map YAML) |
| 2026-07-29 | 2.2 | Built the labeled dataset: `data/labeled_prompts.jsonl` (201 hand-labeled prompts, 72/62/67 across tiers 1/2/3). Added `app/classifier/features.py` extracting 11 features (token/char count, per-tier instruction-verb counts loaded from the tiers YAML, constraint count, context-provided, reasoning-required, question marks, has-numbers, output-format complexity). `scripts/inspect_dataset.py` validates tier balance + duplicates and writes `data/prompt_features.json`. Mean feature values separate cleanly by tier. Added PyYAML to requirements. Also cloned the SIGAI-Pilot team docs repo as a sibling folder for documentation. | Phase 2.3 (train scikit-learn classifier, >80% held-out) |
| 2026-07-24 | 2.1 | Defined the 3 complexity tiers in `configs/complexity_tiers.yaml` (summaries, task types, example prompts, provisional targets, and the feature signals the classifier will use). Disabled the invalid OpenAI key in `.env` so baseline runs skip cleanly. | Phase 2.2 (labeled dataset + features) |
| 2026-07-24 | 1.3 | Reinstalled Ollama (empty app bundle was the blocker), pulled `llama3.2`, started daemon. Ran the full 10-prompt baseline across `claude-sonnet`, `claude-haiku`, `llama-local` → 30 records in `data/baseline_results.json`. Cost totals: Sonnet $0.0386, Haiku $0.0061 (−84%), Llama $0.00. Added Gemini provider: `gemini-flash` (`gemini-flash-latest`) works as a Tier 2 model; `gemini-pro` dropped (free tier has 0-request quota → 429). Installed Figma plugin + drafted architecture diagram (pending plan selection). | Phase 2.1 (complexity tiers) |
| 2026-07-23 | 1.1–1.3 | Built `app/providers/` (ModelConfig registry, Response, unified `send_request`), 10 baseline prompts, and `scripts/baseline_test.py`. Live smoke test passed on Anthropic; OpenAI/Ollama skip gracefully. Also set up foundation (.env, config.py, venv) and a portfolio devlog site. | Full 10-prompt baseline once OpenAI/Ollama available, then Phase 2.1 |
| _YYYY-MM-DD_ | 1.1 | _example: created ModelConfig + registry_ | _1.2_ |