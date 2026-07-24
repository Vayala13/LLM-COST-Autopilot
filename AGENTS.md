# LLM Cost AutoPilot — Project Guide (AGENTS.md)

> **How to use this file:** This is the persistent context for the coding assistant (Fable) in Cursor.
> At the start of every session, read the **Current Status** section to know where we are.
> At the end of every session, update **Current Status** and the **Session Log**. That is a required step, not optional.

---

## Current Status

- **Phase:** 1 of 6 — Unified Model Interface
- **Last completed:** _nothing yet — fresh repo_
- **Next action:** Phase 1, Step 1 — create the `ModelConfig` dataclass and model registry (see Phase 1 below).
- **Blockers:** none
- **Last updated:** _set on first session_

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

- [ ] **2.1 Define complexity tiers** — Tier 1 (simple): reformatting, extraction, basic Q&A from provided context. Tier 2 (moderate): summarization, classification, structured analysis. Tier 3 (complex): multi-step reasoning, creative generation, nuanced judgment.
- [ ] **2.2 Labeled dataset** — Write 200+ example prompts across all three tiers, hand-labeled. Extract features: token count, presence of instructions like "analyze"/"compare", number of constraints, whether context is provided, output format complexity.
- [ ] **2.3 Train the classifier** — Start with simple scikit-learn (logistic regression or random forest) on the extracted features. Goal is the routing skeleton, not perfection. Track accuracy and confusion matrix. >80% on a held-out set is fine for V1.
- [ ] **2.4 Routing map** — Map tier → model. Tier 1 → cheapest (Haiku or local Llama). Tier 2 → mid (GPT-4o-mini or Sonnet). Tier 3 → highest quality (GPT-4o or Opus). Store as configurable YAML so models can be swapped without code changes.

### Phase 3: Async Quality Verification Loop (Day 6–9)

- [ ] **3.1 Quality thresholds per use case** — Define "good enough" per request type. Extraction: got all key fields? Summarization: LLM-as-judge score >4/5. Classification: label matches what GPT-4o would have said?
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
| _YYYY-MM-DD_ | 1.1 | _example: created ModelConfig + registry_ | _1.2_ |