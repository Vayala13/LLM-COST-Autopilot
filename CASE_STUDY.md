# Case Study: LLM Cost AutoPilot

# I built a system that reduced LLM API costs by **30.6%** while keeping a quality safety net in the loop.

**Baseline:** send every request to GPT-4o.  
**Result (offline load test, n=750):** actual **$0.8286** vs GPT-4o counterfactual **$1.1942** → saved **$0.3656** (**30.6%**).  
**Artifacts:** [`reports/load_test_savings.md`](./reports/load_test_savings.md) · [`reports/load_test_dashboard.png`](./reports/load_test_dashboard.png)

> Honest scope: the headline number comes from a **realistic offline load test** — real complexity classifier + YAML routing map, with provider costs mocked from `MODEL_REGISTRY` list prices (so we don’t burn API quota). Live providers are key-gated and still work when configured.

---

## The problem

Teams default to one expensive model for every prompt. Most traffic is mechanical (extract a field, reformat JSON, short classify). Paying Sonnet/GPT-4o rates for that is waste. Paying a tiny local model for multi-step reasoning is a quality risk.

**Goal:** route each request to the *cheapest* model that can still clear a “good enough” bar — then prove it.

---

## Routing logic

```
prompt → features → complexity tier (1/2/3) → routing map → model → response
                                                              ↓
                                                    audit row (prompt_hash)
                                                              ↓
                                              async verify → escalate? → feedback
```

1. **Classify complexity** — logistic regression on 11 features (88.2% held-out). Tiers live in `configs/complexity_tiers.yaml`.
2. **Map tier → model** (`configs/routing_map.yaml`, swappable without code changes):
   - Tier 1 → `llama-local` ($0)
   - Tier 2 → `gemini-flash`
   - Tier 3 → `claude-sonnet`
3. **Unified provider call** — `send_request(prompt, model_config)` only; no SDK leaks into router/classifier.
4. **Client never picks the model** — FastAPI `POST /v1/completions` rejects a client `model` field.

### Load-test mix (why 30.6% happens)

| Model | Share |
|---|---:|
| `llama-local` | 56.7% |
| `gemini-flash` | 28.7% |
| `claude-sonnet` | 14.7% |

Most mass lands on free/cheap tiers; expensive capacity is reserved for hard prompts.

---

## Quality safety net (the “parity” story)

Cost savings only matter if wrong routes get caught.

| Layer | What it does |
|---|---|
| **Thresholds** | YAML bars per use case — extraction field coverage, summarization judge &gt;4/5, classification label agreement (`configs/quality_thresholds.yaml`). |
| **Async verifier** | After the user gets a response, compare cheap vs high-tier / judge; log routing failures (`prompt_hash` only). |
| **Auto-escalation** | On failure, re-run with the escalation model (latency-gated); log cost delta + quality gap. |
| **Feedback flywheel** | Failures become labeled examples (`tier + 1`, capped at 3) → weekly `retrain_from_feedback`. |

In the offline load harness, **3.2%** of requests were marked escalated (24 / 750) — i.e. **~96.8%** stayed on the original cheap route without an escalation flag. That is the operational quality signal we measured in this portfolio run; live LLM-as-judge scores still apply when API keys are present.

---

## Observability & delivery

- **SQLite audit** — one row per completion: tier, model, cost, latency, quality score, escalated.
- **Dashboard** — Streamlit hero = cost-reduction % vs all GPT-4o.
- **API** — FastAPI completions + models/stats/routing-config.
- **Compose** — `api` + `worker` on shared `./data` (worker = retrain flywheel; verify stays in-process asyncio in the API).

---

## How to reproduce the headline

```bash
PYTHONPATH=. python -m scripts.load_test          # n=750 → reports/
PYTHONPATH=. python -m scripts.show_savings --demo  # smaller 36.7% seed (screenshots)
```

---

## Takeaway

**Lead with the number:** *Reduced LLM API costs by 30.6% vs all GPT-4o.*  
**How:** complexity-aware routing to cheaper models.  
**Why it’s not reckless:** async verification, escalation, and a classifier feedback loop close the quality gap over time.
