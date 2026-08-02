# Load-test cost savings report (Phase 6.1)

**Headline:** Reduced LLM API costs by 30.6% vs all GPT-4o (saved $0.3656; actual $0.8286 vs GPT-4o $1.1942; n=750)

- **n:** 750
- **cost_reduction_pct:** 30.6% (vs all GPT-4o)
- **saved:** $0.3656
- **actual:** $0.8286
- **GPT-4o counterfactual:** $1.1942
- **escalation_rate:** 3.2% (24 / 750)
- **mean_quality_score:** 2.148
- **mode:** full

## Routing distribution

| Model | Requests | Share |
|---|---:|---:|
| `llama-local` | 425 | 56.7% |
| `gemini-flash` | 215 | 28.7% |
| `claude-sonnet` | 110 | 14.7% |

## Tier distribution (classifier)

| Tier | Count |
|---|---:|
| 1 | 425 |
| 2 | 215 |
| 3 | 110 |

## Notes

- Offline load test: real classifier + routing map; mocked provider costs from `MODEL_REGISTRY`.
- Audit DB uses `prompt_hash` only (never raw prompts).
- Counterfactual: Hypothetical GPT-4o cost uses MODEL_REGISTRY['gpt-4o'] list pricing on per-row input/output tokens when present; otherwise falls back to 500 in / 250 out tokens per request.

Re-run:

```bash
PYTHONPATH=. python -m scripts.load_test
PYTHONPATH=. python -m scripts.load_test --smoke
```

Dashboard PNG: `load_test_dashboard.png`
