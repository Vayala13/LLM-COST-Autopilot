"""Phase 1.3 — send the same baseline prompts to every model in the registry.

Logs outputs, costs, and latencies to data/baseline_results.json and prints a
summary table. Models whose provider isn't configured (missing API key, Ollama
not running) are skipped gracefully so the harness still runs on whatever you
have set up today.

Run:
    python -m scripts.baseline_test            # all 10 prompts, every model
    python -m scripts.baseline_test --limit 1  # quick smoke test
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.providers import MODEL_REGISTRY, ProviderNotConfigured, send_request
from app.providers.prompts import BASELINE_PROMPTS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(BASELINE_PROMPTS),
                        help="number of prompts to send (default: all 10)")
    args = parser.parse_args()

    prompts = BASELINE_PROMPTS[: args.limit]
    results = []
    skipped: dict[str, str] = {}

    for name, cfg in MODEL_REGISTRY.items():
        print(f"\n=== {name} ({cfg.provider}:{cfg.model_id}) ===")
        for i, prompt in enumerate(prompts, start=1):
            try:
                resp = send_request(prompt, cfg)
            except ProviderNotConfigured as e:
                print(f"  SKIP {name}: {e}")
                skipped[name] = str(e)
                break  # no point trying more prompts for an unconfigured provider
            except Exception as e:  # noqa: BLE001 - log any provider error, keep going
                print(f"  [{i}] ERROR: {e}")
                results.append({"model": name, "prompt_index": i, "error": str(e)})
                continue

            print(f"  [{i}] {resp.latency_s:.2f}s  ${resp.cost_usd:.6f}  "
                  f"{resp.total_tokens} tok  -> {resp.output_text[:60]!r}")
            results.append({
                "model": name,
                "provider": resp.provider,
                "prompt_index": i,
                "prompt": prompt,
                "output": resp.output_text,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "latency_s": round(resp.latency_s, 3),
                "cost_usd": resp.cost_usd,
            })

    _print_summary(results, skipped)
    _write_results(results, skipped)


def _print_summary(results: list[dict], skipped: dict[str, str]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY (per model)")
    print("=" * 60)
    by_model: dict[str, list[dict]] = {}
    for r in results:
        if "cost_usd" in r:
            by_model.setdefault(r["model"], []).append(r)

    print(f"{'model':<16}{'calls':>6}{'avg_latency':>13}{'total_cost':>13}")
    for model, rows in by_model.items():
        avg_lat = sum(r["latency_s"] for r in rows) / len(rows)
        total_cost = sum(r["cost_usd"] for r in rows)
        print(f"{model:<16}{len(rows):>6}{avg_lat:>12.2f}s{total_cost:>12.6f}$")

    for model, reason in skipped.items():
        print(f"{model:<16}  SKIPPED — {reason}")


def _write_results(results: list[dict], skipped: dict[str, str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "baseline_results.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "skipped": skipped,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(results)} records to {out}")


if __name__ == "__main__":
    sys.exit(main())
