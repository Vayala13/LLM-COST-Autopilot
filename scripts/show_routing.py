"""Phase 2.4 — print the tier→model routing map and smoke-test route_prompt.

Run:
    python -m scripts.show_routing
"""

from app.providers import MODEL_REGISTRY
from app.router import load_routing_map, route_prompt


def main() -> None:
    mapping = load_routing_map()
    print("Routing map (configs/routing_map.yaml):\n")
    print(f"{'tier':<6}{'model':<16}{'provider':<12}{'quality':<10}cost_in/1M")
    for tier in sorted(mapping):
        key = mapping[tier]
        cfg = MODEL_REGISTRY[key]
        print(
            f"{tier:<6}{key:<16}{cfg.provider:<12}{cfg.quality_tier:<10}"
            f"${cfg.cost_per_input_token * 1_000_000:.2f}"
        )

    samples = [
        "Convert 'March 3, 2026' to YYYY-MM-DD.",
        "Summarize this product review in one sentence: Great battery, slow camera.",
        "Argue both sides of SQLite vs PostgreSQL for a small web app, then recommend one.",
    ]
    print("\nSmoke route_prompt:")
    for prompt in samples:
        tier, key, cfg = route_prompt(prompt)
        print(f"  T{tier} → {key} ({cfg.provider})  |  {prompt[:60]}")


if __name__ == "__main__":
    main()
