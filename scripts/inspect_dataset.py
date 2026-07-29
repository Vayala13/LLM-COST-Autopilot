"""Phase 2.2 — validate the labeled dataset and preview extracted features.

Loads data/labeled_prompts.jsonl, checks the tier balance, extracts features for
every prompt, and prints the mean of each feature per tier so you can eyeball
whether the signals actually separate the tiers before training (2.3). Also
writes the feature matrix to data/prompt_features.json for the classifier step.

Run:
    python -m scripts.inspect_dataset
"""

import json
import sys
from collections import Counter
from pathlib import Path

from app.classifier import FEATURE_NAMES, extract_features, load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    rows = load_dataset()
    tiers = Counter(r["tier"] for r in rows)

    print(f"Loaded {len(rows)} labeled prompts from data/labeled_prompts.jsonl\n")
    print("Tier balance:")
    for tier in sorted(tiers):
        bar = "#" * tiers[tier]
        print(f"  Tier {tier}: {tiers[tier]:>3}  {bar}")

    if len(tiers) != 3:
        print("\nWARNING: expected 3 tiers.")
    if min(tiers.values()) < 60:
        print("\nWARNING: at least one tier has < 60 examples; add more for balance.")

    dupes = [p for p, c in Counter(r["prompt"] for r in rows).items() if c > 1]
    if dupes:
        print(f"\nWARNING: {len(dupes)} duplicate prompt(s) found:")
        for p in dupes:
            print(f"  {p!r}")
    else:
        print("\nNo duplicate prompts.")

    features = [extract_features(r["prompt"]) for r in rows]

    print("\nMean feature value per tier (want these to separate across tiers):")
    header = "feature".ljust(22) + "".join(f"T{t}".rjust(9) for t in sorted(tiers))
    print("  " + header)
    for name in FEATURE_NAMES:
        line = name.ljust(22)
        for tier in sorted(tiers):
            vals = [f[name] for f, r in zip(features, rows) if r["tier"] == tier]
            line += f"{sum(vals) / len(vals):>9.2f}"
        print("  " + line)

    _write_features(rows, features)


def _write_features(rows: list[dict], features: list[dict]) -> None:
    out = DATA_DIR / "prompt_features.json"
    payload = {
        "feature_names": FEATURE_NAMES,
        "rows": [
            {"prompt": r["prompt"], "tier": r["tier"],
             "features": [f[n] for n in FEATURE_NAMES]}
            for r, f in zip(rows, features)
        ],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(rows)} feature rows to {out}")


if __name__ == "__main__":
    sys.exit(main())
