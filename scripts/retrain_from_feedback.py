"""Phase 3.4 — weekly retrain from accumulated routing-failure feedback.

Merges ``data/labeled_prompts.jsonl`` with ``data/feedback_prompts.jsonl``,
rebuilds the feature matrix, and reuses ``train_and_save`` from
``scripts.train_classifier`` to write an updated joblib + metrics.

"Weekly" is operational, not built-in scheduling: run this script manually
or via cron (e.g. ``0 3 * * 0``). No Celery / worker infra in Phase 3.

Offline-safe — no LLM API calls.

Run:
    python -m scripts.retrain_from_feedback
    python -m scripts.retrain_from_feedback --dry-run
    python -m scripts.retrain_from_feedback --force   # retrain even with 0 feedback
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from app.classifier.features import FEATURE_NAMES, extract_features, load_dataset
from app.classifier.feedback import DEFAULT_FEEDBACK_PATH, load_feedback, merge_training_rows
from scripts.train_classifier import (
    METRICS_PATH,
    MODEL_PATH,
    train_and_save,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BASE_PATH = DATA_DIR / "labeled_prompts.jsonl"
FEATURES_PATH = DATA_DIR / "prompt_features.json"


def build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract features for merged rows; return (X, y, payload_dict)."""
    features = [extract_features(r["prompt"]) for r in rows]
    x = np.array([[f[n] for n in FEATURE_NAMES] for f in features], dtype=float)
    y = np.array([r["tier"] for r in rows], dtype=int)
    payload = {
        "feature_names": FEATURE_NAMES,
        "rows": [
            {
                "prompt": r["prompt"],
                "tier": r["tier"],
                "features": [f[n] for n in FEATURE_NAMES],
            }
            for r, f in zip(rows, features)
        ],
    }
    return x, y, payload


def retrain(
    *,
    base_path: Path = BASE_PATH,
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    dry_run: bool = False,
    force: bool = False,
) -> dict | None:
    """Merge base+feedback, extract features, train. Returns metrics or None."""
    base_rows = load_dataset(base_path)
    feedback_rows = load_feedback(feedback_path)
    n_feedback = len(feedback_rows)

    print(f"Base labeled prompts: {len(base_rows)} ({base_path})")
    print(f"Feedback examples:    {n_feedback} ({feedback_path})")

    if n_feedback == 0 and not force:
        print("No feedback examples — nothing to do (pass --force to retrain on base only).")
        return None

    merged = merge_training_rows(base_rows, feedback_rows)
    print(f"Merged training rows: {len(merged)} (feedback overrides same prompt)")

    if dry_run:
        from collections import Counter

        tiers = Counter(r["tier"] for r in merged)
        print("Dry-run tier balance:", dict(sorted(tiers.items())))
        print("Dry-run: skip feature write + train.")
        return {"dry_run": True, "n_examples": len(merged), "n_feedback": n_feedback}

    x, y, payload = build_feature_matrix(merged)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote features → {features_path}")

    return train_and_save(
        x,
        y,
        model_path=model_path,
        metrics_path=metrics_path,
        n_feedback=n_feedback,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrain complexity classifier from routing-failure feedback."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=BASE_PATH,
        help="Base labeled prompts JSONL (default: data/labeled_prompts.jsonl)",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=DEFAULT_FEEDBACK_PATH,
        help="Feedback JSONL (default: data/feedback_prompts.jsonl)",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        default=FEATURES_PATH,
        help="Feature matrix JSON output (default: data/prompt_features.json)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=MODEL_PATH,
        help="Joblib model output (default: models/complexity_classifier.joblib)",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=METRICS_PATH,
        help="Metrics JSON output (default: data/classifier_metrics.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show merge counts only; do not write model/features.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when feedback file is empty/missing.",
    )
    args = parser.parse_args(argv)

    result = retrain(
        base_path=args.base,
        feedback_path=args.feedback,
        features_path=args.features_out,
        model_path=args.model_out,
        metrics_path=args.metrics_out,
        dry_run=args.dry_run,
        force=args.force,
    )
    if result is None:
        return 0
    if result.get("dry_run"):
        return 0
    print(
        f"\nretrain_from_feedback: winner={result['winner']} "
        f"accuracy={result['winner_accuracy']:.1%} "
        f"n_feedback={result['n_feedback']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
