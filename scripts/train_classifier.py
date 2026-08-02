"""Phase 2.3 — train a complexity-tier classifier on extracted features.

Compares logistic regression and random forest on a stratified held-out split,
prints accuracy + confusion matrix for each, picks the better model, and writes:
  - models/complexity_classifier.joblib  (fitted estimator + metadata)
  - data/classifier_metrics.json         (accuracy, confusion matrices, winner)

Requires data/prompt_features.json (run `python -m scripts.inspect_dataset` first).

Phase 3.4 reuses ``train_and_save`` from ``scripts.retrain_from_feedback``.

Run:
    python -m scripts.train_classifier
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.classifier.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FEATURES_PATH = DATA_DIR / "prompt_features.json"
METRICS_PATH = DATA_DIR / "classifier_metrics.json"
MODEL_PATH = MODELS_DIR / "complexity_classifier.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET_ACCURACY = 0.80


def _load_xy(features_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = features_path or FEATURES_PATH
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run: python -m scripts.inspect_dataset"
        )
    # Trusted local feature dump from inspect_dataset / retrain — not user upload.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("feature_names") != FEATURE_NAMES:
        raise SystemExit(
            "prompt_features.json feature_names do not match FEATURE_NAMES — "
            "re-run scripts.inspect_dataset."
        )
    rows = payload["rows"]
    x = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r["tier"] for r in rows], dtype=int)
    return x, y


def _evaluate(name: str, model: Any, x_test: Any, y_test: Any) -> dict:
    pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, pred))
    labels = [1, 2, 3]
    cm = confusion_matrix(y_test, pred, labels=labels).tolist()
    print(f"\n=== {name} ===")
    print(
        f"Held-out accuracy: {acc:.1%}  "
        f"({accuracy_score(y_test, pred, normalize=False)}/{len(y_test)} correct)"
    )
    print("Confusion matrix (rows=true, cols=pred) tiers 1/2/3:")
    for i, row in enumerate(cm):
        print(f"  T{labels[i]}  {row}")
    return {"name": name, "accuracy": acc, "confusion_matrix": cm, "labels": labels}


def train_and_save(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_path: Path | None = None,
    metrics_path: Path | None = None,
    n_feedback: int = 0,
) -> dict:
    """Fit LR vs RF, write joblib + metrics. Returns the metrics dict.

    Only write ``model_path`` with bundles produced here — ``joblib.load``
    elsewhere must only load these trusted local trainer outputs.
    """
    model_path = model_path or MODEL_PATH
    metrics_path = metrics_path or METRICS_PATH

    print(f"Loaded {len(y)} examples, {x.shape[1]} features")
    print(f"Stratified split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")

    # Stratify needs ≥2 samples per class; fall back if a tiny smoke set.
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique) >= 2 and int(counts.min()) >= 2
    split_kwargs: dict[str, Any] = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    if can_stratify:
        split_kwargs["stratify"] = y
    else:
        print("WARNING: not enough per-class samples to stratify; using plain split.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, **split_kwargs)
    print(f"Train={len(y_train)}  Test={len(y_test)}")

    # Scale features for LR (RF is scale-invariant; keep it unscaled).
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    forest = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced"
    )

    logistic.fit(x_train, y_train)
    forest.fit(x_train, y_train)

    results = [
        _evaluate("logistic_regression", logistic, x_test, y_test),
        _evaluate("random_forest", forest, x_test, y_test),
    ]
    # Prefer logistic regression on a tie — simpler model for V1 routing.
    winner = max(
        results, key=lambda r: (r["accuracy"], r["name"] == "logistic_regression")
    )
    chosen = logistic if winner["name"] == "logistic_regression" else forest

    print(f"\nWinner: {winner['name']} @ {winner['accuracy']:.1%}")
    if winner["accuracy"] < TARGET_ACCURACY:
        print(f"WARNING: below V1 target of {TARGET_ACCURACY:.0%}.")
    else:
        print(f"Meets V1 target of >{TARGET_ACCURACY:.0%} held-out accuracy.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": chosen,
        "model_name": winner["name"],
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "held_out_accuracy": winner["accuracy"],
        "n_examples": int(len(y)),
        "n_feedback": int(n_feedback),
    }
    # Trusted local path only — never load joblib from untrusted uploads.
    joblib.dump(bundle, model_path)
    print(f"\nWrote model → {model_path}")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_examples": int(len(y)),
        "n_feedback": int(n_feedback),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target_accuracy": TARGET_ACCURACY,
        "models": results,
        "winner": winner["name"],
        "winner_accuracy": winner["accuracy"],
        "model_path": str(model_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics → {metrics_path}")
    return metrics


def main() -> None:
    x, y = _load_xy()
    train_and_save(x, y)


if __name__ == "__main__":
    main()
