"""Phase 2.3 — train a complexity-tier classifier on extracted features.

Compares logistic regression and random forest on a stratified held-out split,
prints accuracy + confusion matrix for each, picks the better model, and writes:
  - models/complexity_classifier.joblib  (fitted estimator + metadata)
  - data/classifier_metrics.json         (accuracy, confusion matrices, winner)

Requires data/prompt_features.json (run `python -m scripts.inspect_dataset` first).

Run:
    python -m scripts.train_classifier
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

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


def _load_xy() -> tuple[np.ndarray, np.ndarray]:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Missing {FEATURES_PATH}. Run: python -m scripts.inspect_dataset"
        )
    payload = json.loads(FEATURES_PATH.read_text())
    if payload.get("feature_names") != FEATURE_NAMES:
        raise SystemExit(
            "prompt_features.json feature_names do not match FEATURE_NAMES — "
            "re-run scripts.inspect_dataset."
        )
    rows = payload["rows"]
    x = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r["tier"] for r in rows], dtype=int)
    return x, y


def _evaluate(name: str, model, x_test, y_test) -> dict:
    pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, pred))
    labels = [1, 2, 3]
    cm = confusion_matrix(y_test, pred, labels=labels).tolist()
    print(f"\n=== {name} ===")
    print(f"Held-out accuracy: {acc:.1%}  ({accuracy_score(y_test, pred, normalize=False)}"
          f"/{len(y_test)} correct)")
    print("Confusion matrix (rows=true, cols=pred) tiers 1/2/3:")
    for i, row in enumerate(cm):
        print(f"  T{labels[i]}  {row}")
    return {"name": name, "accuracy": acc, "confusion_matrix": cm, "labels": labels}


def main() -> None:
    x, y = _load_xy()
    print(f"Loaded {len(y)} examples, {x.shape[1]} features")
    print(f"Stratified split: test_size={TEST_SIZE}, random_state={RANDOM_STATE}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
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
    winner = max(results, key=lambda r: (r["accuracy"], r["name"] == "logistic_regression"))
    chosen = logistic if winner["name"] == "logistic_regression" else forest

    print(f"\nWinner: {winner['name']} @ {winner['accuracy']:.1%}")
    if winner["accuracy"] < TARGET_ACCURACY:
        print(f"WARNING: below V1 target of {TARGET_ACCURACY:.0%}.")
    else:
        print(f"Meets V1 target of >{TARGET_ACCURACY:.0%} held-out accuracy.")

    MODELS_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": chosen,
        "model_name": winner["name"],
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "held_out_accuracy": winner["accuracy"],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nWrote model → {MODEL_PATH}")

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_examples": int(len(y)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "target_accuracy": TARGET_ACCURACY,
        "models": results,
        "winner": winner["name"],
        "winner_accuracy": winner["accuracy"],
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote metrics → {METRICS_PATH}")


if __name__ == "__main__":
    main()
