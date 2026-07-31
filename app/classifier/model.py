"""Phase 2.3 — load a trained complexity classifier and predict tiers.

The training script writes a joblib bundle with the fitted estimator plus the
feature-name list so inference stays aligned with the vector used at train time.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from app.classifier.features import FEATURE_NAMES, extract_features

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = _ROOT / "models" / "complexity_classifier.joblib"


def load_classifier(path: Path | None = None) -> dict:
    """Load the joblib bundle saved by scripts/train_classifier.py.

    Only load bundles produced by this project — joblib/pickle can execute
    code from untrusted files.
    """
    path = path or DEFAULT_MODEL_PATH
    bundle = joblib.load(path)
    if bundle.get("feature_names") != FEATURE_NAMES:
        raise ValueError(
            "Saved feature_names do not match FEATURE_NAMES — retrain the model."
        )
    return bundle


def predict_tier(prompt: str, bundle: dict | None = None) -> int:
    """Return predicted complexity tier (1, 2, or 3) for a raw prompt."""
    if bundle is None:
        bundle = load_classifier()
    feats = extract_features(prompt)
    x = [[feats[n] for n in FEATURE_NAMES]]
    return int(bundle["model"].predict(x)[0])
