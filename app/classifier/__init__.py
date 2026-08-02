"""Complexity classifier package (Phase 2 + 3.4 feedback).

2.2: feature extraction from raw prompts.
2.3: trained model load + tier prediction.
2.4: routing map builds on predicted tiers.
3.4: routing-failure feedback → weekly retrain flywheel.
"""

from app.classifier.features import FEATURE_NAMES, extract_features, load_dataset
from app.classifier.feedback import (
    corrected_tier,
    load_feedback,
    merge_training_rows,
    record_routing_failure_example,
)
from app.classifier.model import load_classifier, predict_tier

__all__ = [
    "FEATURE_NAMES",
    "corrected_tier",
    "extract_features",
    "load_classifier",
    "load_dataset",
    "load_feedback",
    "merge_training_rows",
    "predict_tier",
    "record_routing_failure_example",
]
