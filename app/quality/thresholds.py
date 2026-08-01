"""Phase 3.1 — load YAML quality thresholds per use case.

Thresholds live in configs/quality_thresholds.yaml so "good enough" bars can
change without code edits. Registry keys (judge_model / reference_model) are
validated on load when present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.providers.registry import MODEL_REGISTRY

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_THRESHOLDS_PATH = _ROOT / "configs" / "quality_thresholds.yaml"

REQUIRED_USE_CASES = frozenset({"extraction", "summarization", "classification"})
ALLOWED_COMPARISONS = frozenset({">", ">=", "<", "<=", "=="})


@dataclass(frozen=True)
class QualityThreshold:
    """One use-case pass/fail bar from the YAML config."""

    use_case: str
    metric: str
    threshold: float
    comparison: str
    notes: str = ""
    judge_model: str | None = None
    reference_model: str | None = None
    scale_max: float | None = None


def load_quality_thresholds(
    path: str | None = None,
) -> dict[str, QualityThreshold]:
    """Return {use_case_id: QualityThreshold} from the thresholds YAML.

    Always re-reads the file so YAML edits apply without a process restart.
    """
    thresholds_path = Path(path) if path else DEFAULT_THRESHOLDS_PATH
    # Only load project config paths in production callers — path is unconstrained
    # for local smoke/tests; do not pass untrusted user paths here.
    data = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("quality_thresholds.yaml: top level must be a mapping")
    use_cases = data.get("use_cases") or {}
    if not isinstance(use_cases, dict):
        raise ValueError("quality_thresholds.yaml: 'use_cases' must be a mapping")

    resolved: dict[str, QualityThreshold] = {}
    for use_case, entry in use_cases.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"quality_thresholds use_case {use_case!r} must be a mapping"
            )
        try:
            metric = str(entry["metric"])
            threshold = float(entry["threshold"])
            comparison = entry["comparison"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"quality_thresholds use_case {use_case!r} missing/invalid "
                f"metric, threshold, or comparison: {exc}"
            ) from exc

        if comparison not in ALLOWED_COMPARISONS:
            raise ValueError(
                f"quality_thresholds use_case {use_case!r}: comparison "
                f"{comparison!r} not in {sorted(ALLOWED_COMPARISONS)}"
            )

        judge_model = entry.get("judge_model")
        reference_model = entry.get("reference_model")
        if metric == "llm_judge_score" and not judge_model:
            raise ValueError(
                f"quality_thresholds use_case {use_case!r}: "
                f"metric llm_judge_score requires judge_model"
            )
        if metric == "label_agreement" and not reference_model:
            raise ValueError(
                f"quality_thresholds use_case {use_case!r}: "
                f"metric label_agreement requires reference_model"
            )
        for model_key in (judge_model, reference_model):
            if model_key is not None and model_key not in MODEL_REGISTRY:
                raise ValueError(
                    f"quality_thresholds use_case {use_case!r} → "
                    f"{model_key!r} not in MODEL_REGISTRY"
                )

        scale_max = entry.get("scale_max")
        if scale_max is not None:
            scale_max = float(scale_max)
            if scale_max <= 0:
                raise ValueError(
                    f"quality_thresholds use_case {use_case!r}: "
                    f"scale_max must be > 0, got {scale_max}"
                )
            if threshold > scale_max:
                raise ValueError(
                    f"quality_thresholds use_case {use_case!r}: "
                    f"threshold {threshold} exceeds scale_max {scale_max}"
                )

        notes = entry.get("notes") or ""
        if isinstance(notes, str):
            notes = " ".join(notes.split())
        else:
            notes = str(notes)

        resolved[str(use_case)] = QualityThreshold(
            use_case=str(use_case),
            metric=metric,
            threshold=threshold,
            comparison=str(comparison),
            notes=notes,
            judge_model=str(judge_model) if judge_model is not None else None,
            reference_model=(
                str(reference_model) if reference_model is not None else None
            ),
            scale_max=scale_max,
        )

    missing = REQUIRED_USE_CASES - set(resolved)
    if missing:
        raise ValueError(
            f"quality_thresholds must define {sorted(REQUIRED_USE_CASES)}, "
            f"missing {sorted(missing)}"
        )
    return resolved


def threshold_for(use_case: str, path: str | None = None) -> QualityThreshold:
    """Look up the QualityThreshold configured for a use case."""
    thresholds = load_quality_thresholds(path)
    if use_case not in thresholds:
        raise ValueError(
            f"No quality threshold for use_case {use_case!r}; "
            f"known: {sorted(thresholds)}"
        )
    return thresholds[use_case]
