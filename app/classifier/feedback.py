"""Phase 3.4 — turn routing failures into classifier training examples.

When verification catches a routing failure, the cheap model was not good
enough for that prompt. The flywheel records a corrected complexity-tier
label so the weekly retrain can push similar prompts upward.

Relabel rule (documented, intentional):
  1. Infer the routed complexity tier from ``routed_model`` via the
     reverse of ``configs/routing_map.yaml`` (model → tier).
  2. If the model is unknown / not in the map, fall back to a use-case
     base tier: extraction→1, summarization→2, classification→2.
  3. Corrected label = min(inferred_tier + 1, 3).

Failure JSONL under ``data/routing_failures.jsonl`` stores only
``prompt_hash`` (no raw prompt). Full prompt text is written here at
failure time while it is still in memory — required to build training
rows. ``data/feedback_prompts.jsonl`` is gitignored local data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.router.map import load_routing_map

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FEEDBACK_PATH = _ROOT / "data" / "feedback_prompts.jsonl"

# Fallback when routed_model is not in the routing map.
USE_CASE_BASE_TIER: dict[str, int] = {
    "extraction": 1,
    "summarization": 2,
    "classification": 2,
}

RELABEL_RULE = "bump_routed_tier_plus_1_cap_3"


def infer_routed_tier(
    routed_model: str,
    *,
    routing_map_path: str | None = None,
) -> int | None:
    """Reverse-lookup complexity tier for a MODEL_REGISTRY key, or None."""
    if not routed_model or routed_model == "unknown":
        return None
    mapping = load_routing_map(routing_map_path)
    for tier, model_key in mapping.items():
        if model_key == routed_model:
            return int(tier)
    return None


def corrected_tier(
    *,
    routed_model: str,
    use_case: str,
    routing_map_path: str | None = None,
) -> tuple[int, int]:
    """Return (inferred_routed_tier, corrected_tier) for a routing failure.

    Corrected tier is always in {1, 2, 3} and >= inferred tier.
    """
    inferred = infer_routed_tier(routed_model, routing_map_path=routing_map_path)
    if inferred is None:
        inferred = USE_CASE_BASE_TIER.get(use_case, 1)
    corrected = min(inferred + 1, 3)
    return inferred, corrected


def load_feedback(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load accumulated feedback JSONL rows (empty list if missing).

    Skips blank / corrupt lines so one bad row cannot block retrain.
    """
    path = Path(path) if path else DEFAULT_FEEDBACK_PATH
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.error("failed to read feedback log %s: %s", path, exc)
        return []
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            # Trusted local feedback file written by this module — not user upload.
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skip corrupt feedback line %s:%s", path, line_no)
    return rows


def _feedback_has_hash(path: Path, prompt_hash: str) -> bool:
    """True if ``prompt_hash`` already appears in the feedback JSONL."""
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("prompt_hash") == prompt_hash:
                    return True
    except OSError as exc:
        logger.error("failed to scan feedback log %s: %s", path, exc)
        return False
    return False


def record_routing_failure_example(
    prompt: str,
    *,
    routed_model: str,
    use_case: str,
    prompt_hash: str,
    feedback_path: Path | str | None = None,
    routing_map_path: str | None = None,
) -> dict[str, Any] | None:
    """Append one labeled feedback example from a live routing failure.

    Returns the written record, or ``None`` if skipped (empty prompt or
    duplicate ``prompt_hash`` already present). Never raises on disk errors
    after logging — callers on the verify path must stay resilient.
    """
    if not prompt or not str(prompt).strip():
        logger.warning("skip feedback: empty prompt (hash=%s)", prompt_hash)
        return None

    path = Path(feedback_path) if feedback_path else DEFAULT_FEEDBACK_PATH
    inferred, tier = corrected_tier(
        routed_model=routed_model,
        use_case=use_case,
        routing_map_path=routing_map_path,
    )

    # Dedup by prompt_hash — one correction per prompt is enough for V1.
    if _feedback_has_hash(path, prompt_hash):
        logger.info("skip feedback: duplicate prompt_hash=%s", prompt_hash)
        return None

    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "routing_failure",
        "prompt": prompt,
        "tier": tier,
        "prompt_hash": prompt_hash,
        "routed_model": routed_model,
        "inferred_routed_tier": inferred,
        "relabel_rule": RELABEL_RULE,
        "use_case": use_case,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError as exc:
        logger.error("failed to write feedback log %s: %s", path, exc)
        return None

    logger.info(
        "classifier_feedback prompt_hash=%s inferred_tier=%s corrected_tier=%s "
        "routed=%s use_case=%s",
        prompt_hash,
        inferred,
        tier,
        routed_model,
        use_case,
    )
    return record


def merge_training_rows(
    base_rows: list[dict],
    feedback_rows: list[dict],
) -> list[dict]:
    """Merge base labeled prompts with feedback; feedback wins on same prompt.

    Returns ``[{prompt, tier}, ...]`` suitable for feature extraction /
    training. Feedback rows may carry extra keys; only prompt+tier are kept.
    """
    by_prompt: dict[str, int] = {}
    order: list[str] = []

    for row in base_rows:
        prompt = row.get("prompt")
        tier = row.get("tier")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(tier, int) or tier not in (1, 2, 3):
            continue
        if prompt not in by_prompt:
            order.append(prompt)
        by_prompt[prompt] = tier

    for row in feedback_rows:
        prompt = row.get("prompt")
        tier = row.get("tier")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(tier, int) or tier not in (1, 2, 3):
            continue
        if prompt not in by_prompt:
            order.append(prompt)
        by_prompt[prompt] = tier  # feedback overrides base label

    return [{"prompt": p, "tier": by_prompt[p]} for p in order]
