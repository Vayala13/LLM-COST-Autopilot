"""Phase 2.2 — feature extraction for the complexity classifier.

Turns a raw prompt string into a fixed-length numeric feature vector. The
features mirror the `signals` block in configs/complexity_tiers.yaml, so the
verb lists are loaded from that file rather than hardcoded here — edit the YAML
to reshape the signal, no code change needed.

Feature vector (see FEATURE_NAMES for order):
    token_count            words split on whitespace
    char_count             raw length
    tier1_verbs            count of Tier 1 instruction verbs present
    tier2_verbs            count of Tier 2 instruction verbs present
    tier3_verbs            count of Tier 3 instruction verbs present
    constraint_count       explicit constraints (tone/length/format/audience)
    context_provided       1 if the prompt embeds the content it operates on
    output_format_complex  0/1/2 hint at how structured the output should be
    reasoning_required     1 if step-by-step / judgment is demanded
    question_count         number of '?' characters
    has_numbers            1 if any digit is present
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
_TIERS_YAML = _ROOT / "configs" / "complexity_tiers.yaml"
_DATASET = _ROOT / "data" / "labeled_prompts.jsonl"

FEATURE_NAMES = [
    "token_count",
    "char_count",
    "tier1_verbs",
    "tier2_verbs",
    "tier3_verbs",
    "constraint_count",
    "context_provided",
    "output_format_complex",
    "reasoning_required",
    "question_count",
    "has_numbers",
]

# Signal keywords that aren't tier-specific verbs. Kept here (not the YAML)
# because they describe generic linguistic cues, not routing intent.
_CONSTRAINT_CUES = [
    "in one sentence", "one sentence", "two sentences", "in a phrase",
    "in one line", "one word", "briefly", "short", "concise", "concisely",
    "professional tone", "formal", "informal", "polite", "friendly", "warm",
    "for a 10-year-old", "a 10-year-old", "for a beginner", "beginner",
    "no more than", "at least", "exactly", "step by step", "step-by-step",
    "bullet", "numbered", "as json", "csv", "in yyyy-mm-dd", "format",
    "4-line", "30-second", "one-sentence", "tailored",
]
_REASONING_CUES = [
    "show your work", "show your reasoning", "step by step", "step-by-step",
    "explain your reasoning", "walk through", "walk me through", "reason it",
    "reason through", "justify", "explain how you would", "how did you",
    "and explain", "defend your", "weigh the", "trade-off", "trade-offs",
    "recommend", "analyze", "evaluate", "synthesize", "argue both sides",
]


@lru_cache(maxsize=1)
def _verb_lists() -> dict[str, list[str]]:
    """Load the tier verb lists from the complexity tiers YAML."""
    data = yaml.safe_load(_TIERS_YAML.read_text())
    verbs = data["signals"]["instruction_verbs"]
    return {
        "tier1_verbs": [v.lower() for v in verbs["tier_1"]],
        "tier2_verbs": [v.lower() for v in verbs["tier_2"]],
        "tier3_verbs": [v.lower() for v in verbs["tier_3"]],
    }


def _count_verbs(words: set[str], verbs: list[str]) -> int:
    return sum(1 for v in verbs if v in words)


def _count_cues(text: str, cues: list[str]) -> int:
    return sum(1 for c in cues if c in text)


def _output_format_complexity(text: str) -> int:
    """0 = short scalar output, 1 = a sentence/paragraph, 2 = structured/long-form."""
    structured = ["json", "csv", "table", "list", "bullet", "numbered",
                  "plan", "outline", "email", "letter", "poem", "story",
                  "speech", "essay", "paragraph", "cover letter"]
    scalar = ["convert", "extract", "what is", "how many", "round", "spell",
              "capital of", "reverse", "count how many"]
    lower = text.lower()
    if any(s in lower for s in structured):
        return 2
    if any(s in lower for s in scalar):
        return 0
    return 1


def extract_features(prompt: str) -> dict[str, int]:
    """Return the feature dict for a single prompt (order = FEATURE_NAMES)."""
    lower = prompt.lower()
    words = set(re.findall(r"[a-z]+", lower))
    verbs = _verb_lists()

    context_provided = int(
        "'" in prompt or '"' in prompt or ": " in prompt or "given " in lower
    )

    return {
        "token_count": len(prompt.split()),
        "char_count": len(prompt),
        "tier1_verbs": _count_verbs(words, verbs["tier1_verbs"]),
        "tier2_verbs": _count_verbs(words, verbs["tier2_verbs"]),
        "tier3_verbs": _count_verbs(words, verbs["tier3_verbs"]),
        "constraint_count": _count_cues(lower, _CONSTRAINT_CUES),
        "context_provided": context_provided,
        "output_format_complex": _output_format_complexity(prompt),
        "reasoning_required": int(_count_cues(lower, _REASONING_CUES) > 0),
        "question_count": prompt.count("?"),
        "has_numbers": int(bool(re.search(r"\d", prompt))),
    }


def load_dataset(path: Path | None = None) -> list[dict]:
    """Load the labeled prompt dataset as a list of {prompt, tier} dicts."""
    path = path or _DATASET
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
