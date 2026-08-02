"""Phase 4.2 — aggregate cost / quality / escalation metrics from audit DB.

GPT-4o counterfactual (honest, documented)
-----------------------------------------
Prefer exact math when ``input_tokens`` / ``output_tokens`` are present on the
audit row::

    gpt4o_cost = in * MODEL_REGISTRY["gpt-4o"].cost_per_input_token
               + out * MODEL_REGISTRY["gpt-4o"].cost_per_output_token

When tokens are missing (Phase 4.1 rows written before the nullable columns),
fall back to a fixed typical-request size
(``FALLBACK_INPUT_TOKENS`` / ``FALLBACK_OUTPUT_TOKENS``) priced at GPT-4o
registry rates. That is an approximation — not reverse-engineered from the
routed model's ``cost`` (which is $0 for ``llama-local`` and would understate
savings). Demo seed rows always include token counts.

Security: aggregates only — never select or return raw prompt text. All SQL
uses ``?`` parameter binds. Dashboard should bind Streamlit to localhost.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from app.audit.store import connect, count_requests, init_db, log_completion
from app.providers.registry import MODEL_REGISTRY

# Typical short completion used only when audit tokens are NULL.
FALLBACK_INPUT_TOKENS = 500
FALLBACK_OUTPUT_TOKENS = 250

GPT4O_KEY = "gpt-4o"

GPT4O_COUNTERFACTUAL_NOTE = (
    "Hypothetical GPT-4o cost uses MODEL_REGISTRY['gpt-4o'] list pricing on "
    "per-row input/output tokens when present; otherwise falls back to "
    f"{FALLBACK_INPUT_TOKENS} in / {FALLBACK_OUTPUT_TOKENS} out tokens per request."
)

PeriodGrain = Literal["day", "week"]


@dataclass(frozen=True)
class DashboardSummary:
    """Headline numbers for the cost dashboard."""

    request_count: int
    actual_cost_usd: float
    gpt4o_cost_usd: float
    savings_usd: float
    savings_pct: float
    escalation_count: int
    escalation_rate: float
    scored_count: int
    mean_quality_score: float | None
    tokens_known_count: int
    tokens_fallback_count: int
    counterfactual_note: str


@dataclass(frozen=True)
class CostPeriodRow:
    period: str  # YYYY-MM-DD (day) or ISO week start
    actual_cost_usd: float
    gpt4o_cost_usd: float
    request_count: int


@dataclass(frozen=True)
class RoutingShare:
    routed_model: str
    request_count: int
    share: float


@dataclass(frozen=True)
class QualityBucket:
    """One bin in the quality-score distribution."""

    label: str
    count: int


@dataclass(frozen=True)
class EscalationPeriodRow:
    period: str
    request_count: int
    escalation_count: int
    escalation_rate: float


def _gpt4o_config():
    cfg = MODEL_REGISTRY.get(GPT4O_KEY)
    if cfg is None:
        raise KeyError(f"{GPT4O_KEY!r} missing from MODEL_REGISTRY")
    return cfg


def gpt4o_cost_for_tokens(input_tokens: int, output_tokens: int) -> float:
    """USD cost if the same token counts ran on GPT-4o (registry pricing)."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    cfg = _gpt4o_config()
    return (
        input_tokens * cfg.cost_per_input_token
        + output_tokens * cfg.cost_per_output_token
    )


def _row_gpt4o_cost(
    input_tokens: int | None, output_tokens: int | None
) -> tuple[float, bool]:
    """Return (gpt4o_usd, used_fallback)."""
    if input_tokens is not None and output_tokens is not None:
        return gpt4o_cost_for_tokens(input_tokens, output_tokens), False
    return (
        gpt4o_cost_for_tokens(FALLBACK_INPUT_TOKENS, FALLBACK_OUTPUT_TOKENS),
        True,
    )


_SELECT_COST_ROWS_SQL = """
SELECT
    cost,
    input_tokens,
    output_tokens,
    escalated,
    verifier_quality_score
FROM requests
"""


def compute_summary(*, db_path: str | Path | None = None) -> DashboardSummary:
    """Actual vs GPT-4o totals, savings, escalation + quality headlines."""
    init_db(db_path)
    actual = 0.0
    gpt4o = 0.0
    n = 0
    esc = 0
    scored = 0
    score_sum = 0.0
    tokens_known = 0
    tokens_fallback = 0

    with connect(db_path) as conn:
        for r in conn.execute(_SELECT_COST_ROWS_SQL):
            n += 1
            actual += float(r["cost"])
            in_tok = None if r["input_tokens"] is None else int(r["input_tokens"])
            out_tok = None if r["output_tokens"] is None else int(r["output_tokens"])
            row_gpt4o, used_fb = _row_gpt4o_cost(in_tok, out_tok)
            gpt4o += row_gpt4o
            if used_fb:
                tokens_fallback += 1
            else:
                tokens_known += 1
            if int(r["escalated"]):
                esc += 1
            if r["verifier_quality_score"] is not None:
                scored += 1
                score_sum += float(r["verifier_quality_score"])

    savings = gpt4o - actual
    if gpt4o > 0:
        savings_pct = (savings / gpt4o) * 100.0
    else:
        savings_pct = 0.0
    mean_q = (score_sum / scored) if scored else None
    rate = (esc / n) if n else 0.0

    return DashboardSummary(
        request_count=n,
        actual_cost_usd=actual,
        gpt4o_cost_usd=gpt4o,
        savings_usd=savings,
        savings_pct=savings_pct,
        escalation_count=esc,
        escalation_rate=rate,
        scored_count=scored,
        mean_quality_score=mean_q,
        tokens_known_count=tokens_known,
        tokens_fallback_count=tokens_fallback,
        counterfactual_note=GPT4O_COUNTERFACTUAL_NOTE,
    )


# Fixed SQL only — grain selects which constant query to run (no user input).
_COST_BY_DAY_SQL = """
SELECT
    date(timestamp) AS period,
    cost,
    input_tokens,
    output_tokens
FROM requests
WHERE timestamp IS NOT NULL
ORDER BY period ASC
"""

_COST_BY_WEEK_SQL = """
SELECT
    date(timestamp, 'weekday 0', '-6 days') AS period,
    cost,
    input_tokens,
    output_tokens
FROM requests
WHERE timestamp IS NOT NULL
ORDER BY period ASC
"""


def cost_by_period(
    *,
    grain: PeriodGrain,
    db_path: str | Path | None = None,
) -> list[CostPeriodRow]:
    """Actual + GPT-4o cost rolled up by calendar day or week."""
    init_db(db_path)
    if grain == "day":
        sql = _COST_BY_DAY_SQL
    elif grain == "week":
        sql = _COST_BY_WEEK_SQL
    else:
        raise ValueError(f"unknown grain: {grain!r}")

    # Aggregate in Python so GPT-4o counterfactual math stays in one place.
    buckets: dict[str, list[tuple[float, float]]] = {}
    with connect(db_path) as conn:
        for r in conn.execute(sql):
            period = str(r["period"] or "unknown")
            in_tok = None if r["input_tokens"] is None else int(r["input_tokens"])
            out_tok = None if r["output_tokens"] is None else int(r["output_tokens"])
            gpt4o, _ = _row_gpt4o_cost(in_tok, out_tok)
            buckets.setdefault(period, []).append((float(r["cost"]), gpt4o))

    rows: list[CostPeriodRow] = []
    for period in sorted(buckets):
        pairs = buckets[period]
        rows.append(
            CostPeriodRow(
                period=period,
                actual_cost_usd=sum(a for a, _ in pairs),
                gpt4o_cost_usd=sum(g for _, g in pairs),
                request_count=len(pairs),
            )
        )
    return rows


def cost_by_day(*, db_path: str | Path | None = None) -> list[CostPeriodRow]:
    return cost_by_period(grain="day", db_path=db_path)


def cost_by_week(*, db_path: str | Path | None = None) -> list[CostPeriodRow]:
    return cost_by_period(grain="week", db_path=db_path)


_ROUTING_SQL = """
SELECT routed_model, COUNT(*) AS n
FROM requests
GROUP BY routed_model
ORDER BY n DESC
"""


def routing_distribution(
    *, db_path: str | Path | None = None
) -> list[RoutingShare]:
    """Pie-chart data: share of requests per routed_model."""
    init_db(db_path)
    with connect(db_path) as conn:
        raw = [(str(r["routed_model"]), int(r["n"])) for r in conn.execute(_ROUTING_SQL)]
    total = sum(n for _, n in raw) or 1
    return [
        RoutingShare(routed_model=m, request_count=n, share=n / total) for m, n in raw
    ]


_QUALITY_SQL = """
SELECT verifier_quality_score
FROM requests
WHERE verifier_quality_score IS NOT NULL
"""


def quality_score_distribution(
    *, db_path: str | Path | None = None
) -> list[QualityBucket]:
    """Histogram-friendly buckets over verifier_quality_score.

    Scores are use-case dependent (0–1 agreement vs 1–5 judge). We bucket
    by rounded value so mixed scales still show a distribution shape.
    """
    init_db(db_path)
    counts: dict[str, int] = {}
    with connect(db_path) as conn:
        for r in conn.execute(_QUALITY_SQL):
            score = float(r["verifier_quality_score"])
            # One decimal place keeps 0/1 and 1–5 scales readable.
            label = f"{score:.1f}"
            counts[label] = counts.get(label, 0) + 1
    return [
        QualityBucket(label=label, count=counts[label])
        for label in sorted(counts, key=lambda s: float(s))
    ]


def escalation_rate_by_day(
    *, db_path: str | Path | None = None
) -> list[EscalationPeriodRow]:
    """Daily escalation rate (escalated / requests) over time."""
    init_db(db_path)
    sql = """
SELECT
    date(timestamp) AS period,
    COUNT(*) AS n,
    SUM(CASE WHEN escalated = 1 THEN 1 ELSE 0 END) AS esc
FROM requests
GROUP BY date(timestamp)
ORDER BY period ASC
"""
    rows: list[EscalationPeriodRow] = []
    with connect(db_path) as conn:
        for r in conn.execute(sql):
            n = int(r["n"])
            esc = int(r["esc"] or 0)
            rows.append(
                EscalationPeriodRow(
                    period=str(r["period"]),
                    request_count=n,
                    escalation_count=esc,
                    escalation_rate=(esc / n) if n else 0.0,
                )
            )
    return rows


# --- Demo seed (portfolio screenshots; hashes only, no raw prompts) ---------

# Weighted toward tier-1/2 traffic so demo savings vs all-GPT-4o is visible
# (portfolio screenshots). Tier-3 Sonnet rows still appear for the pie chart.
_DEMO_PROFILES: tuple[dict, ...] = (
    {
        "complexity_tier": 1,
        "routed_model": "llama-local",
        "use_case": "extraction",
        "input_tokens": 220,
        "output_tokens": 60,
        "latency": 0.35,
        "score": 1.0,
        "escalated": False,
    },
    {
        "complexity_tier": 1,
        "routed_model": "llama-local",
        "use_case": "extraction",
        "input_tokens": 180,
        "output_tokens": 40,
        "latency": 0.28,
        "score": 1.0,
        "escalated": False,
    },
    {
        "complexity_tier": 1,
        "routed_model": "llama-local",
        "use_case": "extraction",
        "input_tokens": 300,
        "output_tokens": 80,
        "latency": 0.4,
        "score": 1.0,
        "escalated": False,
    },
    {
        "complexity_tier": 2,
        "routed_model": "gemini-flash",
        "use_case": "summarization",
        "input_tokens": 900,
        "output_tokens": 220,
        "latency": 0.9,
        "score": 4.2,
        "escalated": False,
    },
    {
        "complexity_tier": 2,
        "routed_model": "gemini-flash",
        "use_case": "classification",
        "input_tokens": 400,
        "output_tokens": 20,
        "latency": 0.55,
        "score": 1.0,
        "escalated": False,
    },
    {
        "complexity_tier": 2,
        "routed_model": "gemini-flash",
        "use_case": "summarization",
        "input_tokens": 1100,
        "output_tokens": 280,
        "latency": 1.1,
        "score": 3.0,
        "escalated": True,
        "escalation_model": "claude-sonnet",
    },
    {
        "complexity_tier": 3,
        "routed_model": "claude-sonnet",
        "use_case": "reasoning",
        "input_tokens": 1500,
        "output_tokens": 600,
        "latency": 2.1,
        "score": 4.8,
        "escalated": False,
    },
)


def _cost_for_model(model_key: str, input_tokens: int, output_tokens: int) -> float:
    cfg = MODEL_REGISTRY[model_key]
    return (
        input_tokens * cfg.cost_per_input_token
        + output_tokens * cfg.cost_per_output_token
    )


def seed_demo_requests(
    *,
    db_path: str | Path | None = None,
    days: int = 14,
    per_day: int = 6,
) -> int:
    """Insert synthetic audit rows for empty-DB demo / screenshots.

    Only prompt hashes are stored (SHA-256 hex of a synthetic label) — never
    raw prompt text. Returns the number of rows inserted.
    """
    if days < 1 or per_day < 1:
        raise ValueError("days and per_day must be >= 1")
    init_db(db_path)
    if count_requests(db_path=db_path) > 0:
        return 0

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    inserted = 0
    n_profiles = len(_DEMO_PROFILES)

    for d in range(days):
        day = now - timedelta(days=(days - 1 - d))
        for i in range(per_day):
            profile = _DEMO_PROFILES[(d * per_day + i) % n_profiles]
            model = str(profile["routed_model"])
            in_tok = int(profile["input_tokens"])
            out_tok = int(profile["output_tokens"])
            cost = _cost_for_model(model, in_tok, out_tok)
            escalated = bool(profile["escalated"])
            esc_model = profile.get("escalation_model")
            cost_delta = None
            if escalated and esc_model:
                # Extra spend of re-running on the escalation model.
                cost_delta = _cost_for_model(str(esc_model), in_tok, out_tok)

            label = f"demo:{d}:{i}:{model}"
            ph = hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
            ts = (day + timedelta(minutes=15 * i)).isoformat()

            log_completion(
                prompt_hash=ph,
                complexity_tier=int(profile["complexity_tier"]),
                routed_model=model,
                cost=cost,
                latency=float(profile["latency"]),
                verifier_quality_score=float(profile["score"]),
                escalated=escalated,
                escalation_model=esc_model if escalated else None,
                cost_delta=cost_delta,
                use_case=str(profile["use_case"]),
                input_tokens=in_tok,
                output_tokens=out_tok,
                timestamp=ts,
                db_path=db_path,
            )
            inserted += 1
    return inserted
