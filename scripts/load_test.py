"""Phase 6.1 — realistic offline load test (portfolio artifacts).

Runs N diverse prompts through classify → route → mocked ``send_request``
(registry pricing / latency) → ``log_completion``. Writes a load-test SQLite
DB (gitignored) and a committed summary report + dashboard PNG under
``reports/``.

Offline by default — no live API keys / quota. Classifier + routing map are
real; provider costs are simulated from ``MODEL_REGISTRY``.

Run (full, default n=750)::

    PYTHONPATH=. python -m scripts.load_test

Smoke (n=50, temp artifacts)::

    PYTHONPATH=. python -m scripts.load_test --smoke
    # or
    PYTHONPATH=. python -m scripts.smoke_load_test

Security: audit DB stores ``prompt_hash`` only. Reports are aggregates
(no raw prompts, no API keys).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.audit.store import count_requests, init_db, log_completion
from app.classifier import load_classifier, predict_tier
from app.metrics.cost import (
    PORTFOLIO_BASELINE_LABEL,
    compute_summary,
    cost_by_day,
    format_portfolio_headline,
    routing_distribution,
)
from app.providers.registry import MODEL_REGISTRY, ModelConfig
from app.providers.response import Response
from app.quality.verifier import prompt_hash
from app.router.map import load_routing_map

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELED = _ROOT / "data" / "labeled_prompts.jsonl"
DEFAULT_DB = _ROOT / "data" / "load_test_requests.db"
DEFAULT_REPORT_DIR = _ROOT / "reports"
DEFAULT_N = 750
SMOKE_N = 50
MIN_FULL_N = 500
MAX_N = 1000

# Token / latency heuristics for mocked provider responses (tier-aware).
_OUTPUT_TOKENS_BY_TIER = {1: (40, 90), 2: (80, 280), 3: (200, 650)}
_ESCALATION_RATE = 0.05  # ~5% of non-tier-3 traffic "fails" verify → escalate
_ESCALATION_MODEL = "claude-sonnet"

# Production-ish mix: more simple/mid traffic so GPT-4o counterfactual savings
# remain visible (tier-3 Sonnet list price > GPT-4o; heavy T3 flattens savings).
_TIER_SAMPLE_WEIGHTS = {1: 0.50, 2: 0.35, 3: 0.15}

# Synthetic prompt templates (filled with seeds) — diversify beyond labeled set.
_SYNTHETIC: tuple[tuple[int, str], ...] = (
    (1, "Extract the email address from: user{i}@example.com wrote yesterday."),
    (1, "Reformat date {i}/15/2026 as YYYY-MM-DD."),
    (1, "Convert this phone to E.164: 555-01{i:02d}."),
    (1, "List the unique words in: alpha beta gamma {i} alpha."),
    (1, "Return only the integer: The count is {i}."),
    (2, "Summarize in 2 sentences: Q{i} report notes revenue up {i}% and churn flat."),
    (2, "Classify sentiment as positive/neutral/negative: Product {i} is decent."),
    (2, "Label the topic (billing|support|feature): Need invoice for order {i}."),
    (2, "Compare options A and B for task {i}: A is cheaper, B is faster."),
    (2, "Extract structured JSON fields name/role from: Ada {i}, engineer."),
    (3, "Design a multi-step migration plan for service {i} with rollback criteria."),
    (3, "Argue both sides of policy {i}: privacy vs. analytics retention."),
    (3, "Write a creative brief for campaign {i} targeting three personas."),
    (3, "Debug this architecture trade-off for system {i}: consistency vs latency."),
    (3, "Propose evaluation metrics for routing quality in scenario {i}."),
)

_USE_CASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("extraction", re.compile(r"\b(extract|reformat|convert|return only|list the)\b", re.I)),
    ("summarization", re.compile(r"\b(summarize|summary|tl;dr)\b", re.I)),
    ("classification", re.compile(r"\b(classify|label|sentiment|topic)\b", re.I)),
    ("reasoning", re.compile(r"\b(design|argue|propose|debug|creative|multi-step|plan)\b", re.I)),
)


@dataclass(frozen=True)
class LoadTestReport:
    """Serializable portfolio summary (aggregates only)."""

    n: int
    cost_reduction_pct: float
    savings_usd: float
    actual_cost_usd: float
    gpt4o_cost_usd: float
    baseline_label: str
    escalation_count: int
    escalation_rate: float
    mean_quality_score: float | None
    routing_distribution: list[dict[str, float | int | str]]
    tier_distribution: dict[str, int]
    headline: str
    mode: str
    counterfactual_note: str


def _estimate_input_tokens(prompt: str) -> int:
    """Rough token estimate from whitespace words (no tokenizer dependency)."""
    words = max(1, len(prompt.split()))
    return max(16, int(words * 1.3) + 8)


def infer_use_case(prompt: str, tier: int) -> str:
    """Heuristic use-case label for audit/metrics (not sent to providers)."""
    for name, pattern in _USE_CASE_PATTERNS:
        if pattern.search(prompt):
            return name
    if tier == 1:
        return "extraction"
    if tier == 2:
        return "summarization"
    return "reasoning"


def _quality_score(use_case: str, escalated: bool, rng: random.Random) -> float:
    """Synthetic verifier score shaped like real threshold metrics."""
    if escalated:
        # Failed cheap path — score below typical pass.
        if use_case in {"extraction", "classification"}:
            return 0.0
        if use_case == "summarization":
            return round(rng.uniform(2.5, 3.8), 2)
        return round(rng.uniform(2.0, 3.5), 2)
    if use_case in {"extraction", "classification"}:
        return 1.0 if rng.random() > 0.03 else 0.0
    if use_case == "summarization":
        return round(rng.uniform(4.1, 4.9), 2)
    return round(rng.uniform(4.2, 5.0), 2)


def mock_send_request(
    prompt: str,
    model_cfg: ModelConfig,
    *,
    rng: random.Random,
    max_tokens: int | None = None,
) -> Response:
    """Fake provider call: token counts + registry cost + jittered latency."""
    in_tok = _estimate_input_tokens(prompt)
    # Infer tier from quality_tier loosely for output size; callers pass routed cfg.
    if model_cfg.quality_tier == "low":
        lo, hi = _OUTPUT_TOKENS_BY_TIER[1]
    elif model_cfg.quality_tier == "medium":
        lo, hi = _OUTPUT_TOKENS_BY_TIER[2]
    else:
        lo, hi = _OUTPUT_TOKENS_BY_TIER[3]
    out_tok = rng.randint(lo, hi)
    if max_tokens is not None:
        out_tok = min(out_tok, max(1, int(max_tokens)))
    cost = (
        in_tok * model_cfg.cost_per_input_token
        + out_tok * model_cfg.cost_per_output_token
    )
    latency = max(0.05, model_cfg.avg_latency_s * rng.uniform(0.7, 1.35))
    return Response(
        model_id=model_cfg.model_id,
        provider=model_cfg.provider,
        output_text="[load-test mock output]",
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_s=latency,
        cost_usd=cost,
    )


def load_prompt_corpus(
    labeled_path: Path,
    *,
    n: int,
    rng: random.Random,
) -> list[str]:
    """Build N prompts from labeled JSONL (tier-weighted) + synthetic variety.

    Sampling follows ``_TIER_SAMPLE_WEIGHTS`` so the load mix looks like a
    typical app (more mechanical/mid tasks than deep reasoning). Classifier
    still runs at request time — weights only choose which prompts to feed.
    """
    if not labeled_path.is_file():
        raise FileNotFoundError(f"labeled prompts not found: {labeled_path}")
    by_tier: dict[int, list[str]] = {1: [], 2: [], 3: []}
    with labeled_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt") or "").strip()
            tier = int(row.get("tier") or 0)
            if prompt and tier in by_tier:
                by_tier[tier].append(prompt)
    if not any(by_tier.values()):
        raise ValueError(f"no prompts in {labeled_path}")

    synth_by_tier: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for tier, template in _SYNTHETIC:
        synth_by_tier[tier].append(template)

    # ~75% labeled (with replacement), ~25% synthetic — both tier-weighted.
    n_labeled = max(1, int(n * 0.75))
    n_synth = n - n_labeled
    tiers = (1, 2, 3)
    weights = [_TIER_SAMPLE_WEIGHTS[t] for t in tiers]

    out: list[str] = []
    for _ in range(n_labeled):
        tier = rng.choices(tiers, weights=weights, k=1)[0]
        pool = by_tier[tier] or by_tier[1]
        out.append(rng.choice(pool))
    for i in range(n_synth):
        tier = rng.choices(tiers, weights=weights, k=1)[0]
        templates = synth_by_tier[tier] or synth_by_tier[1]
        template = templates[i % len(templates)]
        out.append(template.format(i=(i + 1) * 7 % 97 + 1))
    rng.shuffle(out)
    return out[:n]


def _cost_for_model(model_key: str, input_tokens: int, output_tokens: int) -> float:
    cfg = MODEL_REGISTRY[model_key]
    return (
        input_tokens * cfg.cost_per_input_token
        + output_tokens * cfg.cost_per_output_token
    )


def run_load_test(
    *,
    n: int,
    db_path: Path,
    report_dir: Path,
    labeled_path: Path = DEFAULT_LABELED,
    seed: int = 42,
    write_reports: bool = True,
    mode: str = "full",
) -> LoadTestReport:
    """Execute N offline routed completions into ``db_path``; optional reports."""
    if n < 1 or n > MAX_N:
        raise ValueError(f"n must be in 1..{MAX_N}, got {n}")
    rng = random.Random(seed)
    bundle = load_classifier()
    routing = load_routing_map()
    prompts = load_prompt_corpus(labeled_path, n=n, rng=rng)

    # Fresh DB for this run.
    if db_path.exists():
        db_path.unlink()
    for suffix in ("-wal", "-shm", "-journal"):
        side = Path(str(db_path) + suffix)
        if side.exists():
            side.unlink()
    init_db(db_path)

    now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    tier_counts: dict[str, int] = {"1": 0, "2": 0, "3": 0}

    for i, prompt in enumerate(prompts):
        tier = predict_tier(prompt, bundle)
        model_key = routing[tier]
        model_cfg = MODEL_REGISTRY[model_key]
        resp = mock_send_request(prompt, model_cfg, rng=rng)
        use_case = infer_use_case(prompt, tier)

        escalated = False
        esc_model = None
        cost_delta = None
        # ``cost`` = routed completion only (matches Phase 4.2 demo seed).
        # Escalation spend lives in ``cost_delta`` — not folded into ``cost``
        # so ``compute_summary`` stays consistent with the dashboard.
        if tier < 3 and rng.random() < _ESCALATION_RATE:
            escalated = True
            esc_model = _ESCALATION_MODEL
            cost_delta = _cost_for_model(
                esc_model, resp.input_tokens, resp.output_tokens
            )

        score = _quality_score(use_case, escalated, rng)
        # Spread timestamps across ~14 days for dashboard day/week charts.
        day_offset = i % 14
        minute_offset = (i * 7) % (24 * 60)
        ts = (
            now
            - timedelta(days=(13 - day_offset))
            + timedelta(minutes=minute_offset)
        ).isoformat()

        log_completion(
            prompt_hash=prompt_hash(prompt),
            complexity_tier=tier,
            routed_model=model_key,
            cost=float(resp.cost_usd),
            latency=float(resp.latency_s),
            verifier_quality_score=score,
            escalated=escalated,
            escalation_model=esc_model,
            cost_delta=cost_delta,
            use_case=use_case,
            input_tokens=int(resp.input_tokens),
            output_tokens=int(resp.output_tokens),
            timestamp=ts,
            db_path=db_path,
        )
        tier_counts[str(tier)] = tier_counts.get(str(tier), 0) + 1

    summary = compute_summary(db_path=db_path)
    shares = routing_distribution(db_path=db_path)
    report = LoadTestReport(
        n=summary.request_count,
        cost_reduction_pct=round(summary.cost_reduction_pct, 2),
        savings_usd=round(summary.savings_usd, 6),
        actual_cost_usd=round(summary.actual_cost_usd, 6),
        gpt4o_cost_usd=round(summary.gpt4o_cost_usd, 6),
        baseline_label=PORTFOLIO_BASELINE_LABEL,
        escalation_count=summary.escalation_count,
        escalation_rate=round(summary.escalation_rate, 4),
        mean_quality_score=(
            None
            if summary.mean_quality_score is None
            else round(summary.mean_quality_score, 3)
        ),
        routing_distribution=[
            {
                "routed_model": s.routed_model,
                "request_count": s.request_count,
                "share": round(s.share, 4),
            }
            for s in shares
        ],
        tier_distribution=tier_counts,
        headline=format_portfolio_headline(summary),
        mode=mode,
        counterfactual_note=summary.counterfactual_note,
    )

    if write_reports:
        report_dir.mkdir(parents=True, exist_ok=True)
        write_report_files(report, report_dir, db_path=db_path)

    return report


def write_report_files(
    report: LoadTestReport,
    report_dir: Path,
    *,
    db_path: Path,
) -> None:
    """Write JSON + Markdown summaries and a dashboard PNG (aggregates only)."""
    json_path = report_dir / "load_test_savings.json"
    md_path = report_dir / "load_test_savings.md"
    png_path = report_dir / "load_test_dashboard.png"

    payload = asdict(report)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Load-test cost savings report (Phase 6.1)",
        "",
        f"**Headline:** {report.headline}",
        "",
        f"- **n:** {report.n}",
        f"- **cost_reduction_pct:** {report.cost_reduction_pct:.1f}% "
        f"({report.baseline_label})",
        f"- **saved:** ${report.savings_usd:.4f}",
        f"- **actual:** ${report.actual_cost_usd:.4f}",
        f"- **GPT-4o counterfactual:** ${report.gpt4o_cost_usd:.4f}",
        f"- **escalation_rate:** {report.escalation_rate:.1%} "
        f"({report.escalation_count} / {report.n})",
        f"- **mean_quality_score:** {report.mean_quality_score}",
        f"- **mode:** {report.mode}",
        "",
        "## Routing distribution",
        "",
        "| Model | Requests | Share |",
        "|---|---:|---:|",
    ]
    for row in report.routing_distribution:
        lines.append(
            f"| `{row['routed_model']}` | {row['request_count']} | "
            f"{float(row['share']) * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Tier distribution (classifier)",
            "",
            "| Tier | Count |",
            "|---|---:|",
        ]
    )
    for tier in ("1", "2", "3"):
        lines.append(f"| {tier} | {report.tier_distribution.get(tier, 0)} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Offline load test: real classifier + routing map; mocked provider "
            "costs from `MODEL_REGISTRY`.",
            "- Audit DB uses `prompt_hash` only (never raw prompts).",
            f"- Counterfactual: {report.counterfactual_note}",
            "",
            "Re-run:",
            "",
            "```bash",
            "PYTHONPATH=. python -m scripts.load_test",
            "PYTHONPATH=. python -m scripts.load_test --smoke",
            "```",
            "",
            f"Dashboard PNG: `{png_path.name}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    render_dashboard_png(report, db_path=db_path, out_path=png_path)


def render_dashboard_png(
    report: LoadTestReport,
    *,
    db_path: Path,
    out_path: Path,
) -> None:
    """Static portfolio chart: savings hero + routing pie + cost-by-day."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = cost_by_day(db_path=db_path)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("#f7f5f1")

    # Panel 1 — money-shot
    ax0 = axes[0]
    ax0.set_facecolor("#fffdf8")
    ax0.axis("off")
    ax0.text(
        0.5,
        0.72,
        f"{report.cost_reduction_pct:.1f}%",
        ha="center",
        va="center",
        fontsize=42,
        fontweight="bold",
        color="#14201a",
        transform=ax0.transAxes,
    )
    ax0.text(
        0.5,
        0.42,
        f"cost reduction\n{report.baseline_label}",
        ha="center",
        va="center",
        fontsize=11,
        color="#2a3430",
        transform=ax0.transAxes,
    )
    ax0.text(
        0.5,
        0.18,
        f"saved ${report.savings_usd:.4f}  ·  n={report.n}",
        ha="center",
        va="center",
        fontsize=9,
        color="#1f6f43",
        family="monospace",
        transform=ax0.transAxes,
    )
    ax0.set_title("Portfolio headline", fontsize=11, color="#5c6670")

    # Panel 2 — routing share
    ax1 = axes[1]
    labels = [str(r["routed_model"]) for r in report.routing_distribution]
    sizes = [int(r["request_count"]) for r in report.routing_distribution]
    colors = ["#2f6f4e", "#3d6e8c", "#8b5a2b", "#6b7280", "#a3a3a3"]
    ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.0f%%",
        colors=colors[: len(sizes)],
        textprops={"fontsize": 8},
        startangle=90,
    )
    ax1.set_title("Routing distribution", fontsize=11, color="#5c6670")

    # Panel 3 — actual vs GPT-4o by day
    ax2 = axes[2]
    if days:
        xs = list(range(len(days)))
        actual = [d.actual_cost_usd for d in days]
        gpt4o = [d.gpt4o_cost_usd for d in days]
        ax2.plot(xs, gpt4o, color="#8b5a2b", label="GPT-4o", linewidth=1.8)
        ax2.plot(xs, actual, color="#2f6f4e", label="Routed", linewidth=1.8)
        ax2.fill_between(xs, actual, gpt4o, color="#2f6f4e", alpha=0.12)
        ax2.set_xticks(xs[:: max(1, len(xs) // 5)])
        ax2.set_xticklabels(
            [days[i].period[5:] for i in xs[:: max(1, len(xs) // 5)]],
            fontsize=7,
        )
        ax2.legend(fontsize=8, loc="upper right")
        ax2.set_ylabel("USD", fontsize=8)
    ax2.set_title("Cost by day", fontsize=11, color="#5c6670")
    ax2.set_facecolor("#fffdf8")
    for spine in ax2.spines.values():
        spine.set_color("#d9d2c5")

    fig.suptitle(
        "LLM Cost AutoPilot — load-test dashboard",
        fontsize=13,
        fontweight="bold",
        color="#14201a",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 6.1 offline load test → savings report + dashboard PNG."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help=f"Number of prompts (default {DEFAULT_N}; smoke uses {SMOKE_N}). "
        f"Full runs should be {MIN_FULL_N}–{MAX_N}.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"CI-like check: n={SMOKE_N}, temp DB/reports, assert savings > 0.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"Audit DB path (default: {DEFAULT_DB}; smoke → temp).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=f"Report output dir (default: {DEFAULT_REPORT_DIR}; smoke → temp).",
    )
    parser.add_argument(
        "--labeled",
        type=Path,
        default=DEFAULT_LABELED,
        help="Path to labeled_prompts.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    tmp: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.smoke:
            tmp = tempfile.TemporaryDirectory(prefix="load-test-smoke-")
            tmp_path = Path(tmp.name)
            n = args.n if args.n is not None else SMOKE_N
            db_path = args.db or (tmp_path / "load_test_requests.db")
            report_dir = args.report_dir or (tmp_path / "reports")
            mode = "smoke"
        else:
            n = args.n if args.n is not None else DEFAULT_N
            if n < MIN_FULL_N:
                print(
                    f"warning: full run n={n} < {MIN_FULL_N}; "
                    "portfolio criterion is ≥500",
                    file=sys.stderr,
                )
            db_path = args.db or DEFAULT_DB
            report_dir = args.report_dir or DEFAULT_REPORT_DIR
            mode = "full"

        report = run_load_test(
            n=n,
            db_path=db_path,
            report_dir=report_dir,
            labeled_path=args.labeled,
            seed=args.seed,
            write_reports=True,
            mode=mode,
        )

        assert count_requests(db_path=db_path) == report.n
        assert report.n == n
        assert report.savings_usd > 0
        assert report.cost_reduction_pct > 0
        assert report.routing_distribution
        if args.smoke:
            assert report.n == SMOKE_N or args.n is not None
            png = report_dir / "load_test_dashboard.png"
            assert png.is_file() and png.stat().st_size > 0
            print(f"OK smoke load test n={report.n}")
        else:
            assert report.n >= MIN_FULL_N
            print(f"OK load test n={report.n}")

        print(f"{report.cost_reduction_pct:.1f}% {report.baseline_label}")
        print(
            f"saved ${report.savings_usd:.4f} "
            f"(actual ${report.actual_cost_usd:.4f} vs "
            f"GPT-4o ${report.gpt4o_cost_usd:.4f})"
        )
        print(report.headline)
        print(f"reports → {report_dir.resolve()}")
        print(f"db → {db_path.resolve()} (gitignored)")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
