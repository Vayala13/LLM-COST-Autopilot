"""Phase 4.2/4.3 — cost / quality metrics over the SQLite audit trail."""

from app.metrics.cost import (
    GPT4O_COUNTERFACTUAL_NOTE,
    PORTFOLIO_BASELINE_LABEL,
    CostPeriodRow,
    DashboardSummary,
    EscalationPeriodRow,
    QualityBucket,
    RoutingShare,
    compute_summary,
    cost_by_day,
    cost_by_week,
    escalation_rate_by_day,
    format_portfolio_headline,
    load_portfolio_headline,
    quality_score_distribution,
    routing_distribution,
    seed_demo_requests,
)

__all__ = [
    "GPT4O_COUNTERFACTUAL_NOTE",
    "PORTFOLIO_BASELINE_LABEL",
    "CostPeriodRow",
    "DashboardSummary",
    "EscalationPeriodRow",
    "QualityBucket",
    "RoutingShare",
    "compute_summary",
    "cost_by_day",
    "cost_by_week",
    "escalation_rate_by_day",
    "format_portfolio_headline",
    "load_portfolio_headline",
    "quality_score_distribution",
    "routing_distribution",
    "seed_demo_requests",
]
