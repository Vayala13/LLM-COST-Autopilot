"""Phase 4.3 — LLM Cost AutoPilot Streamlit dashboard (money-shot hero).

Run (localhost only by default — do not expose publicly without auth)::

    streamlit run dashboard/app.py --server.address=127.0.0.1 --server.port=8501

Charts use aggregates from ``data/requests.db`` only (prompt hashes / metrics —
never raw prompts). If the DB is empty, use the sidebar button to load
synthetic demo rows for portfolio screenshots.

Portfolio headline number (CLI)::

    PYTHONPATH=. python -m scripts.show_savings --demo
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run dashboard/app.py` from repo root without PYTHONPATH.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app.audit.store import DEFAULT_DB_PATH, count_requests, init_db
from app.metrics.cost import (
    GPT4O_COUNTERFACTUAL_NOTE,
    PORTFOLIO_BASELINE_LABEL,
    compute_summary,
    cost_by_day,
    cost_by_week,
    escalation_rate_by_day,
    quality_score_distribution,
    routing_distribution,
    seed_demo_requests,
)

st.set_page_config(
    page_title="LLM Cost AutoPilot",
    page_icon=None,
    layout="wide",
)

# Hero typography — large cost-reduction % is the portfolio money-shot.
_HERO_CSS = """
<style>
div[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 420px at 12% -8%, rgba(34, 120, 90, 0.10), transparent 55%),
    radial-gradient(700px 380px at 88% 0%, rgba(30, 58, 95, 0.12), transparent 50%),
    #f7f5f1;
}
.money-shot {
  margin: 0.25rem 0 1.25rem 0;
  padding: 1.75rem 1.5rem 1.5rem;
  border: 1px solid #d9d2c5;
  background: linear-gradient(180deg, #fffdf8 0%, #f3efe6 100%);
  text-align: center;
}
.money-shot .eyebrow {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #5c6670;
  margin: 0 0 0.35rem 0;
}
.money-shot .pct {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: clamp(4.2rem, 10vw, 6.5rem);
  font-weight: 700;
  line-height: 0.95;
  letter-spacing: -0.03em;
  color: #14201a;
  margin: 0;
}
.money-shot .baseline {
  font-size: 1.15rem;
  color: #2a3430;
  margin: 0.55rem 0 0.15rem 0;
}
.money-shot .saved {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 1.05rem;
  color: #1f6f43;
  margin: 0.35rem 0 0 0;
}
.money-shot .sub {
  font-size: 0.85rem;
  color: #6b7280;
  margin: 0.55rem 0 0 0;
}
</style>
"""


def _fmt_usd(value: float) -> str:
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:,.4f}"


def _render_money_shot(summary) -> None:
    """Dominant hero: cost_reduction_pct + $ saved + GPT-4o baseline."""
    pct = summary.cost_reduction_pct
    st.markdown(
        f"""
<div class="money-shot">
  <p class="eyebrow">Portfolio headline</p>
  <p class="pct">{pct:.1f}%</p>
  <p class="baseline">cost reduction {PORTFOLIO_BASELINE_LABEL}</p>
  <p class="saved">You saved {_fmt_usd(summary.savings_usd)}</p>
  <p class="sub">
    actual {_fmt_usd(summary.actual_cost_usd)} ·
    if all GPT-4o {_fmt_usd(summary.gpt4o_cost_usd)} ·
    {summary.request_count} requests
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(_HERO_CSS, unsafe_allow_html=True)
    st.title("LLM Cost AutoPilot")
    st.caption(
        "Phase 4.3 — money-shot cost reduction % vs all GPT-4o, "
        "plus routing / quality / escalation from the audit trail."
    )

    db_path = DEFAULT_DB_PATH
    init_db(db_path)

    with st.sidebar:
        st.header("Data")
        st.write(f"DB: `{db_path}`")
        n = count_requests(db_path=db_path)
        st.metric("Audit rows", n)
        if n == 0:
            st.info("DB is empty. Load demo rows for screenshots.")
            if st.button("Load demo data", type="primary"):
                inserted = seed_demo_requests(db_path=db_path)
                st.success(f"Inserted {inserted} synthetic rows.")
                st.rerun()
        else:
            st.caption("Demo seed only runs when the DB is empty.")
        st.divider()
        st.caption(
            "Bind to localhost: "
            "`streamlit run dashboard/app.py --server.address=127.0.0.1`"
        )
        st.caption("Aggregates only — no raw prompts.")
        st.caption("Headline CLI: `python -m scripts.show_savings --demo`")

    summary = compute_summary(db_path=db_path)
    if summary.request_count == 0:
        st.warning("No audit rows yet. Use **Load demo data** in the sidebar.")
        return

    # --- Money-shot hero (Phase 4.3) ----------------------------------------
    _render_money_shot(summary)
    st.caption(GPT4O_COUNTERFACTUAL_NOTE)
    if summary.tokens_fallback_count:
        st.caption(
            f"Token fallback used on {summary.tokens_fallback_count} of "
            f"{summary.request_count} rows (missing input/output tokens)."
        )

    # Secondary metrics under the hero
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual cost", _fmt_usd(summary.actual_cost_usd))
    c2.metric("If all GPT-4o", _fmt_usd(summary.gpt4o_cost_usd))
    c3.metric("Dollars saved", _fmt_usd(summary.savings_usd))
    c4.metric(
        "Escalation rate",
        f"{summary.escalation_rate * 100:.1f}%",
        help=f"{summary.escalation_count} / {summary.request_count} requests",
    )

    # --- Cost by day / week -------------------------------------------------
    st.subheader("Cost over time")
    grain = st.radio("Period", ("Day", "Week"), horizontal=True)
    period_rows = cost_by_day(db_path=db_path) if grain == "Day" else cost_by_week(
        db_path=db_path
    )
    if period_rows:
        cost_df = pd.DataFrame(
            {
                "period": [r.period for r in period_rows],
                "actual": [r.actual_cost_usd for r in period_rows],
                "gpt-4o": [r.gpt4o_cost_usd for r in period_rows],
            }
        ).set_index("period")
        st.bar_chart(cost_df, stack=False)
    else:
        st.write("No cost series.")

    left, right = st.columns(2)

    # --- Routing pie --------------------------------------------------------
    with left:
        st.subheader("Routing distribution")
        shares = routing_distribution(db_path=db_path)
        if shares:
            pie_df = pd.DataFrame(
                {
                    "model": [s.routed_model for s in shares],
                    "requests": [s.request_count for s in shares],
                }
            )
            st.dataframe(
                pie_df.assign(share=[f"{s.share * 100:.1f}%" for s in shares]),
                hide_index=True,
                use_container_width=True,
            )
            # Streamlit has no native pie; Altair is a Streamlit dependency.
            import altair as alt

            chart = (
                alt.Chart(pie_df)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("requests:Q"),
                    color=alt.Color("model:N", legend=alt.Legend(title="Model")),
                    tooltip=["model", "requests"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No routing data.")

    # --- Quality distribution -----------------------------------------------
    with right:
        st.subheader("Quality score distribution")
        if summary.mean_quality_score is not None:
            st.caption(
                f"Mean score {summary.mean_quality_score:.2f} "
                f"across {summary.scored_count} scored requests "
                "(unscored / pending verify excluded)."
            )
        buckets = quality_score_distribution(db_path=db_path)
        if buckets:
            q_df = pd.DataFrame(
                {"score": [b.label for b in buckets], "count": [b.count for b in buckets]}
            ).set_index("score")
            st.bar_chart(q_df)
        else:
            st.write("No quality scores yet.")

    # --- Escalation rate over time ------------------------------------------
    st.subheader("Escalation rate over time")
    esc_rows = escalation_rate_by_day(db_path=db_path)
    if esc_rows:
        esc_df = pd.DataFrame(
            {
                "day": [r.period for r in esc_rows],
                "escalation_rate": [r.escalation_rate for r in esc_rows],
            }
        ).set_index("day")
        st.line_chart(esc_df)
    else:
        st.write("No escalation series.")


if __name__ == "__main__":
    main()
