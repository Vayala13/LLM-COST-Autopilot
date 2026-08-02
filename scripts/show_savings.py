"""Phase 4.3 — print the portfolio money-shot cost-reduction %.

Reads aggregates from ``data/requests.db`` (or ``--db``). With ``--demo``,
seeds synthetic rows when the DB is empty so the headline works offline.

Never prints raw prompts — cost aggregates only.

Run:
    PYTHONPATH=. python -m scripts.show_savings
    PYTHONPATH=. python -m scripts.show_savings --demo
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from app.audit.store import DEFAULT_DB_PATH
from app.metrics.cost import (
    PORTFOLIO_BASELINE_LABEL,
    format_portfolio_headline,
    load_portfolio_headline,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print portfolio cost-reduction % (vs all GPT-4o)."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help=f"Audit DB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="If DB empty, seed demo rows (or use a temp DB when --db omitted).",
    )
    args = parser.parse_args(argv)

    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    db_path = args.db
    seed_if_empty = bool(args.demo)

    if args.demo and db_path is None:
        # Isolated temp DB so --demo never mutates a real empty requests.db
        # unless the caller explicitly passed --db.
        tmp_dir = tempfile.TemporaryDirectory(prefix="show-savings-")
        db_path = Path(tmp_dir.name) / "requests.db"
        seed_if_empty = True
    elif db_path is None:
        db_path = DEFAULT_DB_PATH

    try:
        summary = load_portfolio_headline(
            db_path=db_path, seed_if_empty=seed_if_empty
        )
        if summary.request_count == 0:
            print(
                "No audit rows. Re-run with --demo, or load demo data in the "
                "dashboard sidebar, then retry.",
                file=sys.stderr,
            )
            return 1

        # Money-shot first line — easy to copy into portfolio case study.
        print(f"{summary.cost_reduction_pct:.1f}%")
        print(f"cost reduction {PORTFOLIO_BASELINE_LABEL}")
        print(
            f"saved ${summary.savings_usd:,.4f} "
            f"(actual ${summary.actual_cost_usd:,.4f} vs "
            f"GPT-4o ${summary.gpt4o_cost_usd:,.4f}; "
            f"n={summary.request_count})"
        )
        print(format_portfolio_headline(summary))
        return 0
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()


if __name__ == "__main__":
    sys.exit(main())
