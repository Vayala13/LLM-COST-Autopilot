"""Phase 4.1 — SQLite audit trail for every routed completion.

One row per request. Stores ``prompt_hash`` only (never raw prompt text
or API keys). All writes use parameterized SQL.

Call site for the future FastAPI path (and smokes today)::

    from app.audit import log_completion
    from app.quality.verifier import prompt_hash

    row_id = log_completion(
        prompt_hash=prompt_hash(prompt),
        complexity_tier=tier,
        routed_model=model_key,
        cost=response.cost_usd,
        latency=response.latency_s,
        verifier_quality_score=verification.score,
        escalated=escalation.escalated,
        escalation_model=escalation.escalated_model,
        cost_delta=escalation.cost_delta_usd,
        use_case=use_case,
    )
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _ROOT / "data" / "requests.db"

# Schema: one audit row per request. Optional escalation columns stay NULL
# when no escalation ran. verifier_quality_score may be NULL if verify
# has not completed yet (async path).
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    complexity_tier INTEGER NOT NULL,
    routed_model TEXT NOT NULL,
    cost REAL NOT NULL,
    latency REAL NOT NULL,
    verifier_quality_score REAL,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalation_model TEXT,
    cost_delta REAL,
    use_case TEXT
);
"""

_INSERT_SQL = """
INSERT INTO requests (
    timestamp,
    prompt_hash,
    complexity_tier,
    routed_model,
    cost,
    latency,
    verifier_quality_score,
    escalated,
    escalation_model,
    cost_delta,
    use_case
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_RECENT_SQL = """
SELECT
    id,
    timestamp,
    prompt_hash,
    complexity_tier,
    routed_model,
    cost,
    latency,
    verifier_quality_score,
    escalated,
    escalation_model,
    cost_delta,
    use_case
FROM requests
ORDER BY id DESC
LIMIT ?
"""


@dataclass(frozen=True)
class RequestLogRow:
    """One audit-trail row (readback / smoke)."""

    id: int
    timestamp: str
    prompt_hash: str
    complexity_tier: int
    routed_model: str
    cost: float
    latency: float
    verifier_quality_score: float | None
    escalated: bool
    escalation_model: str | None
    cost_delta: float | None
    use_case: str | None


def _resolve_db_path(db_path: str | Path | None) -> Path:
    """Resolve DB path; default is ``data/requests.db`` under the project root."""
    if db_path is None:
        return DEFAULT_DB_PATH
    return Path(db_path)


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection; create parent dirs. Caller must close/commit."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path | None = None) -> Path:
    """Create the ``requests`` table if missing. Returns the DB path used."""
    path = _resolve_db_path(db_path)
    with _connect(path) as conn:
        conn.execute(_SCHEMA_SQL)
        conn.commit()
    return path


def log_request(
    *,
    prompt_hash: str,
    complexity_tier: int,
    routed_model: str,
    cost: float,
    latency: float,
    verifier_quality_score: float | None = None,
    escalated: bool = False,
    escalation_model: str | None = None,
    cost_delta: float | None = None,
    use_case: str | None = None,
    timestamp: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Insert one audit row. Returns the new row ``id``.

    Security: never pass raw prompt text or secrets here — only a
    ``prompt_hash`` (see ``app.quality.verifier.prompt_hash``).
    """
    if not prompt_hash or not isinstance(prompt_hash, str):
        raise ValueError("prompt_hash must be a non-empty string")
    if not isinstance(complexity_tier, int) or complexity_tier < 1 or complexity_tier > 3:
        raise ValueError("complexity_tier must be an int in {1, 2, 3}")
    if not routed_model or not isinstance(routed_model, str):
        raise ValueError("routed_model must be a non-empty string")
    if not isinstance(cost, (int, float)) or cost < 0:
        raise ValueError("cost must be a non-negative number")
    if not isinstance(latency, (int, float)) or latency < 0:
        raise ValueError("latency must be a non-negative number")

    ts = timestamp or datetime.now(timezone.utc).isoformat()
    path = init_db(db_path)
    escalated_flag = 1 if escalated else 0

    with _connect(path) as conn:
        cur = conn.execute(
            _INSERT_SQL,
            (
                ts,
                prompt_hash,
                complexity_tier,
                routed_model,
                float(cost),
                float(latency),
                None if verifier_quality_score is None else float(verifier_quality_score),
                escalated_flag,
                escalation_model,
                None if cost_delta is None else float(cost_delta),
                use_case,
            ),
        )
        conn.commit()
        row_id = int(cur.lastrowid)
    return row_id


def log_completion(
    *,
    prompt_hash: str,
    complexity_tier: int,
    routed_model: str,
    cost: float,
    latency: float,
    verifier_quality_score: float | None = None,
    escalated: bool = False,
    escalation_model: str | None = None,
    cost_delta: float | None = None,
    use_case: str | None = None,
    timestamp: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Call-site helper after a routed completion (API / smoke / scripts).

    Thin alias over ``log_request`` with the Phase 4.1 field set. Future
    ``POST /v1/completions`` should call this once per request after
    routing (and optionally after verify/escalate fill score + flag).
    """
    return log_request(
        prompt_hash=prompt_hash,
        complexity_tier=complexity_tier,
        routed_model=routed_model,
        cost=cost,
        latency=latency,
        verifier_quality_score=verifier_quality_score,
        escalated=escalated,
        escalation_model=escalation_model,
        cost_delta=cost_delta,
        use_case=use_case,
        timestamp=timestamp,
        db_path=db_path,
    )


def fetch_requests(
    *,
    limit: int = 50,
    db_path: str | Path | None = None,
) -> list[RequestLogRow]:
    """Read recent audit rows (newest first). Used by smoke / future dashboard."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive int")
    path = init_db(db_path)
    rows: list[RequestLogRow] = []
    with _connect(path) as conn:
        # Parameterized LIMIT — never interpolate into SQL text.
        for r in conn.execute(_SELECT_RECENT_SQL, (limit,)):
            rows.append(
                RequestLogRow(
                    id=int(r["id"]),
                    timestamp=str(r["timestamp"]),
                    prompt_hash=str(r["prompt_hash"]),
                    complexity_tier=int(r["complexity_tier"]),
                    routed_model=str(r["routed_model"]),
                    cost=float(r["cost"]),
                    latency=float(r["latency"]),
                    verifier_quality_score=(
                        None
                        if r["verifier_quality_score"] is None
                        else float(r["verifier_quality_score"])
                    ),
                    escalated=bool(r["escalated"]),
                    escalation_model=r["escalation_model"],
                    cost_delta=(
                        None if r["cost_delta"] is None else float(r["cost_delta"])
                    ),
                    use_case=r["use_case"],
                )
            )
    return rows


def row_to_dict(row: RequestLogRow) -> dict[str, Any]:
    """Serialize a row for JSON-friendly smoke output."""
    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "prompt_hash": row.prompt_hash,
        "complexity_tier": row.complexity_tier,
        "routed_model": row.routed_model,
        "cost": row.cost,
        "latency": row.latency,
        "verifier_quality_score": row.verifier_quality_score,
        "escalated": row.escalated,
        "escalation_model": row.escalation_model,
        "cost_delta": row.cost_delta,
        "use_case": row.use_case,
    }
