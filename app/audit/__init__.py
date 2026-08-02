"""Request audit trail (Phase 4.1) — SQLite logging of routed completions."""

from app.audit.store import (
    DEFAULT_DB_PATH,
    fetch_requests,
    init_db,
    log_completion,
    log_request,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "fetch_requests",
    "init_db",
    "log_completion",
    "log_request",
]
