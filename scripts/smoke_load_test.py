"""Phase 6.1 — smoke wrapper for the offline load test (n=50).

Run:
    PYTHONPATH=. python -m scripts.smoke_load_test
"""

from __future__ import annotations

import sys

from scripts.load_test import main as load_test_main


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else []
    if "--smoke" not in args:
        args = ["--smoke", *args]
    return load_test_main(args)


if __name__ == "__main__":
    sys.exit(main())
