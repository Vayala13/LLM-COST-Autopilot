"""Phase 5.3 background worker package.

See ``app.worker.main`` — watches shared ``data/`` volume; does not drain the
API in-process asyncio verification queue.
"""
