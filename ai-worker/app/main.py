"""Worker entrypoint.

Runs the queue consumer in the foreground. Designed for:
  - `python -m app.main`
  - a container CMD
  - a systemd unit

Kept deliberately FastAPI-free in phase 1: the worker has no inbound HTTP
surface of its own — core-api talks to it only via the Redis queue.
FastAPI remains in the dependency set so phase 2 can add a /health or
/internal/debug endpoint without a dependency churn.
"""

from __future__ import annotations

import logging
import signal
import sys

from app.capabilities.registry import build_default_registry
from app.clients.core_api_client import CoreApiClient
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.queue.redis_consumer import RedisQueueConsumer
from app.services.task_runner import TaskRunner
from app.storage.resolver import StorageResolver

log = logging.getLogger(__name__)


def run() -> int:
    configure_logging()
    settings = get_settings()

    log.info("Starting worker id=%s core_api=%s", settings.worker_id, settings.core_api_base_url)

    core_api = CoreApiClient(
        base_url=settings.core_api_base_url,
        timeout_seconds=settings.core_api_request_timeout_seconds,
    )
    registry = build_default_registry(settings)
    resolver = StorageResolver(local_root=settings.local_storage_root)
    runner = TaskRunner(
        core_api=core_api,
        registry=registry,
        resolver=resolver,
        worker_id=settings.worker_id,
    )
    consumer = RedisQueueConsumer(
        redis_url=settings.redis_url,
        pending_key=settings.queue_pending_key,
        block_timeout_seconds=settings.queue_block_timeout_seconds,
    )

    try:
        consumer.ping()
    except Exception:
        log.exception("Redis ping failed — aborting startup")
        core_api.close()
        return 2

    def _stop(signum, frame):  # noqa: ANN001
        log.info("Received signal %s — stopping consumer", signum)
        consumer.stop()

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError):
        # SIGTERM not available on Windows; SIGINT is enough for local dev
        pass

    try:
        consumer.consume_forever(runner.handle)
    finally:
        core_api.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
