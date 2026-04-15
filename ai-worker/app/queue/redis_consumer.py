"""Redis queue consumer.

Thin wrapper around `redis.Redis` that does a blocking BRPOP loop. The
consumer is intentionally simple: one key, one worker process, one message
at a time. That's enough for phase 1 — it matches the core-api's
single-pending-list dispatch and keeps the claim/callback path as the only
point of coordination between workers.

Scale-out story for later: multiple worker processes BRPOPping the same
key share load automatically, and the atomic claim on core-api prevents
any two workers from running the same job.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

import redis

from app.queue.messages import QueueMessage

log = logging.getLogger(__name__)


class RedisQueueConsumer:
    def __init__(
        self,
        redis_url: str,
        pending_key: str,
        block_timeout_seconds: int,
    ) -> None:
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._pending_key = pending_key
        self._block_timeout = block_timeout_seconds
        self._stopped = False

    def ping(self) -> None:
        """Fail fast if Redis is unreachable at startup."""
        self._client.ping()

    def stop(self) -> None:
        self._stopped = True

    def consume_forever(self, handler: Callable[[QueueMessage], None]) -> None:
        """Block on BRPOP and invoke the handler for each message.

        The handler is expected to own all retry / error semantics for an
        individual job. If it raises, we log and keep going — the job will
        get retried via the claim lease expiry path in a later phase.
        """
        log.info(
            "Queue consumer started key=%s block_timeout=%ds",
            self._pending_key,
            self._block_timeout,
        )
        while not self._stopped:
            raw: Optional[tuple[str, str]] = self._client.brpop(
                self._pending_key, timeout=self._block_timeout
            )
            if raw is None:
                continue
            _, payload = raw
            try:
                data = json.loads(payload)
                message = QueueMessage.model_validate(data)
            except Exception:
                log.exception("Dropping malformed queue message: %s", payload)
                continue
            try:
                handler(message)
            except Exception:
                log.exception("Handler raised for job %s", message.job_id)
        log.info("Queue consumer stopped")
