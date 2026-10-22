"""
core/workers/pool.py
====================
A bounded, queue-based async worker pool.

``run_in_worker_pool`` takes a list of items and an async ``handler``,
puts every item on an ``asyncio.Queue``, and spins up ``workers``
concurrent worker coroutines that drain the queue until it is empty.
Results are returned in the SAME order as the input items so callers can
zip them back against their inputs without tracking indices by hand.

Why a queue (and not just ``asyncio.gather`` over every item)?
--------------------------------------------------------------
``gather`` would schedule *all* items at once and lean entirely on the
downstream semaphore to throttle. A fixed pool of ``workers`` draining a
queue gives an explicit, independently-tunable cap on how many items are
*in flight* (each item may fan out into several downstream calls), keeps
memory/log noise bounded, and makes the degree of parallelism a single
obvious ``.env`` knob. It is also the shape that maps cleanly onto an
external broker later (Celery/RQ) without changing call sites.

No threads are created here: workers are plain ``asyncio.Task``s, so this
never nests a thread pool on top of the blocking clients the handlers
call into. Any genuinely blocking work inside a handler should still be
pushed off the loop with ``asyncio.to_thread`` by the handler itself.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

from core.logging_config import get_agent_logger

logger = get_agent_logger("workers")

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class WorkerResult(Generic[R]):
    """
    Outcome of processing a single item.

    Exactly one of ``value`` / ``error`` is meaningful: ``ok`` is True
    when the handler returned normally (``value`` holds its result), and
    False when it raised (``error`` holds the exception). This lets a
    batch keep going when one item fails instead of the whole run
    aborting on the first exception.
    """

    index: int
    ok: bool
    value: R | None = None
    error: BaseException | None = None


def _normalize_worker_count(workers: int, item_count: int) -> int:
    """
    Clamp the requested worker count to a sane range: at least 1, never
    more than the number of items (extra workers would just idle). A
    non-positive configured value falls back to 1 (sequential), which
    preserves the previous behaviour when someone sets the knob to 0.
    """
    if workers < 1:
        workers = 1
    if item_count and workers > item_count:
        workers = item_count
    return workers


async def run_in_worker_pool(
    items: list[T],
    handler: Callable[[int, T], Awaitable[R]],
    *,
    workers: int,
    label: str = "work",
) -> list[WorkerResult[R]]:
    """
    Process ``items`` concurrently with a fixed pool of ``workers``
    queue-draining worker coroutines.

    Args:
        items: the units of work. May be empty (returns ``[]``).
        handler: ``async def handler(index, item) -> result``. ``index``
            is the item's position in ``items`` (handy for correlated
            logging). The handler owns its own retry/error semantics; any
            exception it raises is captured per-item, not propagated.
        workers: desired concurrency. Clamped to ``[1, len(items)]``; a
            value < 1 is treated as 1 (fully sequential).
        label: short noun used in log lines (e.g. "cluster").

    Returns:
        A list of :class:`WorkerResult`, one per input item, in the SAME
        order as ``items``.
    """
    total = len(items)
    if total == 0:
        return []

    worker_count = _normalize_worker_count(workers, total)

    # Pre-size the results list so each worker can write its slot by index
    # without any shared-list append race — ordering is preserved for free.
    results: list[WorkerResult[R]] = [
        WorkerResult(index=i, ok=False) for i in range(total)
    ]

    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(total):
        queue.put_nowait(i)

    logger.info(
        "[WORKER-POOL] starting label=%s items=%d workers=%d",
        label, total, worker_count,
    )

    async def _worker(worker_id: int) -> None:
        while True:
            try:
                index = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                value = await handler(index, items[index])
                results[index] = WorkerResult(index=index, ok=True, value=value)
            except asyncio.CancelledError:
                # Propagate cancellation so the whole pool can be torn
                # down promptly (e.g. on shutdown); do not swallow it.
                raise
            except BaseException as exc:  # noqa: BLE001 - captured per item
                logger.error(
                    "[WORKER-POOL] worker=%d label=%s item=%d failed: %s",
                    worker_id, label, index, exc,
                )
                results[index] = WorkerResult(index=index, ok=False, error=exc)
            finally:
                queue.task_done()

    tasks = [asyncio.create_task(_worker(w)) for w in range(worker_count)]
    try:
        await asyncio.gather(*tasks)
    finally:
        # Defensive: if gather was interrupted (cancellation), make sure
        # no worker task is left dangling.
        for task in tasks:
            if not task.done():
                task.cancel()

    failed = sum(1 for r in results if not r.ok)
    logger.info(
        "[WORKER-POOL] finished label=%s items=%d ok=%d failed=%d workers=%d",
        label, total, total - failed, failed, worker_count,
    )
    return results
