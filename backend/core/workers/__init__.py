"""
core.workers
============
Reusable, infrastructure-level async worker pools for fanning out
independent units of work across a configurable number of concurrent
workers.

This package deliberately lives in ``core`` (not inside any single
agent) because a bounded worker pool is cross-cutting infrastructure:
the email agent's cluster-synthesis pipeline is the first consumer, but
any agent that needs "process N independent items with at most K in
flight at a time" can reuse it.

Design goals / invariants
--------------------------
* **Queue-based, not threadpool-based.** Work is dispatched through an
  ``asyncio.Queue`` drained by a fixed set of worker *coroutines* (plain
  ``asyncio.Task``s). There are NO ``ThreadPoolExecutor``s and no manual
  ``threading.Thread`` worker pools here, so this pool never stacks a
  second thread pool on top of the ones the blocking clients already use
  internally.
* **Cooperates with the existing semaphores.** Outbound Ollama / Graph
  calls are already clamped by ``core.concurrency`` semaphores. This pool
  only controls how many *items* are processed concurrently; the real
  ceiling on outbound model calls stays whatever
  ``MAX_CONCURRENT_OLLAMA_CALLS`` is. So raising the worker count can
  never exploit a locally-hosted service beyond that cap — it just keeps
  the semaphore saturated instead of idle between sequential items.
* **One failing item never sinks the batch.** Each item's result (or the
  exception it raised) is captured per-item; the caller decides what to
  do with failures.
"""

from core.workers.pool import WorkerResult, run_in_worker_pool

__all__ = ["WorkerResult", "run_in_worker_pool"]
