"""
scheduler.py — your continuous batching scheduler.

The grader instantiates Scheduler, submits a stream of Requests, drives
ticks, and checks throughput / latency / utilization / fairness against
thresholds.
"""

from inference_stub import Request, forward, MAX_KV_BLOCKS, EOS_TOKEN, kv_blocks_for
import time
from collections import defaultdict, deque


class Scheduler:
    def __init__(self) -> None:
        # Use available KV blocks as the practical batch size limit
        self.max_batch_size = MAX_KV_BLOCKS
        # Waiting requests are grouped by user to ensure fairness.
        self.waiting_by_user: dict[str, deque[Request]] = defaultdict(deque)
        self.user_order: deque[str] = deque()

        # Requests currently occupying GPU batch slots.
        self.active: list[Request] = []

        # Metrics
        self._tick = 0
        self._non_empty_ticks = 0
        self._generated_tokens = 0
        self._latencies: list[float] = []
        self._start_time = time.time()

    def submit(self, request: Request) -> None:
        """Accept a new request. Returns immediately."""
        raise NotImplementedError

    def tick(self) -> list[Request]:
        """Run one model iteration. Return list of requests that finished this tick."""
        raise NotImplementedError

    def has_active(self) -> bool:
        """True if any request is in-flight or waiting."""
        raise NotImplementedError

    def metrics(self) -> dict:
        """{ throughput_tokens_per_sec, avg_latency_s, p99_latency_s, gpu_utilization }"""
        raise NotImplementedError
