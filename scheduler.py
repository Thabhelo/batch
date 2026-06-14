"""
scheduler.py — your continuous batching scheduler.

The grader instantiates Scheduler, submits a stream of Requests, drives
ticks, and checks throughput / latency / utilization / fairness against
thresholds.
"""

from inference_stub import Request, forward, MAX_KV_BLOCKS, EOS_TOKEN, kv_blocks_for
import time
from collections import defaultdict, deque

# Chunk size for prefill processing (tokens per tick)
CHUNK_SIZE = 64


class Scheduler:
    def __init__(self) -> None:
        # Use available KV blocks as the practical batch size limit
        self.max_batch_size = MAX_KV_BLOCKS
        # Waiting requests are grouped by user to ensure fairness.
        self.waiting_by_user: dict[str, deque[Request]] = defaultdict(deque)
        self.user_order: deque[str] = deque()

        # Requests currently occupying GPU batch slots.
        self.active: list[Request] = []

        # Track prefill progress for chunked processing
        self._prefill_remaining: dict[str, int] = {}

        # Metrics
        self._tick = 0
        self._non_empty_ticks = 0
        self._generated_tokens = 0
        self._latencies: list[float] = []
        self._start_time = time.time()

    def submit(self, request: Request) -> None:
        """Accept a new request. Returns immediately."""
        now = time.time()

        if not request.submitted_at:
            request.submitted_at = now

        user_id = request.user_id
        was_empty = len(self.waiting_by_user[user_id]) == 0

        self.waiting_by_user[user_id].append(request)

        if was_empty and user_id not in self.user_order:
            self.user_order.append(user_id)
        
        # Short prompts should not be delayed
        # Prompts over 64 tokens get chunked across multiple ticks
        self._prefill_remaining[request.request_id] = max(0, (len(request.prompt) - 1) // CHUNK_SIZE)
    def tick(self) -> list[Request]:
        """Run one model iteration. Return list of requests that finished this tick."""
        raise NotImplementedError

    def has_active(self) -> bool:
        """True if any request is in-flight or waiting."""
        raise NotImplementedError

    def metrics(self) -> dict:
        """{ throughput_tokens_per_sec, avg_latency_s, p99_latency_s, gpu_utilization }"""
        raise NotImplementedError
