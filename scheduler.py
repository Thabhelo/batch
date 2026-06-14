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

    def _admit_waiting_requests(self) -> None:
        """Move waiting requests to active batch with fair round-robin scheduling."""
        while len(self.active) < self.max_batch_size and self.user_order:
            user_id = self.user_order.popleft()

            if user_id in self.waiting_by_user and self.waiting_by_user[user_id]:
                request = self.waiting_by_user[user_id].popleft()
                self.active.append(request)

                # If this user still has waiting requests, put them back in queue
                if self.waiting_by_user[user_id]:
                    self.user_order.append(user_id)

    def tick(self) -> list[Request]:
        """Run one model iteration. Return list of requests that finished this tick."""
        self._admit_waiting_requests()

        if not self.active:
            self._tick += 1
            return [] # No active requests, so no completions

        self._tick += 1
        self._non_empty_ticks += 1

        decode_batch: list[Request] = []

        for request in self.active:
            remaining_prefill = self._prefill_remaining.get(request.request_id, 0)

            if remaining_prefill > 0:
                self._prefill_remaining[request.request_id] = remaining_prefill - 1
                decode_batch.append(request)
            else:
                request.started_at = time.time()
                decode_batch.append(request)

        if not decode_batch:
            return [] # No requests to decode this tick

        # Decode the batch
        if decode_batch:
            tokens = forward(decode_batch)
            for request, token in zip(decode_batch, tokens):
                if request.started_at is None:
                    request.started_at = time.time()
                request.generated.append(token)
                self._generated_tokens += 1

        finished: list[Request] = []
        still_active: list[Request] = []

        for request in self.active:
            reached_max_tokens = len(request.generated) >= request.max_tokens
            saw_eos = bool(request.generated and request.generated[-1] == EOS_TOKEN)

            if reached_max_tokens or saw_eos:
                request.completed_at = time.time()
                finished.append(request)

                if request.submitted_at is not None:
                    self._latencies.append(time.time() - request.submitted_at)

                    self._prefill_remaining.pop(request.request_id, None)
            else:
                still_active.append(request)

        self.active = still_active
        return finished
                        
                        

    def has_active(self) -> bool:
        """True if any request is in-flight or waiting."""
        if not self.active:
            return False
        return True

    def metrics(self) -> dict:
        """{ throughput_tokens_per_sec, avg_latency_s, p99_latency_s, gpu_utilization }"""
        
        if not self._latencies:
            return {
                "throughput_tokens_per_sec": 0.0,
                "avg_latency_s": 0.0,
                "p99_latency_s": 0.0,
                "gpu_utilization": 0.0,
            }

        return {
            "throughput_tokens_per_sec": self._generated_tokens / (time.time() - self._start_time),
            "avg_latency_s": sum(self._latencies) / len(self._latencies),
            "p99_latency_s": sorted(self._latencies)[int(0.99 * len(self._latencies))],
            "gpu_utilization": self._non_empty_ticks / max(1, self._tick),
        }