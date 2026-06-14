"""
scheduler.py — your continuous batching scheduler.

The grader instantiates Scheduler, submits a stream of Requests, drives
ticks, and checks throughput / latency / utilization / fairness against
thresholds.
"""

from inference_stub import Request, forward, MAX_KV_BLOCKS, EOS_TOKEN, kv_blocks_for


class Scheduler:
    def __init__(self) -> None:
        raise NotImplementedError

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
