"""
Visible tests for the scheduler. The grader runs a larger suite including
fairness, KV cap pressure, and high-load throughput.
"""

import pytest

try:
    from scheduler import Scheduler
    from inference_stub import Request, EOS_TOKEN
    from request_gen import stream
except ImportError:
    pytest.skip("scheduler not implemented yet", allow_module_level=True)


def test_single_request_completes():
    s = Scheduler()
    s.submit(Request(user_id="u1", prompt=[1, 2, 3], max_tokens=5, request_id="r1"))
    finished_total = 0
    for _ in range(200):
        finished_total += len(s.tick())
        if not s.has_active():
            break
    assert finished_total == 1


def test_multiple_concurrent():
    s = Scheduler()
    for i in range(5):
        s.submit(Request(user_id=f"u{i}", prompt=[1, 2, 3], max_tokens=10, request_id=f"r{i}"))
    finished_count = 0
    for _ in range(500):
        finished_count += len(s.tick())
        if not s.has_active():
            break
    assert finished_count == 5


def test_metrics_shape():
    s = Scheduler()
    s.submit(Request(user_id="u1", prompt=[1, 2, 3], max_tokens=5, request_id="r1"))
    while s.has_active():
        s.tick()
    m = s.metrics()
    for key in ("throughput_tokens_per_sec", "avg_latency_s", "p99_latency_s", "gpu_utilization"):
        assert key in m, f"missing metric: {key}"


def test_stream_completes():
    s = Scheduler()
    for r in stream(n=20, lam=10.0, seed=1):
        s.submit(r)
    finished_count = 0
    for _ in range(5000):
        finished_count += len(s.tick())
        if not s.has_active():
            break
    assert finished_count == 20
