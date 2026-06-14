# Batch

*A continuous-batching scheduler for shared-GPU LLM inference.*

## Overview

Static batching is what naive inference servers do: collect N requests, run them as a batch through the model, return all responses, repeat. It looks efficient but it isn't. A long request blocks short requests sitting next to it in the batch, GPU utilization drops to 30% for most of the run, and high-percentile latency explodes. The fix is continuous batching (also called iteration-level scheduling, or in-flight batching). Instead of treating a batch as atomic, the scheduler decides at every model iteration which requests participate in the next forward pass. Short requests finish and leave; new requests join. The GPU stays saturated; tail latency stays bounded.

This is the technique behind the most-used open-source LLM inference servers, and the reason every serious inference startup eventually builds their own. You're going to build the scheduler.

## Problem Statement

Build a continuous-batching scheduler for a shared GPU running an LLM. Incoming requests stream in over time with variable prompt lengths and variable output lengths. The scheduler decides at each iteration which requests are in the current batch. It must keep the GPU saturated under load, complete each request correctly, respect a strict KV memory cap, and serve requests fairly across users so no single user can starve others.

## Getting Started

### Prerequisites
- Python 3.11+

### Setup
Dependencies are installed automatically when you initialize the assessment with the Litmus CLI. You're ready to start coding.

Files in the workspace:
- `inference_stub.py` provides the stubbed GPU. `forward(batch)` runs one iteration and returns one new token per request in the batch. Per-iteration latency scales with batch size. KV memory is simulated: total active KV across the batch is capped at `MAX_KV_BLOCKS`.
- `request_gen.py` produces a Poisson-arrival stream of requests with mixed prompt and max-token lengths across multiple users.
- `scheduler.py` is your entrypoint. Implement the `Scheduler` class.
- `tests/test_scheduler.py` covers the basic contract. The grader runs a larger suite that exercises fairness, KV pressure, and high-load throughput.

## Requirements

1. Implement a `Scheduler` class with:
   - `submit(request)` accepts a new request. Returns immediately.
   - `tick()` runs one model iteration. Returns the list of requests that completed this tick.
   - `has_active()` returns True if any request is in-flight or waiting.
   - `metrics()` returns `{ throughput_tokens_per_sec, avg_latency_s, p99_latency_s, gpu_utilization }`. `gpu_utilization` is the fraction of ticks where the batch was non-empty.
2. Iteration-level scheduling. The scheduler must keep multiple requests in-flight simultaneously. A request that finishes early must free its slot for a waiting request without delaying others in the batch.
3. KV memory cap. The active batch's total KV blocks (1 block = 16 tokens; see `inference_stub.kv_blocks_for`) must never exceed `MAX_KV_BLOCKS = 256`. When the cap is hit, new requests wait until memory frees.
4. EOS handling. A request stops when the stub emits the EOS token (id 0). The scheduler must detect EOS per request and remove that request from the batch in the same tick without affecting others.
5. Fairness across users. No single user can dominate. The grader runs a fairness test: 100 requests from user A and 5 from user B all submitted within one second. User B's median time-to-first-token must be within 2x of the time it would have taken if A weren't there.
6. Chunked prefill. Requests with prompts longer than 64 tokens require chunked prefill: split the prompt into chunks of 64, prefill in successive iterations. This prevents a long-prompt request from monopolizing one iteration and starving everyone else.

## Examples

**Example 1: Single request lifecycle**
```python
sched = Scheduler()
sched.submit(Request(user_id="u1", prompt=[1, 2, 3], max_tokens=10, request_id="r1"))
while sched.has_active():
    finished = sched.tick()
    if finished:
        print("done:", finished[0].request_id, len(finished[0].generated))
```

**Example 2: Mixed-load stream**
```python
sched = Scheduler()
for r in request_gen.stream(n=50):
    sched.submit(r)
while sched.has_active():
    sched.tick()
print(sched.metrics())  # throughput, latency, utilization
```

**Example 3: KV pressure**
```python
# Submit 30 long-running requests; KV cap is 256 blocks.
# Some requests should queue until others free memory.
# Throughput should be steady; no requests dropped or stuck.
```

## Submission Guidelines

### What to Submit
- `scheduler.py` with your Scheduler class, plus any supporting modules.

### How to Submit
```bash
litmus submit
```
