"""
inference_stub.py — stubbed GPU running an LLM.

forward(batch) runs one iteration: advances each request in the batch by
one token. Latency scales linearly with batch size (simulating one GPU
forward pass per iteration; in real systems it's sub-linear, but linear
is fine for testing scheduler behavior).

The grader replaces this with a real GPU/model at evaluation time. Your
scheduler should not depend on the stub's internals.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

VOCAB_SIZE = 64
EOS_TOKEN = 0
KV_BLOCK_TOKENS = 16
MAX_KV_BLOCKS = 256

# Latency model
_BASE_LATENCY_S = 0.005
_PER_REQ_LATENCY_S = 0.001


@dataclass
class Request:
    user_id: str
    prompt: list[int]
    max_tokens: int
    request_id: str = ""
    submitted_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    generated: list[int] = field(default_factory=list)


def forward(batch: list[Request]) -> list[int]:
    """Run one iteration. Return one new token per request in batch."""
    if not batch:
        return []
    time.sleep(_BASE_LATENCY_S + len(batch) * _PER_REQ_LATENCY_S)
    rng = np.random.default_rng(int(time.time() * 1_000_000) % (2**32))
    out = []
    for r in batch:
        gen_count = len(r.generated)
        # Emit EOS with rising probability as the request approaches its max.
        if gen_count >= r.max_tokens - 1:
            out.append(EOS_TOKEN)
        elif rng.random() < 0.02:
            out.append(EOS_TOKEN)
        else:
            out.append(int(rng.integers(1, VOCAB_SIZE)))
    return out


def kv_blocks_for(request: Request) -> int:
    """KV blocks consumed by a request at its current token count."""
    total_tokens = len(request.prompt) + len(request.generated)
    return (total_tokens + KV_BLOCK_TOKENS - 1) // KV_BLOCK_TOKENS
