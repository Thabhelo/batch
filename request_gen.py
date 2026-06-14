"""
request_gen.py — generates request streams for testing your scheduler.
"""

import numpy as np
from inference_stub import Request


def stream(n: int = 50, lam: float = 5.0, seed: int = 42) -> list[Request]:
    """
    Build n Requests with Poisson inter-arrival times (lambda = lam req/sec).

    Returns a list; the candidate's harness drives the actual submit timing.
    """
    rng = np.random.default_rng(seed)
    out: list[Request] = []
    t = 0.0
    for i in range(n):
        t += rng.exponential(1.0 / lam)
        prompt_len = int(rng.integers(5, 200))
        max_tokens = int(rng.integers(10, 100))
        user = f"u{rng.integers(1, 6)}"
        out.append(Request(
            user_id=user,
            prompt=list(rng.integers(1, 64, size=prompt_len).astype(int)),
            max_tokens=max_tokens,
            request_id=f"r{i}",
            submitted_at=t,
        ))
    return out
