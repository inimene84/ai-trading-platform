"""Unit tests for token-bucket rate limiter (sync + async + reservation + cache)."""
import asyncio
import time
from backend.utils.rate_limit import (
    Bucket,
    _LIMITS,
    _buckets,
    _skip_counts,
    acquire,
    acquire_async,
    cached_call,
    cached_call_async,
    get_rate_limiter_status,
)


def test_bucket_refills():
    b = Bucket(capacity=2.0, refill_per_sec=1000.0)
    assert b.take() == 0.0
    assert b.take() == 0.0
    assert b.take(max_wait=1.0) > 0.0  # third must wait
    time.sleep(0.01)
    assert b.take() == 0.0  # refilled


def test_bucket_reserves_tokens_across_concurrent_waiters():
    b = Bucket(capacity=1.0, refill_per_sec=10.0)
    assert b.take(1.0) == 0.0
    w1 = b.take(1.0, max_wait=5.0)
    w2 = b.take(1.0, max_wait=5.0)
    assert 0.05 <= w1 <= 0.15
    assert w2 > w1  # second waiter is serialized behind first waiter


def test_unknown_provider_defaults():
    _buckets.pop("unknown-x", None)
    assert acquire("unknown-x") is True


def test_max_wait_skip_does_not_drain_tokens():
    _buckets.clear()
    _skip_counts.clear()
    _LIMITS["tiny"] = (1.0, 1.0 / 3600.0)
    try:
        assert acquire("tiny") is True
        assert acquire("tiny", max_wait=0.1) is False  # would need ~an hour
        status = get_rate_limiter_status()
        assert status["tiny"]["consecutive_skips"] == 1
    finally:
        _LIMITS.pop("tiny", None)
        _buckets.pop("tiny", None)


def test_cached_call_serves_fresh_and_stale():
    _buckets.clear()
    _LIMITS["strict"] = (1.0, 1.0 / 3600.0)
    cache = {}
    calls = {"n": 0}

    def _fn():
        calls["n"] += 1
        return {"v": calls["n"]}

    try:
        # 1st call acquires token and populates cache
        v1 = cached_call("strict", cache, "k1", ttl_sec=60.0, fn=_fn, max_wait=0.05)
        assert v1 == {"v": 1}
        # 2nd call within TTL serves fresh cache without hitting limiter
        v2 = cached_call("strict", cache, "k1", ttl_sec=60.0, fn=_fn, max_wait=0.05)
        assert v2 == {"v": 1}
        assert calls["n"] == 1
        # Expire cache; limiter rejects -> stale cache is served
        cache["k1"] = (time.time() - 120.0, v1)
        v3 = cached_call("strict", cache, "k1", ttl_sec=60.0, fn=_fn, max_wait=0.05)
        assert v3 == {"v": 1}
        assert calls["n"] == 1
    finally:
        _LIMITS.pop("strict", None)
        _buckets.pop("strict", None)


def test_async_acquire_and_cached_call():
    async def _run():
        _buckets.pop("async-prov", None)
        ok = await acquire_async("async-prov", max_wait=1.0)
        assert ok is True
        cache = {}

        async def _afn():
            return 42

        val = await cached_call_async("async-prov", cache, "ak", ttl_sec=30.0, fn=_afn)
        assert val == 42

    asyncio.run(_run())
