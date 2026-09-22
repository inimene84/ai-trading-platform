"""Deterministic social pre-processing before a Jev call.

Keyword polarity is a feature of the state, not a trade signal.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

FEAR_KEYWORDS = {
    "crash", "dump", "liquidation", "sell", "selling", "dead", "bear", "bearish",
    "scam", "rekt", "drop", "loss", "bleeding", "panic", "fear", "down", "dip", "fall",
}
GREED_KEYWORDS = {
    "pump", "moon", "ath", "buy", "buying", "gem", "bull", "bullish", "breakout",
    "rally", "gain", "accumulate", "rocket", "up", "long", "hold", "squeeze",
}


def _empty() -> dict[str, Any]:
    return {
        "sample_size": 0,
        "unique_authors_count": 0,
        "author_diversity_pct": 0.0,
        "total_likes": 0,
        "total_retweets": 0,
        "avg_engagement": 0.0,
        "fear_mentions": 0,
        "greed_mentions": 0,
        "polarity_score": 0.0,
        "weighted_polarity_score": 0.0,
        "sentiment_label": "Neutral",
        "bot_downweight_mean": 1.0,
        "duplicate_text_clusters": 0,
        "stratified_sample": [],
    }


def _text_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def quality_weights(posts: list[dict[str, Any]]) -> list[float]:
    """Down-weight copy-paste clusters and bursty authors. Never drop a post.

    Dropping suspected bots biases a squeeze tape. A weight in (0, 1] keeps
    the post in the sample and lets later stages discount it.
    """
    if not posts:
        return []
    hashes = [_text_hash(str(post.get("text") or "")) for post in posts]
    authors = [str(post.get("author_username") or "unknown").lower() for post in posts]
    hash_counts = Counter(hashes)
    author_counts = Counter(authors)
    weights: list[float] = []
    for digest, author in zip(hashes, authors):
        dup_extra = max(0, hash_counts[digest] - 1)
        burst = max(0, author_counts[author] - 3)
        weights.append(round(1.0 / (1.0 + dup_extra + 0.25 * burst), 4))
    return weights


def process_posts(posts: list[dict[str, Any]], sample_limit: int = 25) -> dict[str, Any]:
    if not posts:
        return _empty()

    total_likes = 0
    total_retweets = 0
    authors: set[str] = set()
    fear_count = 0
    greed_count = 0

    for post in posts:
        likes = int(post.get("likes") or 0)
        retweets = int(post.get("retweets") or 0)
        total_likes += likes
        total_retweets += retweets
        authors.add(str(post.get("author_username") or "unknown").lower())
        words = set(str(post.get("text") or "").lower().replace("$", "").replace("#", "").split())
        fear_count += len(words.intersection(FEAR_KEYWORDS))
        greed_count += len(words.intersection(GREED_KEYWORDS))

    weights = quality_weights(posts)
    weight_by_id = {
        str(post.get("id") or ""): weight for post, weight in zip(posts, weights)
    }
    sample_size = len(posts)
    unique_authors = len(authors)
    total_polar = fear_count + greed_count
    polarity = round((greed_count - fear_count) / total_polar, 4) if total_polar else 0.0
    if polarity <= -0.4:
        label = "Extreme Panic"
    elif polarity < -0.1:
        label = "Bearish / Fearful"
    elif polarity <= 0.1:
        label = "Neutral / Mixed"
    elif polarity < 0.4:
        label = "Bullish / Optimistic"
    else:
        label = "Euphoric / Greedy"

    ranked = sorted(
        posts,
        key=lambda row: int(row.get("likes") or 0) + int(row.get("retweets") or 0) * 2,
        reverse=True,
    )
    seen: set[str] = set()
    stratified: list[dict[str, Any]] = []
    for kind, batch in (("high_engagement", ranked[:sample_limit]), ("latest_breaking", posts[:sample_limit])):
        for post in batch:
            post_id = str(post.get("id") or "")
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            text = str(post.get("text") or "")[:280]
            stratified.append({
                "author": post.get("author_username"),
                "text": text,
                "likes": int(post.get("likes") or 0),
                "type": kind,
                "quality_weight": weight_by_id.get(post_id, 1.0),
                "untrusted_text": True,
            })

    weighted_polar = 0.0
    weight_sum = sum(weights) or 1.0
    if total_polar:
        for post, weight in zip(posts, weights):
            words = set(str(post.get("text") or "").lower().replace("$", "").replace("#", "").split())
            fear_hits = len(words.intersection(FEAR_KEYWORDS))
            greed_hits = len(words.intersection(GREED_KEYWORDS))
            if fear_hits or greed_hits:
                local = (greed_hits - fear_hits) / (fear_hits + greed_hits)
                weighted_polar += local * weight
        weighted_polar = round(weighted_polar / weight_sum, 4)
    hash_counts = Counter(_text_hash(str(post.get("text") or "")) for post in posts)
    duplicate_clusters = sum(1 for count in hash_counts.values() if count > 1)

    return {
        "sample_size": sample_size,
        "unique_authors_count": unique_authors,
        "author_diversity_pct": round((unique_authors / sample_size) * 100.0, 1),
        "total_likes": total_likes,
        "total_retweets": total_retweets,
        "avg_engagement": round((total_likes + total_retweets * 2) / sample_size, 2),
        "fear_mentions": fear_count,
        "greed_mentions": greed_count,
        "polarity_score": polarity,
        "weighted_polarity_score": weighted_polar,
        "sentiment_label": label,
        "bot_downweight_mean": round(sum(weights) / len(weights), 4),
        "duplicate_text_clusters": duplicate_clusters,
        "stratified_sample": stratified,
    }
