"""Deterministic social pre-processing before a Jev call.

Keyword polarity is a feature of the state, not a trade signal.
"""

from __future__ import annotations

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
        "sentiment_label": "Neutral",
        "stratified_sample": [],
    }


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
            })

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
        "sentiment_label": label,
        "stratified_sample": stratified,
    }
