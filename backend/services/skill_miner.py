"""
Skill Miner — Turn Recalled Trade Outcomes into Named, Reusable Strategies
==========================================================================
The semantic trade memory (Track C) remembers *individual* trades. The skill
miner is the next layer up: it clusters those trades into recurring **market-
setup archetypes** ("skills"), scores each one's realised edge, gives it a
human-readable name, and persists it in the `strategy_skills` table.

At decision time the opinion layer can then match the *current* setup to the
best-fitting learned skill and vote with that skill's historical bias — a
compact, inspectable, leaderboard-able distillation of "what has actually
worked for this agent".

Design (consistent with trade_memory):
  • No heavy ML dependency. Clustering is a deterministic greedy cosine
    grouping over the same feature vectors trade_memory uses, so a "skill" is a
    cluster of trades whose market context was similar.
  • Pure functions for clustering/scoring so they're unit-testable without a DB.
  • Mining never raises into any caller; it returns a structured summary.

Env:
  SKILL_MINER_ENABLED          (default true)
  SKILL_MIN_CLUSTER_SIZE       (default 4)     # trades needed to form a skill
  SKILL_SIMILARITY_THRESHOLD   (default 0.80)  # cosine cutoff for same cluster
  SKILL_MINE_INTERVAL_MIN      (default 360)   # background re-mine cadence (6h)
  SKILL_MINE_LOOKBACK          (default 2000)  # closed trades scanned per mine
"""

from __future__ import annotations

import os
import math
import hashlib
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.services.trade_memory import feature_vector, _FEATURE_ORDER

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ─────────────────────────────────────────────────────────────────────────────
# Pure math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _normalise(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _centroid(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    acc = [x / len(vectors) for x in acc]
    return _normalise(acc)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeSample:
    """A closed trade reduced to what the miner needs."""
    symbol: str
    direction: str
    pnl: float
    context: Dict[str, Any]
    vector: List[float] = field(default_factory=list)

    def ensure_vector(self, size: int):
        if not self.vector:
            self.vector = feature_vector(self.context, size)


@dataclass
class MinedSkill:
    skill_key: str
    name: str
    description: str
    centroid: List[float]
    feature_summary: Dict[str, float]
    direction: str
    sample_count: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    sharpe: Optional[float]
    edge_score: float
    symbols: List[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid"] = [round(x, 6) for x in self.centroid]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Clustering + scoring (pure)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_samples(
    samples: List[TradeSample],
    threshold: float,
    vector_size: int,
) -> List[List[TradeSample]]:
    """Deterministic greedy cosine clustering.

    Process samples in a stable order (by descending |pnl| so the strongest
    outcomes seed clusters first). Each sample joins the existing cluster whose
    centroid is most similar above `threshold`, else seeds a new cluster.
    Centroids are recomputed incrementally.
    """
    for s in samples:
        s.ensure_vector(vector_size)

    ordered = sorted(samples, key=lambda s: (-abs(s.pnl), s.symbol, s.direction))

    clusters: List[List[TradeSample]] = []
    centroids: List[List[float]] = []

    for s in ordered:
        best_idx = -1
        best_sim = threshold
        for i, c in enumerate(centroids):
            sim = _cosine(s.vector, c)
            if sim >= best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0:
            clusters[best_idx].append(s)
            centroids[best_idx] = _centroid([x.vector for x in clusters[best_idx]])
        else:
            clusters.append([s])
            centroids.append(list(s.vector))
    return clusters


def _stdev(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var)


def score_cluster(cluster: List[TradeSample]) -> Tuple[str, float, float, float, Optional[float], float]:
    """Return (direction, win_rate, avg_pnl, total_pnl, sharpe, edge_score).

    Edge is expectancy-based and fail-closed for losers / tiny samples:
    0% win-rate or non-positive avg PnL scores 0 so they cannot rank above
    high-sample winning clusters.
    """
    n = len(cluster)
    pnls = [c.pnl for c in cluster]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n if n else 0.0
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / n if n else 0.0
    sd = _stdev(pnls)
    sharpe = (avg_pnl / sd) if sd > 0 else None

    if avg_pnl > 0 and win_rate >= 0.5:
        direction = "bullish"
    elif avg_pnl < 0 and win_rate <= 0.5:
        direction = "bearish"
    else:
        direction = "neutral"

    min_samples = int(os.getenv("SKILL_MIN_SAMPLES", "8") or 8)
    if n < min_samples or win_rate <= 0.0 or avg_pnl <= 0.0:
        return (
            direction, round(win_rate, 4), round(avg_pnl, 6), round(total_pnl, 6),
            (round(sharpe, 4) if sharpe is not None else None), 0.0,
        )

    losses = [p for p in pnls if p < 0]
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    # Expectancy in R if we have a typical loss; else scale by avg win.
    if avg_loss > 0:
        expectancy_r = avg_pnl / avg_loss
    else:
        expectancy_r = 1.0
    edge = min(max(expectancy_r, 0.0) * win_rate, 1.0)
    support = min(n / 20.0, 1.0)
    consistency = 1.0
    if sharpe is not None:
        consistency = max(0.0, min(max(sharpe, 0.0) / 2.0, 1.0))
    edge_score = round(edge * (0.5 + 0.5 * support) * (0.5 + 0.5 * consistency), 4)
    return direction, round(win_rate, 4), round(avg_pnl, 6), round(total_pnl, 6), \
        (round(sharpe, 4) if sharpe is not None else None), edge_score


def _summary_features(centroid: List[float]) -> Dict[str, float]:
    """Map the leading dims of the centroid back to named features."""
    out: Dict[str, float] = {}
    for i, key in enumerate(_FEATURE_ORDER):
        if i < len(centroid):
            out[key] = round(centroid[i], 4)
    return out


def name_skill(feature_summary: Dict[str, float], symbols: List[str], direction: str) -> str:
    """Produce a human-readable name from the centroid features."""
    parts: List[str] = []
    regime = feature_summary.get("regime", 0.0)
    if regime > 0.33:
        parts.append("Trending-up")
    elif regime < -0.33:
        parts.append("Trending-down")
    else:
        parts.append("Ranging")

    mom = feature_summary.get("momentum", 0.0)
    if mom > 0.33:
        parts.append("bullish-momentum")
    elif mom < -0.33:
        parts.append("bearish-momentum")

    rsi = feature_summary.get("rsi", 0.0)
    if rsi > 0.4:
        parts.append("overbought")
    elif rsi < -0.4:
        parts.append("oversold")

    funding = feature_summary.get("funding", 0.0)
    if funding > 0.33:
        parts.append("high-funding")
    elif funding < -0.33:
        parts.append("neg-funding")

    sym_part = ""
    uniq = sorted(set(symbols))
    if len(uniq) == 1:
        sym_part = f" [{uniq[0]}]"
    elif 1 < len(uniq) <= 3:
        sym_part = f" [{', '.join(uniq)}]"

    label = " ".join(parts) if parts else "Mixed setup"
    return f"{label} \u2192 {direction}{sym_part}"


def skill_key_for(centroid: List[float], direction: str) -> str:
    """Stable key so re-mining a similar cluster updates the same skill row."""
    # Quantise the leading feature dims so tiny drift doesn't fork a new skill.
    lead = centroid[: len(_FEATURE_ORDER)]
    quant = ",".join(str(round(x * 4) / 4) for x in lead)  # 0.25 buckets
    raw = f"{quant}|{direction}"
    # Dedup digest, not a security MAC.
    return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]


def mine_skills(
    samples: List[TradeSample],
    *,
    min_cluster_size: int,
    threshold: float,
    vector_size: int,
) -> List[MinedSkill]:
    """Pure end-to-end: cluster → score → name. Returns skills meeting the
    minimum cluster size, sorted by edge_score desc."""
    clusters = cluster_samples(samples, threshold, vector_size)
    skills: List[MinedSkill] = []
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        cen = _centroid([c.vector for c in cluster])
        direction, win_rate, avg_pnl, total_pnl, sharpe, edge_score = score_cluster(cluster)
        fsum = _summary_features(cen)
        symbols = [c.symbol for c in cluster]
        name = name_skill(fsum, symbols, direction)
        key = skill_key_for(cen, direction)
        skills.append(MinedSkill(
            skill_key=key, name=name,
            description=(
                f"Mined from {len(cluster)} trades. "
                f"{win_rate:.0%} win-rate, avg PnL {avg_pnl:+.4f}."
            ),
            centroid=cen, feature_summary=fsum, direction=direction,
            sample_count=len(cluster), win_rate=win_rate, avg_pnl=avg_pnl,
            total_pnl=total_pnl, sharpe=sharpe, edge_score=edge_score,
            symbols=sorted(set(symbols)),
        ))
    skills.sort(key=lambda s: s.edge_score, reverse=True)
    return skills


# ─────────────────────────────────────────────────────────────────────────────
# Service (DB-bound)
# ─────────────────────────────────────────────────────────────────────────────

class SkillMinerService:
    def __init__(self):
        self.enabled = _env_bool("SKILL_MINER_ENABLED", True)
        self.min_cluster_size = int(os.getenv("SKILL_MIN_CLUSTER_SIZE", "4"))
        self.threshold = float(os.getenv("SKILL_SIMILARITY_THRESHOLD", "0.80"))
        self.vector_size = int(os.getenv("TRADE_MEMORY_VECTOR_SIZE", "64"))
        self.lookback = int(os.getenv("SKILL_MINE_LOOKBACK", "2000"))
        self._task = None

    def start(self) -> None:
        """Start the background miner loop task."""
        import asyncio
        if not self.enabled:
            return
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self.run_miner_loop())
        logger.info("SkillMiner service loop started")

    async def stop(self) -> None:
        """Stop the background miner loop task gracefully."""
        if self._task and not self._task.done():
            logger.info("Stopping SkillMiner miner loop...")
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("SkillMiner miner loop stopped")
        self._task = None

    # -- load --
    def _load_samples(self, limit: int) -> List[TradeSample]:
        from backend.database.connection import SessionLocal
        from backend.database.models import Trade
        import sqlalchemy as sa
        out: List[TradeSample] = []
        with SessionLocal() as db:
            trades = (
                db.query(Trade)
                .filter(sa.and_(
                    Trade.status == "closed",
                    Trade.pnl.isnot(None),
                    Trade.exit_price.isnot(None),
                ))
                .order_by(Trade.closed_at.desc())
                .limit(limit)
                .all()
            )
        for t in trades:
            ctx: Dict[str, Any] = {"direction": t.direction}
            try:
                if t.entry_price and t.exit_price:
                    move = (t.exit_price - t.entry_price) / t.entry_price
                    ctx["kronos_change_pct"] = move * 100.0
            except Exception:
                pass
            if getattr(t, "strategy", None):
                ctx["strategy"] = t.strategy
            out.append(TradeSample(
                symbol=t.symbol, direction=t.direction or "",
                pnl=float(t.pnl), context=ctx,
            ))
        return out

    # -- mine + persist --
    def mine_and_store(self, limit: Optional[int] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "disabled", "skills": 0}
        try:
            samples = self._load_samples(limit or self.lookback)
            if not samples:
                return {"ok": True, "skills": 0, "samples": 0,
                        "note": "no closed trades to mine"}
            mined = mine_skills(
                samples,
                min_cluster_size=self.min_cluster_size,
                threshold=self.threshold,
                vector_size=self.vector_size,
            )
            stored = self._upsert(mined)
            return {"ok": True, "skills": len(mined), "stored": stored,
                    "samples": len(samples)}
        except Exception as e:
            logger.warning(f"SkillMiner: mine_and_store failed: {e}")
            return {"ok": False, "reason": str(e), "skills": 0}

    def _upsert(self, mined: List[MinedSkill]) -> int:
        from backend.database.connection import SessionLocal
        from backend.database.models import StrategySkill
        now = datetime.now(timezone.utc)
        stored = 0

        # Deduplicate mined list by skill_key (keeping the first one, which has the highest edge score)
        seen_keys = set()
        deduped = []
        for m in mined:
            if m.skill_key not in seen_keys:
                seen_keys.add(m.skill_key)
                deduped.append(m)
        mined = deduped

        active_keys = {m.skill_key for m in mined}
        with SessionLocal() as db:
            for m in mined:
                row = db.query(StrategySkill).filter(
                    StrategySkill.skill_key == m.skill_key
                ).one_or_none()
                if row is None:
                    row = StrategySkill(skill_key=m.skill_key)
                    db.add(row)
                row.name = m.name
                row.description = m.description
                row.centroid = m.centroid
                row.feature_summary = m.feature_summary
                row.direction = m.direction
                row.sample_count = m.sample_count
                row.win_rate = m.win_rate
                row.avg_pnl = m.avg_pnl
                row.total_pnl = m.total_pnl
                row.sharpe = m.sharpe
                row.edge_score = m.edge_score
                row.symbols = m.symbols
                row.active = True
                row.last_mined_at = now
                stored += 1
            # Deactivate skills that no longer appear (stale archetypes).
            existing = db.query(StrategySkill).all()
            for row in existing:
                if row.skill_key not in active_keys:
                    row.active = False
            db.commit()
        return stored

    # -- query / match --
    def list_skills(self, active_only: bool = True, limit: int = 50) -> List[dict]:
        from backend.database.connection import SessionLocal
        from backend.database.models import StrategySkill
        try:
            with SessionLocal() as db:
                q = db.query(StrategySkill)
                if active_only:
                    q = q.filter(StrategySkill.active == True)  # noqa: E712
                rows = q.order_by(StrategySkill.edge_score.desc()).limit(limit).all()
                return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"SkillMiner: list_skills failed: {e}")
            return []

    def match_skill(self, context: Dict[str, Any]) -> Optional[dict]:
        """Find the active skill whose centroid best matches the live context.
        Returns the skill dict augmented with a `similarity` field, or None."""
        try:
            vec = feature_vector(context, self.vector_size)
            best = None
            best_sim = self.threshold
            for s in self.list_skills(active_only=True, limit=200):
                cen = s.get("centroid") or []
                sim = _cosine(vec, cen)
                if sim >= best_sim:
                    best_sim = sim
                    best = {**s, "similarity": round(sim, 4)}
            return best
        except Exception as e:
            logger.warning(f"SkillMiner: match_skill failed: {e}")
            return None

    @staticmethod
    def _row_to_dict(r) -> dict:
        return {
            "skill_key": r.skill_key,
            "name": r.name,
            "description": r.description,
            "direction": r.direction,
            "sample_count": r.sample_count,
            "win_rate": r.win_rate,
            "avg_pnl": r.avg_pnl,
            "total_pnl": r.total_pnl,
            "sharpe": r.sharpe,
            "edge_score": r.edge_score,
            "symbols": r.symbols,
            "active": r.active,
            "centroid": r.centroid,
            "feature_summary": r.feature_summary,
            "last_mined_at": r.last_mined_at.isoformat() if r.last_mined_at else None,
        }

    def status(self) -> Dict[str, Any]:
        info = {
            "enabled": self.enabled,
            "min_cluster_size": self.min_cluster_size,
            "similarity_threshold": self.threshold,
            "vector_size": self.vector_size,
            "skills_active": 0,
            "skills_total": 0,
        }
        try:
            from backend.database.connection import SessionLocal
            from backend.database.models import StrategySkill
            with SessionLocal() as db:
                info["skills_total"] = db.query(StrategySkill).count()
                info["skills_active"] = db.query(StrategySkill).filter(
                    StrategySkill.active == True  # noqa: E712
                ).count()
        except Exception as e:
            info["error"] = str(e)
        return info

    async def run_miner_loop(self) -> None:
        """Background loop: periodically re-mine skills from trade history."""
        import asyncio
        if not self.enabled:
            logger.info("SkillMiner loop not started (disabled)")
            return
        interval = max(5, int(os.getenv("SKILL_MINE_INTERVAL_MIN", "360")))
        logger.info(f"SkillMiner loop started (every {interval}m)")
        try:
            res = await asyncio.to_thread(self.mine_and_store)
            logger.info(f"SkillMiner initial mine: {res}")
        except Exception as e:
            logger.warning(f"SkillMiner initial mine failed: {e}")
        while True:
            try:
                await asyncio.sleep(interval * 60)
                res = await asyncio.to_thread(self.mine_and_store)
                logger.info(f"SkillMiner mine: {res}")
            except asyncio.CancelledError:
                logger.info("SkillMiner loop cancelled")
                raise
            except Exception as e:
                logger.warning(f"SkillMiner loop iteration failed: {e}")


# Singleton
skill_miner = SkillMinerService()
