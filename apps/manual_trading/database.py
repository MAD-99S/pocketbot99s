"""Supabase database operations for predictions."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from apps.manual_trading.models import Prediction

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj: object) -> str:
    """Convert a dict to JSON string, replacing NaN/Inf with null.

    PostgreSQL JSON columns reject NaN tokens.  We walk the dict and
    swap any ``float('nan')`` or ``float('inf')`` values for ``None``
    so ``json.dumps`` produces valid JSON (``null``).
    """

    def _clean(val: object) -> object:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        if isinstance(val, dict):
            return {k: _clean(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_clean(v) for v in val]
        return val

    return json.dumps(_clean(obj))


class PredictionStore:
    """Reads and writes prediction rows via SQLAlchemy."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def insert(self, prediction: Prediction) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO predictions
                        (id, telegram_id, symbol, timeframe_sec, direction,
                         confidence, reasoning, indicators, entry_price,
                         entry_time, expiry_time, result,
                         candle_count, data_age_seconds, data_sufficiency_issues,
                         feature_snapshot, status)
                    VALUES
                        (:id, :telegram_id, :symbol, :timeframe_sec, :direction,
                         :confidence, :reasoning, :indicators, :entry_price,
                         :entry_time, :expiry_time, :result,
                         :candle_count, :data_age_seconds, :data_sufficiency_issues,
                         :feature_snapshot, :status)
                    """
                ),
                {
                    "id": str(prediction.id),
                    "telegram_id": prediction.telegram_id,
                    "symbol": prediction.symbol,
                    "timeframe_sec": prediction.timeframe_sec,
                    "direction": prediction.direction,
                    "confidence": prediction.confidence,
                    "reasoning": prediction.reasoning,
                    "indicators": _sanitize_for_json(prediction.indicators),
                    "entry_price": float(prediction.entry_price),
                    "entry_time": prediction.entry_time,
                    "expiry_time": prediction.expiry_time,
                    "result": prediction.result,
                    "candle_count": prediction.candle_count,
                    "data_age_seconds": prediction.data_age_seconds,
                    "data_sufficiency_issues": _sanitize_for_json(prediction.data_sufficiency_issues),
                    "feature_snapshot": _sanitize_for_json(prediction.feature_snapshot),
                    "status": prediction.status,
                },
            )
            await session.commit()
        logger.info(
            "prediction_inserted",
            prediction_id=str(prediction.id),
            symbol=prediction.symbol,
            direction=prediction.direction,
        )

    async def resolve(
        self,
        prediction_id: UUID,
        exit_price: Decimal,
        result: str,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE predictions
                    SET exit_price = :exit_price, result = :result
                    WHERE id = :id
                    """
                ),
                {
                    "id": str(prediction_id),
                    "exit_price": float(exit_price),
                    "result": result,
                },
            )
            await session.commit()
        logger.info(
            "prediction_resolved",
            prediction_id=str(prediction_id),
            result=result,
        )

    async def mark_result_requested(self, prediction_id: UUID) -> None:
        """Mark that we have sent a result-request message for this prediction."""
        async with self._session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE predictions
                    SET result_requested_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": str(prediction_id)},
            )
            await session.commit()

    async def has_pending_result(self, telegram_id: int) -> bool:
        """Return True if the user has an unresolved result request waiting.

        Checks for predictions where:
        - result IS NULL (not yet resolved)
        - result_requested_at IS NOT NULL (we already asked)
        - The user hasn't responded yet
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT 1 FROM predictions
                    WHERE telegram_id = :tid
                      AND result IS NULL
                      AND result_requested_at IS NOT NULL
                    LIMIT 1
                    """
                ),
                {"tid": telegram_id},
            )
            return result.scalar_one_or_none() is not None

    async def get_pending(self) -> list[dict]:
        """Return expired predictions that have NOT been asked for a result yet.

        Only returns rows where result_requested_at IS NULL so the
        tracker asks at most once per prediction.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT id, telegram_id, symbol, timeframe_sec, direction,
                           confidence, reasoning, indicators, entry_price,
                           entry_time, expiry_time
                    FROM predictions
                    WHERE result IS NULL
                      AND expiry_time <= :now
                      AND result_requested_at IS NULL
                    FOR UPDATE SKIP LOCKED
                    """
                ),
                {"now": datetime.now(timezone.utc)},
            )
            rows = result.mappings().all()
        return [dict(row) for row in rows]

    async def get_user_stats(self, telegram_id: int) -> dict:
        async with self._session_factory() as session:
            total = await session.execute(
                text(
                    "SELECT COUNT(*) as cnt FROM predictions "
                    "WHERE telegram_id = :tid AND result IS NOT NULL"
                ),
                {"tid": telegram_id},
            )
            total_count = total.scalar_one()

            wins = await session.execute(
                text(
                    "SELECT COUNT(*) as cnt FROM predictions "
                    "WHERE telegram_id = :tid AND result = 'win'"
                ),
                {"tid": telegram_id},
            )
            win_count = wins.scalar_one()

            by_symbol = await session.execute(
                text(
                    "SELECT symbol, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins "
                    "FROM predictions "
                    "WHERE telegram_id = :tid AND result IS NOT NULL "
                    "GROUP BY symbol ORDER BY total DESC"
                ),
                {"tid": telegram_id},
            )
            symbol_rows = by_symbol.mappings().all()

            by_confidence = await session.execute(
                text(
                    "SELECT "
                    "CASE "
                    "  WHEN confidence >= 0.9 THEN '90%+' "
                    "  WHEN confidence >= 0.8 THEN '80-89%' "
                    "  WHEN confidence >= 0.7 THEN '70-79%' "
                    "  ELSE '<70%' "
                    "END as bucket, "
                    "COUNT(*) as total, "
                    "SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins "
                    "FROM predictions "
                    "WHERE telegram_id = :tid AND result IS NOT NULL "
                    "GROUP BY bucket ORDER BY bucket DESC"
                ),
                {"tid": telegram_id},
            )
            conf_rows = by_confidence.mappings().all()

        return {
            "total": total_count,
            "wins": win_count,
            "losses": total_count - win_count,
            "win_rate": (win_count / total_count * 100) if total_count > 0 else 0.0,
            "by_symbol": [
                {"symbol": r["symbol"], "total": r["total"], "wins": r["wins"]}
                for r in symbol_rows
            ],
            "by_confidence": [
                {"bucket": r["bucket"], "total": r["total"], "wins": r["wins"]}
                for r in conf_rows
            ],
        }

    async def get_recent(self, telegram_id: int, limit: int = 5) -> list[dict]:
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT symbol, direction, confidence, entry_price,
                           exit_price, result, entry_time, expiry_time
                    FROM predictions
                    WHERE telegram_id = :tid
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"tid": telegram_id, "limit": limit},
            )
            rows = result.mappings().all()
        return [dict(row) for row in rows]


    async def get_labeled_for_calibration(self, limit: int = 1000) -> list[dict]:
        """Return labeled predictions (win/loss) with confidence values for
        building the reliability curve.

        Used by CalibrationStore.build_curve() to compute empirical win
        rates per confidence bucket.  Returns rows with symbol, direction,
        confidence, and result columns.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT symbol, direction, confidence, result
                    FROM predictions
                    WHERE result IN ('win', 'loss')
                      AND confidence IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            rows = result.mappings().all()
        return [dict(row) for row in rows]


class TrainingDataStore:
    """Stores feature snapshots and outcomes for ML model training.

    Each AI Analysis prediction stores its feature vector alongside
    the entry context. When the user reports Win/Loss/Tie, the
    corresponding prediction row is updated with the outcome. A
    training job can then join predictions + training_data to build
    labeled datasets.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def insert(
        self,
        symbol: str,
        timeframe_sec: int,
        direction: str,
        entry_price: float,
        features: dict,
        win_probability: float,
    ) -> None:
        """Store a feature snapshot for an AI Analysis prediction."""
        async with self._session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO training_data
                        (symbol, timeframe_sec, direction, entry_price,
                         features, win_probability)
                    VALUES
                        (:symbol, :timeframe_sec, :direction, :entry_price,
                         :features, :win_probability)
                    """
                ),
                {
                    "symbol": symbol,
                    "timeframe_sec": timeframe_sec,
                    "direction": direction,
                    "entry_price": entry_price,
                    "features": _sanitize_for_json(features),
                    "win_probability": win_probability,
                },
            )
            await session.commit()

    async def get_unlabeled(self, limit: int = 500) -> list[dict]:
        """Get training data rows that have been labeled with outcomes.

        Joins training_data with predictions to get feature vectors
        paired with their Win/Loss outcomes.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT
                        t.symbol,
                        t.timeframe_sec,
                        t.direction,
                        t.entry_price,
                        t.features,
                        t.win_probability,
                        p.result,
                        p.entry_time,
                        p.expiry_time
                    FROM training_data t
                    JOIN predictions p ON
                        p.telegram_id = (
                            SELECT telegram_id FROM predictions
                            WHERE entry_price = t.entry_price
                              AND symbol = t.symbol
                              AND entry_time IS NOT NULL
                            LIMIT 1
                        )
                        AND p.symbol = t.symbol
                        AND ABS(p.entry_price - t.entry_price) < 0.0001
                        AND p.result IS NOT NULL
                    WHERE p.result IN ('win', 'loss')
                    ORDER BY t.created_at DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            rows = result.mappings().all()
        return [dict(row) for row in rows]

    async def count_labeled(self) -> int:
        """Count how many labeled training samples we have."""
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT COUNT(*) as cnt FROM training_data t "
                    "JOIN predictions p ON p.symbol = t.symbol "
                    "AND ABS(p.entry_price - t.entry_price) < 0.0001 "
                    "WHERE p.result IN ('win', 'loss')"
                )
            )
            return result.scalar_one()


class AISignalStore:
    """Logs every AI analysis attempt for performance analysis.

    Every call to the AI analysis engine — whether it produced a signal
    or not — is logged here so you can compute real win rates including
    "declined to answer" cases.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def insert(
        self,
        symbol: str,
        has_signal: bool,
        direction: str,
        confidence: float,
        reasoning: str,
        shadow_mode: bool,
        model_response_raw: str,
        prediction_id: UUID | None = None,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO ai_signals
                        (symbol, has_signal, direction, confidence, reasoning,
                         shadow_mode, model_response_raw, prediction_id)
                    VALUES
                        (:symbol, :has_signal, :direction, :confidence, :reasoning,
                         :shadow_mode, :model_response_raw, :prediction_id)
                    """
                ),
                {
                    "symbol": symbol,
                    "has_signal": has_signal,
                    "direction": direction,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "shadow_mode": shadow_mode,
                    "model_response_raw": model_response_raw,
                    "prediction_id": str(prediction_id) if prediction_id else None,
                },
            )
            await session.commit()

    async def update_outcome(self, prediction_id: UUID, outcome: str) -> None:
        """Update ai_signals outcome when a prediction is resolved."""
        async with self._session_factory() as session:
            await session.execute(
                text(
                    """
                    UPDATE ai_signals
                    SET outcome = :outcome, scored_at = NOW()
                    WHERE prediction_id = :prediction_id
                    """
                ),
                {
                    "prediction_id": str(prediction_id),
                    "outcome": outcome,
                },
            )
            await session.commit()
