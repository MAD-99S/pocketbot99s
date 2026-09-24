"""
Candle persistence — stores completed candles to Postgres so history survives
restarts instead of being held in a 200-item in-memory list and lost.

Schema (idempotent, created by main.py on startup)::

    CREATE TABLE IF NOT EXISTS candles (
        symbol TEXT NOT NULL,
        timeframe_sec INT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        open DOUBLE PRECISION NOT NULL,
        high DOUBLE PRECISION NOT NULL,
        low DOUBLE PRECISION NOT NULL,
        close DOUBLE PRECISION NOT NULL,
        volume DOUBLE PRECISION NOT NULL DEFAULT 0,
        ticks_in_candle INT NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (symbol, timeframe_sec, timestamp)
    );
    CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe
        ON candles (symbol, timeframe_sec, timestamp DESC);

Integration:
- _CandleBuilder.finalise_candle() → call CandleStore.insert_completed()
- On startup, MarketDataCollector.load_history() → load from DB into memory
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


class CandleStore:
    """Persists completed candles to Postgres; survives restarts."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._factory = session_factory

    async def insert_completed(
        self,
        symbol: str,
        timeframe_sec: int,
        candles: list[dict],
    ) -> int:
        """Insert a batch of completed candles. Returns rows inserted."""
        if not candles:
            return 0

        now = datetime.now(timezone.utc)
        rows = [
            {
                "symbol": symbol,
                "timeframe_sec": timeframe_sec,
                "timestamp": c["timestamp"],
                "open": c["open"],
                "high": c["high"],
                "low": c["low"],
                "close": c["close"],
                "volume": c.get("volume", 0),
                "ticks_in_candle": c.get("ticks_in_candle", 0),
                "created_at": now,
            }
            for c in candles
        ]

        async with self._factory() as session:
            for row in rows:
                await session.execute(
                    text(
                        """
                        INSERT INTO candles
                            (symbol, timeframe_sec, timestamp, open, high, low, close, volume, ticks_in_candle, created_at)
                        VALUES
                            (:symbol, :timeframe_sec, :timestamp, :open, :high, :low, :close, :volume, :ticks_in_candle, :created_at)
                        ON CONFLICT (symbol, timeframe_sec, timestamp) DO NOTHING
                        """
                    ),
                    row,
                )
            await session.commit()
        logger.debug(
            "candles_persisted",
            symbol=symbol,
            timeframe=timeframe_sec,
            count=len(rows),
        )
        return len(rows)

    async def load_history(
        self,
        symbol: str,
        timeframe_sec: int,
        limit: int = 500,
    ) -> list[dict]:
        """Load persisted candles for a symbol+timeframe, most-recent first."""
        async with self._factory() as session:
            result = await session.execute(
                text(
                    """
                    SELECT timestamp, open, high, low, close, volume, ticks_in_candle
                    FROM candles
                    WHERE symbol = :symbol
                      AND timeframe_sec = :timeframe_sec
                    ORDER BY timestamp DESC
                    LIMIT :limit
                    """
                ),
                {"symbol": symbol, "timeframe_sec": timeframe_sec, "limit": limit},
            )
            rows = result.mappings().all()

        # Reverse so oldest-first, matching in-memory convention
        candles: list[dict] = []
        for row in reversed(list(rows)):
            candles.append({
                "timestamp": float(row["timestamp"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"] or 0),
                "ticks_in_candle": int(row["ticks_in_candle"] or 0),
            })
        logger.info(
            "candles_loaded_from_db",
            symbol=str(symbol),
            timeframe=str(timeframe_sec),
            count=len(candles),
        )
        return candles
