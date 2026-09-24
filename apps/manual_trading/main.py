"""Entry point for the manual trading bot.

Usage:
    python -m apps.manual_trading.main

This starts the Telegram bot with the shared demo broker connection
for market data. No per-user SSID is needed for manual trading mode.
"""
from __future__ import annotations

import asyncio
import signal as signal_module
import sys
from pathlib import Path

import structlog

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from config.settings import load_settings
from infrastructure.persistence.database import init_db
from infrastructure.broker.pocket_option import PocketOptionBroker
from apps.manual_trading.bot import ManualTradingBot
from apps.manual_trading.database import PredictionStore, TrainingDataStore, AISignalStore
from apps.manual_trading.strategies.ai_analysis.engine import AIAnalysisEngine
from apps.manual_trading.market_data import MarketDataCollector
from apps.manual_trading.candle_store import CandleStore
from apps.manual_trading.trade_tracker import TradeTracker
from infrastructure.ml.model import TradingModel

logger = structlog.get_logger()

RECONNECT_BASE_DELAY = 5
RECONNECT_MAX_DELAY = 120


async def broker_reconnect_loop(broker: PocketOptionBroker) -> None:
    """Continuously try to reconnect the broker if disconnected."""
    delay = RECONNECT_BASE_DELAY
    while True:
        try:
            if not await broker.is_connected():
                logger.info("broker_reconnect_attempt", delay=delay)
                try:
                    await broker.connect()
                    delay = RECONNECT_BASE_DELAY
                    logger.info("broker_reconnected")
                except Exception as e:
                    logger.error("broker_reconnect_failed", error=str(e))
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, RECONNECT_MAX_DELAY)
            else:
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("reconnect_loop_error")
            await asyncio.sleep(delay)


async def main() -> None:
    """Start the manual trading bot."""
    settings = load_settings()
    logger.info("starting_manual_trading_bot")

    # Initialize database (with retry)
    engine = None
    session_factory = None
    for attempt in range(1, 6):
        try:
            engine, session_factory = await init_db(settings.postgres)
            logger.info("db_connected", attempt=attempt)
            break
        except Exception as e:
            logger.warning("db_connect_failed", attempt=attempt, error=str(e))
            if attempt < 5:
                await asyncio.sleep(10)

    if session_factory is None:
        logger.error("db_unavailable_cannot_start")
        return

    # Ensure result_requested_at column exists (idempotent migration)
    try:
        async with engine.connect() as conn:
            await conn.execute(text(
                "ALTER TABLE predictions "
                "ADD COLUMN IF NOT EXISTS result_requested_at TIMESTAMPTZ"
            ))
            await conn.commit()
        logger.info("migration_result_requested_at_applied")
    except Exception:
        logger.warning("migration_result_requested_at_failed", exc_info=True)

    # Ensure training_data table exists (idempotent migration)
    try:
        async with engine.connect() as conn:
            await conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS training_data (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe_sec INT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    features JSONB NOT NULL,
                    win_probability DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            ))
            await conn.commit()
        logger.info("migration_training_data_applied")
    except Exception:
        logger.warning("migration_training_data_failed", exc_info=True)

    # Ensure ai_signals table exists (idempotent migration)
    try:
        async with engine.connect() as conn:
            await conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS ai_signals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    has_signal BOOLEAN NOT NULL,
                    direction TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    reasoning TEXT,
                    shadow_mode BOOLEAN NOT NULL DEFAULT false,
                    model_response_raw TEXT,
                    prediction_id UUID,
                    outcome TEXT,
                    scored_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            ))
            await conn.commit()
        logger.info("migration_ai_signals_applied")
    except Exception:
        logger.warning("migration_ai_signals_failed", exc_info=True)

    # Ensure candles table exists (idempotent migration)
    try:
        async with engine.connect() as conn:
            await conn.execute(text(
                """
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
                )
                """
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe "
                "ON candles (symbol, timeframe_sec, timestamp DESC)"
            ))
            await conn.commit()
        logger.info("migration_candles_table_applied")
    except Exception:
        logger.warning("migration_candles_table_failed", exc_info=True)

    # Ensure predictions.has_feature_snapshot column (idempotent migration)
    try:
        async with engine.connect() as conn:
            await conn.execute(text(
                "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feature_snapshot JSONB"
            ))
            await conn.commit()
        logger.info("migration_feature_snapshot_column_applied")
    except Exception:
        logger.warning("migration_feature_snapshot_column_failed", exc_info=True)

    # Ensure predictions has data-quality columns (idempotent migration)
    for col_sql in [
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS candle_count INT",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS data_age_seconds DOUBLE PRECISION",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS data_sufficiency_issues JSONB",
    ]:
        try:
            async with engine.connect() as conn:
                await conn.execute(text(col_sql))
                await conn.commit()
        except Exception:
            logger.warning("migration_data_quality_columns_failed", exc_info=True)

    # Initialize components
    prediction_store = PredictionStore(session_factory)
    training_data_store = TrainingDataStore(session_factory)
    ai_signal_store = AISignalStore(session_factory)
    candle_store = CandleStore(session_factory)
    market_data = MarketDataCollector()
    broker = PocketOptionBroker(config=settings.broker)

    # Wire candle persistence — new candles are buffered and flushed async
    market_data.set_candle_store(candle_store)

    # Background task: flush pending candles to Postgres every 30s
    async def _candle_flush_loop() -> None:
        while True:
            try:
                n = await market_data.flush_pending_candles()
                if n:
                    logger.info("candles_flushed", count=n)
            except Exception:
                logger.warning("candle_flush_error", exc_info=True)
            await asyncio.sleep(30)

    flush_task = asyncio.create_task(_candle_flush_loop())

    # Load ML model if available
    ml_model = TradingModel()
    model_dir = PROJECT_ROOT / "storage" / "models" / "ml"
    if (model_dir / "model.joblib").exists():
        try:
            ml_model.load(model_dir)
            logger.info(
                "ml_model_loaded",
                version=ml_model.metadata.version if ml_model.metadata else "unknown",
            )
        except Exception:
            logger.warning("ml_model_load_failed", exc_info=True)
    else:
        logger.info("ml_model_not_found_at path=%s", str(model_dir))

    # Register market data collector as a message handler
    broker.on_message(market_data.make_message_handler())

    # Trade tracker — sends Win/Tie/Loss buttons when predictions expire
    trade_tracker = TradeTracker(
        prediction_store=prediction_store,
        get_price_fn=market_data.get_latest_price,
        send_message_fn=_make_send_message_fn(
            settings.telegram.bot_token.get_secret_value(),
        ),
    )

    # Connect broker
    logger.info("connecting_broker")
    try:
        await broker.connect()
        logger.info("broker_connected")
    except Exception as e:
        logger.error("broker_connection_failed", error=str(e))
        logger.info("will_retry_in_background")

    # Build Telegram bot
    ai_engine = AIAnalysisEngine(signal_store=ai_signal_store)
    bot = ManualTradingBot(
        bot_token=settings.telegram.bot_token.get_secret_value(),
    )
    app = bot.build(
        broker=broker,
        prediction_store=prediction_store,
        training_data_store=training_data_store,
        ai_signal_store=ai_signal_store,
        ai_engine=ai_engine,
        market_data=market_data,
        ml_model=ml_model,
    )

    # Start the Telegram bot
    logger.info("starting_telegram_bot")
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await bot.set_commands()

    # Start background tasks
    trade_tracker.start()
    reconnect_task = asyncio.create_task(broker_reconnect_loop(broker))

    logger.info("manual_trading_bot_started")

    # Wait for shutdown
    stop_event = asyncio.Event()

    def signal_handler():
        logger.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal_module.SIGINT, signal_module.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    await stop_event.wait()

    # Graceful shutdown
    logger.info("shutting_down")
    flush_task.cancel()
    await asyncio.gather(flush_task, return_exceptions=True)
    await ai_engine.close()
    trade_tracker.stop()
    reconnect_task.cancel()
    await asyncio.gather(reconnect_task, return_exceptions=True)
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    await broker.disconnect()
    if engine is not None:
        await engine.dispose()
    logger.info("manual_trading_bot_stopped")


def _make_send_message_fn(bot_token: str):
    """Return an async function that sends Telegram messages with keyboards."""

    async def _send(
        chat_id: int,
        text: str,
        reply_markup=None,
    ) -> None:
        from telegram import Bot

        bot = Bot(token=bot_token)
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
        )

    return _send


if __name__ == "__main__":
    asyncio.run(main())
