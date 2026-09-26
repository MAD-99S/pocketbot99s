"""Telegram command handlers for manual trading mode."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pandas as pd
from telegram import Update, CallbackQuery, InputFile
from telegram.ext import ContextTypes

# Resolve the assets directory once at import time
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_SIGNAL_IMAGES = {
    "call": _ASSETS_DIR / "buy.jpg",
    "put": _ASSETS_DIR / "sell.jpg",
}

from apps.manual_trading.constants import (
    POPULAR_PAIRS,
    min_candles_for_timeframe,
    MIN_ASSET_PAYOUT_PCT,
    MAX_ASSET_PAYOUT_PCT,
)
from apps.manual_trading.database import PredictionStore, AISignalStore
from apps.manual_trading.keyboards import (
    pair_selection_keyboard,
    duration_selection_keyboard,
    trade_mode_keyboard,
    filter_assets_by_payout,
)
from apps.manual_trading.market_data import MarketDataCollector
from apps.manual_trading.messages import (
    format_signal,
    format_prediction_confirmed,
    format_result_recorded,
    format_stats,
    format_recent,
    format_no_signal,
)
from apps.manual_trading.models import Prediction
from apps.manual_trading.signal_generator import generate_signal
from apps.manual_trading.strategies.mean_reversion import MeanReversionEngine
from apps.manual_trading.constants import COOLDOWN_BARS
from apps.manual_trading.data_sufficiency_gate import DataSufficiencyGate
from infrastructure.features.indicators.technical import TechnicalIndicators
from apps.manual_trading.confidence_calibration import (
    CalibrationStore,
    ConfidenceCalibrator,
)
from apps.manual_trading.regime_classifier import RegimeReading

logger = logging.getLogger(__name__)

# Maximum time to wait for candle data (seconds)
CANDLE_WAIT_TIMEOUT = 15
CANDLE_POLL_INTERVAL = 0.5


async def _get_filtered_pairs(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    """Return POPULAR_PAIRS filtered by payout percentage.

    Falls back to all POPULAR_PAIRS if broker or payout data is unavailable.
    """
    broker = context.bot_data.get("broker")
    if broker is None:
        return list(POPULAR_PAIRS)
    try:
        payouts = broker.get_payouts()
    except Exception:
        logger.warning("payout_fetch_failed")
        return list(POPULAR_PAIRS)
    if not payouts:
        return list(POPULAR_PAIRS)
    return filter_assets_by_payout(POPULAR_PAIRS, payouts)


async def _has_pending_result(context: ContextTypes.DEFAULT_TYPE, telegram_id: int) -> bool:
    """Return True if the user has an unresolved trade result waiting.

    Checks the database for predictions where we sent a result-request
    but the user hasn't responded yet.
    """
    store: PredictionStore = context.bot_data.get("prediction_store")
    if store is None:
        return False
    return await store.has_pending_result(telegram_id)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if await _has_pending_result(context, update.effective_user.id):
        await update.message.reply_text(
            "\u26a0\ufe0f Please report the result of your last trade first.\n"
            "Tap Win, Tie, or Loss below that message before continuing."
        )
        return

    await update.message.reply_text(
        "\U0001f916 Manual Trading Bot\n\n"
        "Get AI-powered predictions for Pocket Option pairs.\n"
        "No account connection needed.\n\n"
        "Commands:\n"
        "/predict - Get a prediction\n"
        "/stats - Your trading stats\n"
        "/recent - Recent predictions\n"
        "/help - Show this message"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    if await _has_pending_result(context, update.effective_user.id):
        await update.message.reply_text(
            "\u26a0\ufe0f Please report the result of your last trade first.\n"
            "Tap Win, Tie, or Loss below that message before continuing."
        )
        return

    await update.message.reply_text(
        "How it works:\n\n"
        "1. /predict - Choose a pair\n"
        "2. Pick a duration (1min, 5min, or 15min)\n"
        "3. Get a signal with direction and reasoning\n"
        "4. When the trade expires, report the result (Win/Tie/Loss)\n\n"
        "Commands:\n"
        "/predict - Get a prediction\n"
        "/stats - Your trading stats\n"
        "/recent - Recent predictions\n"
        "/help - Show this message"
    )


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /predict command — show trade mode selection."""
    if await _has_pending_result(context, update.effective_user.id):
        await update.message.reply_text(
            "\u26a0\ufe0f Please report the result of your last trade first.\n"
            "Tap Win, Tie, or Loss below that message before continuing."
        )
        return

    await update.message.reply_text(
        "Choose a trading mode:",
        reply_markup=trade_mode_keyboard(),
    )


async def callback_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle trade mode selection: quick or ai."""
    query: CallbackQuery = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("mode:"):
        return

    mode = data.split(":")[1]
    pairs = await _get_filtered_pairs(context)

    if not pairs:
        await query.edit_message_text(
            "\u26a0\ufe0f No assets available with the current payout filter.\n"
            f"Payout range: {MIN_ASSET_PAYOUT_PCT:.0f}% \u2013 {MAX_ASSET_PAYOUT_PCT:.0f}%.\n\n"
            "Adjust MIN_ASSET_PAYOUT_PCT / MAX_ASSET_PAYOUT_PCT in constants.py"
            " to widen the range."
        )
        return

    if mode == "quick":
        await query.edit_message_text(
            "Quick Trade - Rule-based signals\n\nChoose a trading pair:",
            reply_markup=pair_selection_keyboard(pairs),
        )
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command — show win rate and performance."""
    if await _has_pending_result(context, update.effective_user.id):
        await update.message.reply_text(
            "\u26a0\ufe0f Please report the result of your last trade first.\n"
            "Tap Win, Tie, or Loss below that message before continuing."
        )
        return

    store: PredictionStore = context.bot_data["prediction_store"]
    telegram_id = update.effective_user.id

    try:
        stats = await store.get_user_stats(telegram_id)
        await update.message.reply_text(format_stats(stats))
    except Exception:
        logger.exception("stats_error")
        await update.message.reply_text("Error loading stats. Please try again.")


async def cmd_recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /recent command — show recent predictions."""
    if await _has_pending_result(context, update.effective_user.id):
        await update.message.reply_text(
            "\u26a0\ufe0f Please report the result of your last trade first.\n"
            "Tap Win, Tie, or Loss below that message before continuing."
        )
        return

    store: PredictionStore = context.bot_data["prediction_store"]
    telegram_id = update.effective_user.id

    try:
        recent = await store.get_recent(telegram_id, limit=10)
        await update.message.reply_text(format_recent(recent))
    except Exception:
        logger.exception("recent_error")
        await update.message.reply_text("Error loading recent predictions.")


async def callback_pair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pair selection callback — show duration options."""
    query: CallbackQuery = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("pair:"):
        return

    symbol = data.split(":", 1)[1]
    display = symbol.replace("_otc", " (OTC)").replace("_", "/")

    await query.edit_message_text(
        f"Selected: {display}\n\nChoose duration:",
        reply_markup=duration_selection_keyboard(symbol),
    )


async def callback_duration(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle duration selection callback — generate signal and save prediction."""
    query: CallbackQuery = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("dur:"):
        return

    parts = data.split(":")
    symbol = parts[1]
    timeframe_sec = int(parts[2])
    telegram_id = update.effective_user.id

    await _handle_quick_duration(query, context, symbol, timeframe_sec, telegram_id)


    """Legacy AI Analysis mode — now redirects to Quick Trade."""
    await _handle_quick_duration(query, context, symbol, timeframe_sec, telegram_id)

async def _handle_quick_duration(
    query: CallbackQuery,
    context: ContextTypes.DEFAULT_TYPE,
    symbol: str,
    timeframe_sec: int,
    telegram_id: int,
) -> None:
    """Handle Quick Trade duration selection — rule-based signal (original flow)."""
    await query.edit_message_text("\u23f3 Analyzing market conditions...")

    try:
        broker = context.bot_data["broker"]
        collector: MarketDataCollector = context.bot_data["market_data"]

        # Check broker connection
        if not await broker.is_connected():
            logger.warning("predict_broker_disconnected symbol=%s", symbol)
            await query.edit_message_text(
                "Broker not connected. Please wait a moment and try again.\n"
                "The bot is attempting to reconnect in the background."
            )
            return

        # Request candle history from broker
        await collector.request_candles(broker, symbol, timeframe_sec)

        # Wait for candle data to arrive
        df = await _wait_for_candles(collector, symbol, timeframe_sec, CANDLE_WAIT_TIMEOUT)

        min_candles = min_candles_for_timeframe(timeframe_sec)
        if df is None or len(df) < min_candles:
            logger.warning(
                "insufficient_candle_data symbol=%s count=%d needed=%d timeframe=%d known=%s",
                symbol,
                len(df) if df is not None else 0,
                min_candles,
                timeframe_sec,
                list(collector.get_all_prices().keys())[:5],
            )
            await query.edit_message_text(
                f"Insufficient market data ({len(df) if df is not None else 0}/{min_candles} candles).\n"
                f"The pair may not be available right now.\n"
                f"Try again in a moment or pick a different pair."
            )
            return

        # Data sufficiency gate — check quality, not just count
        gate = DataSufficiencyGate()
        tick_history = await collector.get_tick_history(symbol)
        quality_report = gate.evaluate(df, timeframe_sec, tick_counts=tick_history)
        if not quality_report.is_sufficient:
            symbol_display = symbol.replace("_otc", " (OTC)").replace("_", "/")
            issue_str = quality_report.summary()
            logger.warning(
                "data_sufficiency_failed symbol=%s report=%s",
                symbol, issue_str,
            )
            await query.edit_message_text(
                f"Not enough reliable data yet for {symbol_display}.\n"
                f"{issue_str}\n"
                f"Try again in a moment or pick a different pair."
            )
            return

        # Compute indicators
        ti = TechnicalIndicators()
        df_with_indicators = ti.compute(df)

        # Regime classification
        from apps.manual_trading.regime_classifier import RegimeClassifier
        regime = RegimeClassifier().classify(df_with_indicators)

        # Multi-timeframe confirmation (skip for 1-min — no higher TF available)
        htf_note: str | None = None
        if timeframe_sec >= 60:
            from apps.manual_trading.htf_confirmation import HTFConfirmation
            htf = HTFConfirmation()
            htf_result = htf.confirm(df_with_indicators, timeframe_sec)
            if htf_result is not None:
                if htf_result.confidence_penalty:
                    regime = regime.__class__(
                        regime=regime.regime,
                        adx=regime.adx,
                        bb_width=regime.bb_width,
                        bb_width_percentile=regime.bb_width_percentile,
                        trend_weight=regime.trend_weight * (1 - htf_result.confidence_penalty),
                        reversion_weight=regime.reversion_weight * (1 - htf_result.confidence_penalty),
                    )
                if htf_result.note:
                    htf_note = htf_result.note

        # Cooldown check — block signals for the same pair within COOLDOWN_BARS
        cooldown_state: dict[str, int] = context.user_data.setdefault(
            "signal_cooldown", {}
        )
        current_bar = len(df_with_indicators) - 1
        last_bar = cooldown_state.get(symbol)
        if last_bar is not None and (current_bar - last_bar) < COOLDOWN_BARS:
            remaining = COOLDOWN_BARS - (current_bar - last_bar)
            symbol_display = symbol.replace("_otc", " (OTC)").replace("_", "/")
            await query.edit_message_text(
                format_no_signal(
                    symbol_display,
                    f"Cooldown — wait {remaining} more bar{'s' if remaining != 1 else ''} "
                    f"before next signal for this pair",
                )
            )
            return

        # Generate signal — mean-reversion engine for 5-min OTC pairs
        is_otc_5min = symbol.endswith("_otc") and timeframe_sec == 300
        if is_otc_5min:
            engine = MeanReversionEngine()
            signal = engine.generate_signal(df)
        else:
            signal = generate_signal(df_with_indicators, regime=regime)

        # Gate: only proceed if signal is valid
        if not signal.has_signal:
            symbol_display = symbol.replace("_otc", " (OTC)").replace("_", "/")
            await query.edit_message_text(
                format_no_signal(symbol_display, signal.reasoning[0])
            )
            return

        # Get current price
        price = await collector.get_latest_price(symbol)
        if price is None:
            entry_price_float = float(df.iloc[-1]["close"])
        else:
            entry_price_float = float(price)

        # Format signal text and send as a photo with caption
        signal_msg = format_signal(symbol, signal, entry_price_float)
        image_path = _SIGNAL_IMAGES.get(signal.direction)
        photo_sent = False
        if image_path and image_path.exists():
            try:
                with open(image_path, "rb") as photo_file:
                    await query.message.delete()
                    await context.bot.send_photo(
                        chat_id=telegram_id,
                        photo=InputFile(photo_file),
                        caption=signal_msg,
                    )
                photo_sent = True
            except Exception:
                logger.warning("signal_photo_send_failed direction=%s", signal.direction, exc_info=True)
        if not photo_sent:
            await query.edit_message_text(signal_msg)

        # Save prediction to database
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=timeframe_sec)
        from uuid import uuid4

        # Data quality metadata for calibration auditing
        last_ts = df["timestamp"].iloc[-1]
        now_ts = pd.Timestamp.now(timezone.utc)
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        data_age = (now_ts - last_ts).total_seconds()
        candle_count = len(df)
        issues_list = [i.value for i in quality_report.issues] if quality_report.issues else None

        # Confidence calibration — remap raw voteratio through reliability curve
        from apps.manual_trading.confidence_calibration import (
            CalibrationStore,
            ConfidenceCalibrator,
        )
        store: PredictionStore = context.bot_data["prediction_store"]
        if signal.has_signal:
            calibrator = ConfidenceCalibrator(
                await CalibrationStore(store).build_curve()
            )
            signal = signal.model_copy(
                update={"confidence": calibrator.calibrate(signal.confidence)}
            )

        prediction = Prediction(
            id=uuid4(),
            telegram_id=telegram_id,
            symbol=symbol,
            timeframe_sec=timeframe_sec,
            direction=signal.direction,
            confidence=signal.confidence,
            reasoning="\n".join(signal.reasoning),
            indicators=signal.indicators,
            entry_price=Decimal(str(entry_price_float)),
            entry_time=now,
            expiry_time=expiry,
            result=None,
            candle_count=candle_count,
            data_age_seconds=data_age,
            data_sufficiency_issues=issues_list,
            feature_snapshot=signal.indicators,
        )

        store: PredictionStore = context.bot_data["prediction_store"]
        await store.insert(prediction)

        # Update cooldown state — prevent rapid re-signals for same pair
        cooldown_state[symbol] = current_bar

        # Send confirmation
        confirmation = format_prediction_confirmed(prediction)
        await context.bot.send_message(chat_id=telegram_id, text=confirmation)

    except Exception:
        logger.exception("predict_error")
        await query.edit_message_text(
            "Error generating prediction. Please try again."
        )


async def _store_training_data(
    context: ContextTypes.DEFAULT_TYPE,
    symbol: str,
    timeframe_sec: int,
    direction: str,
    entry_price: float,
    features: dict,
    win_probability: float,
) -> None:
    """Store training data for future model improvement."""
    try:
        from apps.manual_trading.database import TrainingDataStore

        store: TrainingDataStore = context.bot_data.get("training_data_store")
        if store is None:
            return

        await store.insert(
            symbol=symbol,
            timeframe_sec=timeframe_sec,
            direction=direction,
            entry_price=entry_price,
            features=features,
            win_probability=win_probability,
        )
    except Exception:
        logger.debug("training_data_store_failed", exc_info=True)


async def _wait_for_candles(
    collector: MarketDataCollector,
    symbol: str,
    timeframe_sec: int,
    timeout: float,
) -> pd.DataFrame | None:
    """Wait for candle data to arrive, polling periodically."""
    min_needed = min_candles_for_timeframe(timeframe_sec)
    deadline = asyncio.get_event_loop().time() + timeout
    attempts = 0
    while asyncio.get_event_loop().time() < deadline:
        df = await collector.get_candles(symbol)
        if df is not None and len(df) >= min_needed:
            # Check staleness of the most recent candle — the slow
            # ``loadHistoryPeriod`` batch is often minutes old, which trips
            # the sufficiency gate's stale_data check. Keep polling until
            # we have fresh candles (last candle within 2× timeframe of now).
            if len(df) > 0:
                raw_ts = df["timestamp"].iloc[-1]
                if hasattr(raw_ts, "tzinfo"):
                    last_ts = pd.Timestamp(raw_ts)
                else:
                    last_ts = pd.to_datetime(raw_ts, unit="s", utc=True)
                now_ts = pd.Timestamp.now(timezone.utc)
                age_sec = (now_ts - last_ts).total_seconds()
                if age_sec <= timeframe_sec * 2.0:
                    logger.info(
                        "candles_ready symbol=%s count=%d attempts=%d age=%.0fs",
                        symbol, len(df), attempts, age_sec,
                    )
                    return df
        attempts += 1
        await asyncio.sleep(CANDLE_POLL_INTERVAL)

    # Return whatever we have, even if less than 30 candles
    df = await collector.get_candles(symbol)
    logger.warning(
        "candles_wait_timeout symbol=%s got=%d attempts=%d known=%s",
        symbol,
        len(df) if df is not None else 0,
        attempts,
        list(collector.get_all_prices().keys())[:10],
    )
    return df


async def callback_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Win / Tie / Loss button press after trade expiry."""
    query: CallbackQuery = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("result:"):
        return

    parts = data.split(":")
    if len(parts) != 3:
        return

    prediction_id = parts[1]
    result = parts[2]

    if result not in ("win", "tie", "loss"):
        return

    telegram_id = update.effective_user.id

    store: PredictionStore = context.bot_data["prediction_store"]

    try:
        from uuid import UUID

        await store.resolve(
            prediction_id=UUID(prediction_id),
            exit_price=Decimal("0"),
            result=result,
        )

        # Confirm to user
        await query.edit_message_text(text=format_result_recorded(result))

        logger.info(
            "result_submitted prediction_id=%s result=%s telegram_id=%s",
            prediction_id,
            result,
            telegram_id,
        )
    except Exception:
        logger.exception("result_callback_error")
        await query.edit_message_text(
            "Error recording result. Please try again."
        )
