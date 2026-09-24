"""Market data collector — builds candles from live broker price ticks.

Pocket Option does NOT reliably send historical candle data after
``changeSymbol``. Instead we collect live ``updateStream`` price ticks
and aggregate them into OHLC candles in real-time. The collector also
handles ``loadHistoryPeriod`` events if the server ever sends them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Callable

import pandas as pd

if TYPE_CHECKING:
    from apps.manual_trading.candle_store import CandleStore

logger = logging.getLogger(__name__)

# Maximum candles to keep per symbol
_MAX_CANDLES = 200


class _CandleBuilder:
    """Accumulates price ticks into OHLC candles for a single symbol."""

    def __init__(self, timeframe_sec: int) -> None:
        self.timeframe_sec = timeframe_sec
        self.candles: list[dict] = []
        # State for the candle currently being built
        self._current_open: float | None = None
        self._current_high: float | None = None
        self._current_low: float | None = None
        self._current_close: float | None = None
        self._current_start: float = 0.0  # unix seconds
        self._current_tick_count: int = 0  # ticks accumulated in current candle
        # Optional callback invoked with each finalized candle batch
        self._on_finalise = None  # callable[[list[dict]], None] | None
        # Optional callback that pushes finalized candles to the collector's pending buffer
        self._on_push_to_flush: callable[[list[dict]], None] | None = None

    def set_on_finalise(self, callback) -> None:
        """Register a callback invoked for each finalized candle batch."""
        self._on_finalise = callback

    def add_tick(self, price: float, tick_time: float) -> None:
        """Ingest a single price tick and update the current candle."""
        if self._current_open is None:
            # First tick — start a new candle window
            self._current_start = tick_time
            self._current_open = price
            self._current_high = price
            self._current_low = price
            self._current_close = price
            self._current_tick_count = 1
            return

        # Check if this tick belongs to a new candle window
        elapsed = tick_time - self._current_start
        if elapsed >= self.timeframe_sec:
            # Close the previous candle
            completed = self._finalise_candle()
            if completed and self._on_finalise is not None:
                self._on_finalise(completed)
            # Start a new window with this tick
            self._current_start = tick_time
            self._current_open = price
            self._current_high = price
            self._current_low = price
            self._current_close = price
            self._current_tick_count = 1
        else:
            # Update current candle
            self._current_high = max(self._current_high, price)  # type: ignore[operator]
            self._current_low = min(self._current_low, price)  # type: ignore[operator]
            self._current_close = price
            self._current_tick_count += 1

    def _finalise_candle(self) -> list[dict] | None:
        """Finalize the current candle. Returns the candle dict, or None if none open."""
        if self._current_open is None:
            return None
        candle = {
            "timestamp": self._current_start,
            "open": self._current_open,
            "high": self._current_high,
            "low": self._current_low,
            "close": self._current_close,
            "volume": 0,
            "ticks_in_candle": self._current_tick_count,
        }
        self.candles.append(candle)
        if len(self.candles) > _MAX_CANDLES:
            self.candles = self.candles[-_MAX_CANDLES:]
        if self._on_push_to_flush is not None:
            try:
                self._on_push_to_flush([candle])
            except Exception:
                pass
        return [candle]

    def flush_pending(self, store: CandleStore) -> int:
        """Persist any candles completed since the last flush to Postgres.

        Must be called from an async context (the caller's event loop).
        Returns the number of candles persisted.
        """
        # Drains _pending_candle_flush; called via MarketDataCollector.flush_pending_candles
        return 0  # overridden by MarketDataCollector's wrapper

    def snapshot(self) -> list[dict]:
        """Return a copy of all candles including the in-progress one."""
        result = list(self.candles)
        if self._current_open is not None:
            result.append({
                "timestamp": self._current_start,
                "open": self._current_open,
                "high": self._current_high,
                "low": self._current_low,
                "close": self._current_close,
                "volume": 0,
            })
        return result


class MarketDataCollector:
    """Captures price ticks from broker messages and builds candles."""

    def __init__(self) -> None:
        self._candles: dict[str, list[dict]] = {}
        self._builders: dict[str, _CandleBuilder] = {}
        self._candle_timeframes: dict[str, int] = {}
        self._latest_prices: dict[str, Decimal] = {}
        self._subscribed_symbols: set[str] = set()
        self._lock = asyncio.Lock()
        # DIAGNOSTIC: count ticks received per symbol
        self._tick_counts: dict[str, int] = {}
        # Optional candle persistence
        self._candle_store: CandleStore | None = None
        # Per-symbol buffer of recently-finalised candles awaiting async flush
        self._pending_candle_flush: dict[str, list[dict]] = {}

    def set_candle_store(self, store: CandleStore | None) -> None:
        """Attach a CandleStore for persisting completed candles to Postgres.

        Each _CandleBuilder's _on_push_to_flush callback is wired to push
        finalized candles into _pending_candle_flush (a plain list, safe
        for synchronous calls).  Call flush_pending_candles() from an
        async context to drain those buffers through the store.
        """
        self._candle_store = store

    async def get_tick_history(self, symbol: str) -> list[int]:
        """Return per-candle tick counts for the recent window (most-recent last).

        Used by DataSufficiencyGate for the liquidity check. If the builder
        doesn't track ticks per candle yet, returns an empty list (gate
        skips the check).
        """
        builder = self._builders.get(symbol)
        if builder is None:
            return []
        ticks: list[int] = []
        for c in reversed(builder.candles):
            ticks.append(c.get("ticks_in_candle", 0))
        return ticks

    async def flush_pending_candles(self, symbol: str | None = None) -> int:
        """Persist buffered candles to the attached CandleStore.

        If *symbol* is None, flushes all symbols.  Returns the total number
        of candles persisted.
        """
        if self._candle_store is None:
            return 0
        flushed = 0
        targets = [symbol] if symbol else list(self._pending_candle_flush.keys())
        for sym in targets:
            batch = self._pending_candle_flush.pop(sym, [])
            if not batch:
                continue
            timeframe = self._candle_timeframes.get(sym, 60)
            await self._candle_store.insert_completed(sym, timeframe, batch)
            flushed += len(batch)
        return flushed

    # ------------------------------------------------------------------
    # Message handler factory
    # ------------------------------------------------------------------

    def make_message_handler(self):
        """Return an async callback suitable for broker.on_message()."""
        async def _handler(msg) -> None:
            await self._process(msg)
        return _handler

    # ------------------------------------------------------------------
    # Raw message parsing
    # ------------------------------------------------------------------

    async def _process(self, msg) -> None:
        """Parse a raw broker message and extract price / candle data."""
        if isinstance(msg, bytes):
            return
        if not isinstance(msg, str):
            return

        # DIAGNOSTIC: log every raw message the collector receives
        logger.warning("collector_raw_msg type=%s preview=%s", type(msg).__name__, msg[:200])

        # Socket.IO BINARY_EVENT: 451-[...]
        if msg.startswith("451-["):
            try:
                json_part = msg.split("-", 1)[1]
                data = json.loads(json_part)
                await self._handle_event(data)
            except (json.JSONDecodeError, IndexError):
                pass
            return

        # Socket.IO EVENT: 42[...]
        if msg.startswith("42"):
            try:
                data = json.loads(msg[2:])
                await self._handle_event(data)
            except (json.JSONDecodeError, IndexError):
                pass
            return

    async def _handle_event(self, data: list) -> None:
        """Dispatch a parsed Socket.IO event."""
        if not data or not isinstance(data, list):
            return

        event_name = data[0]
        event_data = data[1] if len(data) > 1 else None

        # DIAGNOSTIC: log every event the collector sees
        logger.warning(
            "collector_event name=%s data_type=%s data_preview=%s",
            event_name,
            type(event_data).__name__,
            str(event_data)[:150] if event_data is not None else "None",
        )

        if event_name in ("loadHistoryPeriod", "loadHistoryPeriodFast", "candles"):
            await self._handle_candle_history(event_data)
        elif event_name == "updateStream":
            await self._handle_stream_update(event_data)
        elif event_name == "price":
            await self._handle_price_update(event_data)
        elif event_name == "updateCharts":
            # NOTE: updateCharts only contains chart display settings
            # (symbol, period, chartType) — NOT candle data.
            pass

    # ------------------------------------------------------------------
    # Candle history (rare — only if server sends it)
    # ------------------------------------------------------------------

    async def _handle_candle_history(self, event_data) -> None:
        """Parse candle data from loadHistoryPeriod / loadHistoryPeriodFast.

        Server response can be:
        - Dict: ``{"asset": "...", "candles": [[ts,o,c,h,l], ...]}``
        - List: ``["asset", [[ts,o,c,h,l], ...]]``

        Each candle array: ``[timestamp, open, close, high, low]``
        (note: close before high/low — NOT standard OHLC order).
        """
        if event_data is None:
            return

        try:
            asset = ""
            candles: list = []

            if isinstance(event_data, dict):
                asset = event_data.get("asset", "")
                candles = event_data.get("candles", event_data.get("data", []))
            elif isinstance(event_data, list) and len(event_data) >= 2:
                asset = str(event_data[0]) if event_data[0] else ""
                candles = event_data[1] if isinstance(event_data[1], list) else []

            if not asset or not candles:
                return

            parsed = []
            for c in candles:
                if isinstance(c, list) and len(c) >= 4:
                    # Raw array format: [timestamp, open, close, high, low]
                    parsed.append({
                        "timestamp": float(c[0]),
                        "open": float(c[1]),
                        "high": float(c[3]),
                        "low": float(c[4]) if len(c) > 4 else float(c[1]),
                        "close": float(c[2]),
                        "volume": 0,
                    })
                elif isinstance(c, dict):
                    parsed.append({
                        "timestamp": c.get("t", c.get("time", c.get("timestamp", 0))),
                        "open": float(c.get("o", c.get("open", 0))),
                        "high": float(c.get("h", c.get("high", 0))),
                        "low": float(c.get("l", c.get("low", 0))),
                        "close": float(c.get("c", c.get("close", 0))),
                        "volume": float(c.get("v", c.get("volume", 0))),
                    })

            if parsed:
                async with self._lock:
                    if asset not in self._candles:
                        self._candles[asset] = []
                    self._candles[asset].extend(parsed)
                    self._candles[asset] = self._candles[asset][-_MAX_CANDLES:]

                # Extract latest price
                last = parsed[-1]
                self._latest_prices[asset] = Decimal(str(last["close"]))

                logger.warning(
                    "candles_received_from_server symbol=%s count=%d total=%d",
                    asset, len(parsed), len(self._candles.get(asset, [])),
                )
        except (ValueError, TypeError, InvalidOperation):
            logger.debug("candle_parse_error", exc_info=True)

    # ------------------------------------------------------------------
    # Live price tick → candle builder
    # ------------------------------------------------------------------

    async def _handle_stream_update(self, event_data) -> None:
        """Parse candle data from ``updateStream`` events.

        Server payload:
        ``{"asset": "...", "time": <unix>, "data": [[ts,o,c,h,l],...], "history": <bool>}``

        Each element of ``data`` is a candle array:
        ``[timestamp, open, close, high, low]`` (note: close before high/low).
        """
        if not isinstance(event_data, dict):
            return

        asset = event_data.get("asset", "")
        if not asset:
            return

        candle_data = event_data.get("data")

        # --- Full candle array from updateStream ---
        if isinstance(candle_data, list) and candle_data:
            parsed = []
            for c in candle_data:
                if isinstance(c, list) and len(c) >= 4:
                    parsed.append({
                        "timestamp": float(c[0]),
                        "open": float(c[1]),
                        "high": float(c[3]),   # index 3 = high
                        "low": float(c[4]) if len(c) > 4 else float(c[1]),
                        "close": float(c[2]),   # index 2 = close
                        "volume": 0,
                    })

            if parsed:
                async with self._lock:
                    if asset not in self._candles:
                        self._candles[asset] = []
                    self._candles[asset].extend(parsed)
                    self._candles[asset] = self._candles[asset][-_MAX_CANDLES:]

                last = parsed[-1]
                self._latest_prices[asset] = Decimal(str(last["close"]))

                logger.warning(
                    "candles_from_stream symbol=%s count=%d total=%d",
                    asset, len(parsed), len(self._candles.get(asset, [])),
                )
                return

        # --- Fallback: single price tick (legacy format) ---
        price_raw = event_data.get("price") or event_data.get("close")
        if price_raw is None:
            return

        try:
            price = float(price_raw)
            self._latest_prices[asset] = Decimal(str(price))
        except (InvalidOperation, TypeError, ValueError):
            return

        tick_time = _time.time()
        async with self._lock:
            self._tick_counts[asset] = self._tick_counts.get(asset, 0) + 1
            if self._tick_counts[asset] % 10 == 1:
                logger.warning(
                    "tick_received symbol=%s count=%d price=%s",
                    asset, self._tick_counts[asset], price,
                )
            builder = self._builders.get(asset)
            if builder is not None:
                builder.add_tick(price, tick_time)
            else:
                logger.warning(
                    "tick_no_builder symbol=%s price=%s builders=%s",
                    asset, price, list(self._builders.keys()),
                )

    async def _handle_price_update(self, event_data) -> None:
        """Parse price updates from price events."""
        if not isinstance(event_data, dict):
            return
        asset = event_data.get("asset", "")
        price = event_data.get("price")
        if asset and price is not None:
            try:
                self._latest_prices[asset] = Decimal(str(price))
            except (InvalidOperation, TypeError):
                pass

    async def _ingest_candles_list(self, asset: str, candles: list) -> None:
        """Ingest a list of candle dicts or arrays into storage."""
        parsed = []
        for c in candles:
            if isinstance(c, dict):
                parsed.append({
                    "timestamp": c.get("t", c.get("time", c.get("timestamp", 0))),
                    "open": float(c.get("o", c.get("open", 0))),
                    "high": float(c.get("h", c.get("high", 0))),
                    "low": float(c.get("l", c.get("low", 0))),
                    "close": float(c.get("c", c.get("close", 0))),
                    "volume": float(c.get("v", c.get("volume", 0))),
                })
            elif isinstance(c, (list, tuple)) and len(c) >= 4:
                # Raw array format: [timestamp, open, close, high, low]
                parsed.append({
                    "timestamp": float(c[0]),
                    "open": float(c[1]),
                    "high": float(c[3]),
                    "low": float(c[4]) if len(c) > 4 else float(c[1]),
                    "close": float(c[2]),
                    "volume": float(c[5]) if len(c) > 5 else 0,
                })

        if parsed:
            async with self._lock:
                if asset not in self._candles:
                    self._candles[asset] = []
                self._candles[asset].extend(parsed)
                self._candles[asset] = self._candles[asset][-_MAX_CANDLES:]

            # Also extract latest price
            if parsed:
                last = parsed[-1]
                try:
                    self._latest_prices[asset] = Decimal(str(last["close"]))
                except (InvalidOperation, TypeError):
                    pass

            logger.warning(
                "candles_ingested_from_charts symbol=%s count=%d total=%d",
                asset, len(parsed), len(self._candles.get(asset, [])),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def request_candles(self, broker, symbol: str, timeframe_sec: int = 60) -> None:
        """Subscribe to a symbol's price stream and build candles from ticks.

        Sends ``changeSymbol`` to the broker so it starts streaming prices
        for *symbol*, then initialises a candle builder that aggregates
        incoming ticks into OHLC candles.
        """
        is_conn = await broker.is_connected()
        logger.warning(
            "request_candles symbol=%s timeframe=%d broker_connected=%s",
            symbol, timeframe_sec, is_conn,
        )

        if not is_conn:
            logger.warning("broker_not_connected_cannot_request_candles symbol=%s", symbol)
            return

        # Switch active symbol and start streaming
        from domain.value_objects.symbol import Symbol as SymbolVO
        await broker.subscribe_candles(SymbolVO(code=symbol), timeframe_sec)

        # Initialise / reinitialise candle builder
        async with self._lock:
            builder = _CandleBuilder(timeframe_sec)
            # Wire each new builder to push finalized candles into the pending flush buffer
            setattr(
                builder,
                "_on_push_to_flush",
                lambda batch, sym=symbol: self._pending_candle_flush.setdefault(sym, []).extend(batch),
            )
            self._builders[symbol] = builder
            self._candle_timeframes[symbol] = timeframe_sec
            self._tick_counts[symbol] = 0

        self._subscribed_symbols.add(symbol)

        logger.warning(
            "request_candles_done symbol=%s builder_created=True subscribed=%s",
            symbol, list(self._subscribed_symbols),
        )

    async def get_candles(self, symbol: str) -> pd.DataFrame | None:
        """Get accumulated candles for a symbol as a DataFrame."""
        async with self._lock:
            # Prefer server-provided candles if available
            server_candles = self._candles.get(symbol, [])
            builder = self._builders.get(symbol)

        # Merge server candles + builder candles
        all_candles = list(server_candles)
        if builder is not None:
            all_candles.extend(builder.snapshot())

        if not all_candles:
            return None

        df = pd.DataFrame(all_candles)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

        # De-duplicate by timestamp (keep last)
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp", "open", "high", "low", "close"], keep="last")
            df = df.reset_index(drop=True)

        return df

    async def get_latest_price(self, symbol: str) -> Decimal | None:
        """Get the latest known price for a symbol."""
        return self._latest_prices.get(symbol)

    def get_all_prices(self) -> dict[str, Decimal]:
        """Get all latest prices."""
        return dict(self._latest_prices)
