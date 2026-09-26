"""
Data sufficiency gate — decides whether accumulated candle data is trustworthy
enough to generate a trading signal from.

Replaces the flat len(df) < min_candles check currently in handlers.py
(see MIN_CANDLES_BY_TIMEFRAME in constants.py) with a richer set of checks —
candle count, staleness, timestamp gaps, dead/flat candles, and tick liquidity —
and returns a structured verdict instead of a bool, so the caller can tell the
user why it declined rather than just "insufficient data".

Integration point: in handlers.py, after df = await _wait_for_candles(...)::

    gate = DataSufficiencyGate()
    report = gate.evaluate(df, timeframe_sec, tick_counts=collector.get_tick_history(symbol))
    if not report.is_sufficient:
        await query.edit_message_text(f"Not enough reliable data yet: {report.summary()}")
        return

Note: the tick-liquidity check needs per-candle tick counts, which MarketDataCollector
doesn't currently track (only a cumulative per-symbol counter). Add a ticks_in_candle
counter to _CandleBuilder and include it in _finalise_candle's output dict to wire this
up; until then, omit tick_counts and that one check is skipped automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class DataQualityIssue(str, Enum):
    INSUFFICIENT_CANDLES = "insufficient_candles"
    STALE_DATA = "stale_data"
    TIMESTAMP_GAPS = "timestamp_gaps"
    LOW_TICK_LIQUIDITY = "low_tick_liquidity"
    FLAT_CANDLES = "flat_candles"


@dataclass(frozen=True)
class DataQualityReport:
    is_sufficient: bool
    candle_count: int
    required_count: int
    issues: list[DataQualityIssue] = field(default_factory=list)
    detail: dict[str, float | int | str] = field(default_factory=dict)

    def summary(self) -> str:
        if self.is_sufficient:
            return f"OK ({self.candle_count}/{self.required_count} candles)"
        reasons = ", ".join(issue.value for issue in self.issues)
        return f"{self.candle_count}/{self.required_count} candles -- {reasons}"


# Minimum candle count derived from the slowest indicator's warm-up need,
# not shrunk per-timeframe the way MIN_CANDLES_BY_TIMEFRAME currently is.
# MACD(12,26,9) needs ~35 bars before its slow EMA is numerically stable;
# ADX(14)/ATR(14) need ~28-30 for Wilder smoothing to settle.
# This is a floor -- raise it per-symbol if backtesting shows indicators
# still noisy at this size.
MIN_CANDLES_FOR_STABLE_INDICATORS = 40

# Max age of the last candle, as a multiple of the timeframe, before the
# stream is considered dead/stale (e.g. broker stopped pushing ticks).
MAX_STALENESS_MULTIPLE = 2.0

# Max gap between consecutive candle timestamps, as a multiple of the
# timeframe, before it's treated as a dropped/missing candle.
MAX_GAP_MULTIPLE = 1.5

# A recent window is "flat" (dead/illiquid feed, not a real quiet market)
# if this fraction of candles have high-low range below FLAT_RANGE_PCT of price.
FLAT_RANGE_PCT = 0.00005
FLAT_CANDLE_FRACTION = 0.8
FLAT_WINDOW_MIN = 10
FLAT_WINDOW_MAX = 30

# Minimum average ticks per candle over the recent window, below which the
# OHLC values likely don't reflect real intra-candle price movement.
MIN_TICKS_PER_CANDLE = 3.0
TICK_LOOKBACK_CANDLES = 20


class DataSufficiencyGate:
    """Decide whether a candle DataFrame is trustworthy enough to trade on."""

    def __init__(
        self,
        min_candles: int = MIN_CANDLES_FOR_STABLE_INDICATORS,
        max_staleness_multiple: float = MAX_STALENESS_MULTIPLE,
        max_gap_multiple: float = MAX_GAP_MULTIPLE,
        min_ticks_per_candle: float = MIN_TICKS_PER_CANDLE,
    ) -> None:
        self._min_candles = min_candles
        self._max_staleness_multiple = max_staleness_multiple
        self._max_gap_multiple = max_gap_multiple
        self._min_ticks_per_candle = min_ticks_per_candle

    def evaluate(
        self,
        df: pd.DataFrame | None,
        timeframe_sec: int,
        now: pd.Timestamp | None = None,
        tick_counts: list[int] | None = None,
    ) -> DataQualityReport:
        """Run all checks and return a structured verdict.

        Args:
            df: candle DataFrame with a ``timestamp`` column (UTC) plus
                open/high/low/close, as returned by `MarketDataCollector.get_candles`.
            timeframe_sec: candle timeframe in seconds.
            now: current time (UTC); defaults to `pd.Timestamp.utcnow()`.
                Injectable for tests.
            tick_counts: optional per-candle tick counts for the recent window
                (most-recent last). If omitted, the liquidity check is skipped
                rather than failed.
        """
        now = now if now is not None else pd.Timestamp.utcnow()
        issues: list[DataQualityIssue] = []
        detail: dict[str, float | int | str] = {}

        candle_count = 0 if df is None else len(df)
        detail["candle_count"] = candle_count

        if candle_count < self._min_candles:
            issues.append(DataQualityIssue.INSUFFICIENT_CANDLES)

        if df is None or candle_count == 0:
            return DataQualityReport(
                is_sufficient=False,
                candle_count=candle_count,
                required_count=self._min_candles,
                issues=issues or [DataQualityIssue.INSUFFICIENT_CANDLES],
                detail=detail,
            )

        # --- Staleness: how old is the last candle relative to "now"? ---
        last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        staleness_sec = (now - last_ts).total_seconds()
        detail["staleness_sec"] = round(staleness_sec, 1)
        if staleness_sec > timeframe_sec * self._max_staleness_multiple:
            issues.append(DataQualityIssue.STALE_DATA)

        # --- Gaps: any missing candles in the recent window? ---
        # OTC pair history naturally has irregular spacing (market closures,
        # weekend gaps, low-liquidity periods). Only the most recent candles
        # need to be gap-free — old historical gaps don't affect live signals.
        # The merged DataFrame (old accumulated server_candles + new fast batch)
        # has a large boundary gap between the two sources; filter it out by
        # keeping only candles within a generous time band of the newest candle.
        if len(df) >= self._min_candles:
            newest_ts = pd.Timestamp(df["timestamp"].iloc[-1])
            if newest_ts.tzinfo is None:
                newest_ts = newest_ts.tz_localize("UTC")
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            # generous window: 90 min for 60s candles, wider for longer TFs
            lookback = timeframe_sec * 100
            cutoff = pd.Timestamp(newest_ts.timestamp() - lookback, tz="UTC")
            recent_df = df[df["timestamp"] >= cutoff]
            # fallback to tail(min_candles) if window is too small
            if len(recent_df) < self._min_candles:
                recent_df = df.tail(self._min_candles)
            gap_df = recent_df
        else:
            gap_df = df.tail(self._min_candles)
        if len(gap_df) >= 3:
            deltas = gap_df["timestamp"].diff().dt.total_seconds().dropna()
            max_gap = float(deltas.max()) if not deltas.empty else 0.0
            detail["max_gap_sec"] = round(max_gap, 1)
            detail["gap_window"] = f"{len(gap_df)} candles (lookback={lookback}s)"
            if max_gap > timeframe_sec * self._max_gap_multiple:
                issues.append(DataQualityIssue.TIMESTAMP_GAPS)

        # --- Flatness: is the recent window suspiciously dead? ---
        window_size = max(FLAT_WINDOW_MIN, min(candle_count, FLAT_WINDOW_MAX))
        recent = df.tail(window_size)
        candle_range = (recent["high"] - recent["low"]).abs()
        flat_mask = candle_range <= (recent["close"] * FLAT_RANGE_PCT)
        flat_fraction = float(flat_mask.mean()) if len(flat_mask) else 0.0
        detail["flat_fraction"] = round(flat_fraction, 3)
        if flat_fraction >= FLAT_CANDLE_FRACTION:
            issues.append(DataQualityIssue.FLAT_CANDLES)

        # --- Liquidity: were candles built from enough real ticks? ---
        if tick_counts:
            recent_ticks = tick_counts[-min(len(tick_counts), TICK_LOOKBACK_CANDLES):]
            avg_ticks = sum(recent_ticks) / len(recent_ticks) if recent_ticks else 0.0
            detail["avg_ticks_per_candle"] = round(avg_ticks, 1)
            if avg_ticks < self._min_ticks_per_candle:
                issues.append(DataQualityIssue.LOW_TICK_LIQUIDITY)

        return DataQualityReport(
            is_sufficient=len(issues) == 0,
            candle_count=candle_count,
            required_count=self._min_candles,
            issues=issues,
            detail=detail,
        )


# Backwards-compatible helper: thin wrapper producing a bool + short text,
# so existing callers that do `if not enough_data: ...` keep working.
def is_data_sufficient(
    df: pd.DataFrame | None,
    timeframe_sec: int,
    tick_counts: list[int] | None = None,
) -> tuple[bool, str]:
    """Return (is_sufficient, human_readable_reason).

    Kept for backwards compatibility with any code that currently does::

        if not is_data_sufficient(df, tf):
            await query.edit_message_text("Not enough data yet")
            return
    """
    gate = DataSufficiencyGate()
    report = gate.evaluate(df, timeframe_sec, tick_counts=tick_counts)
    return report.is_sufficient, report.summary()
