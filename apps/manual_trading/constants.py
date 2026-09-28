"""Trading constants — supported pairs and timeframes."""
from __future__ import annotations

POPULAR_PAIRS: list[str] = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "USDJPY_otc",
    "AUDUSD_otc",
    "USDCAD_otc",
    "EURGBP_otc",
    "EURJPY_otc",
    "GBPJPY_otc",
    "BTCUSD_otc",
    "ETHUSD_otc",
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
]

ASSET_DISPLAY_NAMES: dict[str, str] = {
    "EURUSD_otc": "EUR/USD (OTC)",
    "GBPUSD_otc": "GBP/USD (OTC)",
    "USDJPY_otc": "USD/JPY (OTC)",
    "AUDUSD_otc": "AUD/USD (OTC)",
    "USDCAD_otc": "USD/CAD (OTC)",
    "EURGBP_otc": "EUR/GBP (OTC)",
    "EURJPY_otc": "EUR/JPY (OTC)",
    "GBPJPY_otc": "GBP/JPY (OTC)",
    "BTCUSD_otc": "BTC/USD (OTC)",
    "ETHUSD_otc": "ETH/USD (OTC)",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
}

# Candle timeframes available for subscription
CANDLE_TIMEFRAMES: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
}

# Base number of candles to fetch for indicator computation.
# MACD(12,26,9) needs ~26 candles for stable slow EMA; with fewer,
# indicators degrade gracefully (NaN → skipped in signal scoring).
CANDLES_NEEDED = 30

# Minimum candles required per timeframe for signal generation.
# Derived from the slowest indicator's warm-up need (MACD(12,26,9) needs
# ~35-40 bars before its slow EMA is numerically stable; ADX/ATR need
# ~28-30 for Wilder smoothing to settle). The gate uses 40 candles as a
# floor, driven by indicator stability — NOT shrunk per-timeframe to hit
# an arbitrary "always respond fast" UX target.
MIN_CANDLES_BY_TIMEFRAME: dict[int, int] = {
    60: 40,   # 1m — MACD stable at 35-40
    300: 40,  # 5m — same indicator set, same warm-up
    900: 40,  # 15m — accumulate real history, surface "still warming up" to user
}


def min_candles_for_timeframe(timeframe_sec: int) -> int:
    """Return the minimum candles needed for a given timeframe."""
    return MIN_CANDLES_BY_TIMEFRAME.get(timeframe_sec, 40)


# --- Always-Signal Directional Confidence Strategy Constants ---
# DEPRECATED: this fallback path forced a directional signal even when data
# was thin, which is the opposite of what we want for prediction quality.
# The DataSufficiencyGate (data_sufficiency_gate.py) now returns "no signal"
# when data doesn't meet the bar. These constants are kept for reference but
# are no longer wired into the active signal path.

# Trend strength thresholds (normalized EMA cross magnitude = |ema_fast - ema_slow| / close).
# Strong trend gets high confidence bonus, mild trend gets moderate bonus.
TREND_STRONG_THRESHOLD: float = 0.002
TREND_MILD_THRESHOLD: float = 0.0005

# RSI boundaries for favorability scoring.
# Below 35 in downtrend = favorable (oversold bounce). Above 65 in uptrend = favorable.
RSI_FAVORABLE_LOW: float = 35.0
RSI_FAVORABLE_HIGH: float = 65.0

# ATR volatility penalty: reduce confidence when ATR% > multiplier * SMA(ATR%, window).
ATR_SPIKE_MULTIPLIER: float = 2.0
ATR_SMA_WINDOW: int = 10  # longest feasible with 16-30 candles

# Payout percentage filter for asset selection.
# Only assets with payout in [MIN_ASSET_PAYOUT_PCT, MAX_ASSET_PAYOUT_PCT] are shown.
MIN_ASSET_PAYOUT_PCT: float = 80.0
MAX_ASSET_PAYOUT_PCT: float = 92.0
