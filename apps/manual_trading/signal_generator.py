"""Rule-based signal generation from technical indicators.

Scoring system:
- Each indicator votes CALL or PUT with a weight, scaled by market regime
- Final direction = majority vote weighted by strength
- Confidence = agreement ratio among indicators, calibrated by regime

Indicators used:
- RSI (oversold/overbought) — mean-reversion group
- MACD (histogram direction + crossover) — trend/momentum group
- EMA cross (fast vs slow) — trend/momentum group
- Bollinger Band position (%b) — mean-reversion group
- Stochastic (K/D crossover) — mean-reversion group
- Price momentum (ROC) — trend/momentum group
- ATR volatility filter — gate, not a vote

Market regime (via RegimeClassifier):
- ADX + Bollinger bandwidth percentile classify trending/ranging/undefined
- Trend group (MACD, EMA, ROC) weights scaled by trend_weight
- Mean-reversion group (RSI, BB, Stoch) weights scaled by reversion_weight
- In UNDEFINED regime, both groups vote on equal terms (weight=1.0)

Data sufficiency:
- A DataQualityReport can be passed in; if not sufficient, returns no signal.
- MIN_CANDLES is now 40 (MACD warm-up floor), not timeframe-shrunk.

No "always-signal" fallback: if no indicator produces a usable vote,
or the gate fails, the output is has_signal=False with no forced direction.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from apps.manual_trading.models import Signal

if TYPE_CHECKING:
    from apps.manual_trading.data_sufficiency_gate import DataQualityReport
    from apps.manual_trading.regime_classifier import RegimeReading

logger = logging.getLogger(__name__)

# Minimum candles needed for signal generation.
# MACD(12,26,9) slow EMA needs ~35-40 bars to be numerically stable;
# this is a floor — not timeframe-shrunk.
MIN_CANDLES = 40


def generate_signal(
    df_with_indicators: pd.DataFrame,
    regime: RegimeReading | None = None,
    quality_report: DataQualityReport | None = None,
) -> Signal:
    """Generate a trading signal from a DataFrame that already has indicators computed.

    Args:
        df_with_indicators: DataFrame with columns from TechnicalIndicators.compute()
        regime: optional RegimeReading from RegimeClassifier. If None, computed here.
        quality_report: optional DataQualityReport from DataSufficiencyGate.
            If not sufficient, returns no-signal immediately.

    Returns:
        Signal with direction, confidence, and reasoning bullets.
        has_signal=False when data is insufficient or no indicator votes.
    """
    if quality_report is not None and not quality_report.is_sufficient:
        return Signal(
            direction="call",
            confidence=0.5,
            reasoning=[f"Data insufficient: {quality_report.summary()}"],
            indicators={},
            has_signal=False,
        )

    if len(df_with_indicators) < MIN_CANDLES:
        return Signal(
            direction="call",
            confidence=0.5,
            reasoning=[f"Insufficient data ({len(df_with_indicators)}/{MIN_CANDLES} candles)"],
            indicators={},
            has_signal=False,
        )

    last = df_with_indicators.iloc[-1]
    prev = df_with_indicators.iloc[-2]

    if regime is None:
        from apps.manual_trading.regime_classifier import RegimeClassifier
        classifier = RegimeClassifier()
        regime = classifier.classify(df_with_indicators)

    votes: list[tuple[str, float]] = []
    reasoning: list[str] = []
    indicator_snapshot: dict[str, float] = {}

    # ATR volatility gate — suppress when ATR% is in bottom or top decile
    # relative to this symbol's recent history (last 20 candles).
    atr_pct = _safe(last, "atr_pct")
    indicator_snapshot["atr_pct"] = atr_pct
    atr_deciles = _compute_atr_deciles(df_with_indicators, window=20)
    if not np.isnan(atr_pct) and atr_deciles is not None:
        low_decile, high_decile = atr_deciles
        if atr_pct <= low_decile:
            reasoning.append(f"ATR% {atr_pct:.4f} in bottom decile ({low_decile:.4f}) — dead market, suppressing signal")
            atr_gate_suppressed = True
        elif atr_pct >= high_decile:
            reasoning.append(f"ATR% {atr_pct:.4f} in top decile ({high_decile:.4f}) — spike environment, suppressing signal")
            atr_gate_suppressed = True
        else:
            atr_gate_suppressed = False
            reasoning.append(f"ATR% {atr_pct:.4f} within normal range")
    else:
        atr_gate_suppressed = False
        if not np.isnan(atr_pct):
            reasoning.append(f"ATR% {atr_pct:.4f} (insufficient history for decile check)")

    # --- RSI (mean-reversion group) ---
    rsi = _safe(last, "rsi")
    indicator_snapshot["rsi"] = rsi
    if not np.isnan(rsi):
        w = 0.8 * regime.reversion_weight
        if rsi < 30:
            votes.append(("call", w))
            reasoning.append(f"RSI {rsi:.0f} — oversold, bullish reversal likely (regime weight: {regime.reversion_weight:.2f})")
        elif rsi > 70:
            votes.append(("put", w))
            reasoning.append(f"RSI {rsi:.0f} — overbought, bearish reversal likely (regime weight: {regime.reversion_weight:.2f})")
        elif rsi < 40:
            votes.append(("call", 0.6 * regime.reversion_weight))
            reasoning.append(f"RSI {rsi:.0f} — leaning oversold")
        elif rsi > 60:
            votes.append(("put", 0.6 * regime.reversion_weight))
            reasoning.append(f"RSI {rsi:.0f} — leaning overbought")
        else:
            votes.append(("call", 0.5 * regime.reversion_weight))
            reasoning.append(f"RSI {rsi:.0f} — neutral zone")

    # --- MACD (trend/momentum group) ---
    macd_hist = _safe(last, "macd_hist")
    prev_macd_hist = _safe(prev, "macd_hist")
    indicator_snapshot["macd_hist"] = macd_hist
    if not np.isnan(macd_hist) and not np.isnan(prev_macd_hist):
        w = 0.85 * regime.trend_weight
        if macd_hist > 0 and prev_macd_hist <= 0:
            votes.append(("call", w))
            reasoning.append(f"MACD bullish crossover — momentum shifting up (regime weight: {regime.trend_weight:.2f})")
        elif macd_hist < 0 and prev_macd_hist >= 0:
            votes.append(("put", w))
            reasoning.append(f"MACD bearish crossover — momentum shifting down (regime weight: {regime.trend_weight:.2f})")
        elif macd_hist > 0:
            votes.append(("call", 0.6 * regime.trend_weight))
            reasoning.append("MACD histogram positive — bullish momentum")
        else:
            votes.append(("put", 0.6 * regime.trend_weight))
            reasoning.append("MACD histogram negative — bearish momentum")

    # --- EMA Cross (trend/momentum group) ---
    ema_cross = _safe(last, "ema_cross")
    indicator_snapshot["ema_cross"] = ema_cross
    if not np.isnan(ema_cross):
        w = 0.7 * regime.trend_weight
        if ema_cross > 0.0005:
            votes.append(("call", w))
            reasoning.append(f"Fast EMA above slow EMA — uptrend (regime weight: {regime.trend_weight:.2f})")
        elif ema_cross < -0.0005:
            votes.append(("put", w))
            reasoning.append(f"Fast EMA below slow EMA — downtrend (regime weight: {regime.trend_weight:.2f})")
        else:
            reasoning.append("EMAs intertwined — no clear trend")

    # --- Bollinger Band %b (mean-reversion group) ---
    bb_pct = _safe(last, "bb_pct")
    indicator_snapshot["bb_pct"] = bb_pct
    if not np.isnan(bb_pct):
        w = 0.75 * regime.reversion_weight
        if bb_pct < 0.05:
            votes.append(("call", w))
            reasoning.append(f"Price at lower Bollinger band ({bb_pct:.2f}) — bounce expected (regime weight: {regime.reversion_weight:.2f})")
        elif bb_pct > 0.95:
            votes.append(("put", w))
            reasoning.append(f"Price at upper Bollinger band ({bb_pct:.2f}) — pullback expected (regime weight: {regime.reversion_weight:.2f})")
        elif bb_pct < 0.2:
            votes.append(("call", 0.6 * regime.reversion_weight))
            reasoning.append(f"Price near lower band ({bb_pct:.2f})")
        elif bb_pct > 0.8:
            votes.append(("put", 0.6 * regime.reversion_weight))
            reasoning.append(f"Price near upper band ({bb_pct:.2f})")

    # --- Stochastic (mean-reversion group) ---
    stoch_k = _safe(last, "stoch_k")
    stoch_d = _safe(last, "stoch_d")
    prev_stoch_k = _safe(prev, "stoch_k")
    prev_stoch_d = _safe(prev, "stoch_d")
    indicator_snapshot["stoch_k"] = stoch_k
    indicator_snapshot["stoch_d"] = stoch_d
    if not any(np.isnan(x) for x in [stoch_k, stoch_d, prev_stoch_k, prev_stoch_d]):
        w = 0.8 * regime.reversion_weight
        if stoch_k > stoch_d and prev_stoch_k <= prev_stoch_d and stoch_k < 30:
            votes.append(("call", w))
            reasoning.append(f"Stochastic bullish crossover in oversold zone (regime weight: {regime.reversion_weight:.2f})")
        elif stoch_k < stoch_d and prev_stoch_k >= prev_stoch_d and stoch_k > 70:
            votes.append(("put", w))
            reasoning.append(f"Stochastic bearish crossover in overbought zone (regime weight: {regime.reversion_weight:.2f})")
        elif stoch_k < 20:
            votes.append(("call", 0.6 * regime.reversion_weight))
            reasoning.append(f"Stochastic oversold ({stoch_k:.0f})")
        elif stoch_k > 80:
            votes.append(("put", 0.6 * regime.reversion_weight))
            reasoning.append(f"Stochastic overbought ({stoch_k:.0f})")

    # --- ROC (trend/momentum group) ---
    roc = _safe(last, "roc_5")
    indicator_snapshot["roc_5"] = roc
    if not np.isnan(roc):
        w = 0.6 * regime.trend_weight
        if roc > 0.002:
            votes.append(("call", w))
            reasoning.append(f"Price rising ({roc:.3%} over 5 bars) (regime weight: {regime.trend_weight:.2f})")
        elif roc < -0.002:
            votes.append(("put", w))
            reasoning.append(f"Price falling ({roc:.3%} over 5 bars) (regime weight: {regime.trend_weight:.2f})")

    # --- Build feature snapshot for calibration logging ---
    feature_snapshot = {
        "rsi": _nan_to_none(rsi),
        "macd_hist": _nan_to_none(macd_hist),
        "ema_cross": _nan_to_none(ema_cross),
        "bb_pct": _nan_to_none(bb_pct),
        "stoch_k": _nan_to_none(stoch_k),
        "stoch_d": _nan_to_none(stoch_d),
        "roc_5": _nan_to_none(roc),
        "atr_pct": _nan_to_none(atr_pct),
        "adx": _nan_to_none(last.get("adx")),
        "bb_width": _nan_to_none(last.get("bb_width")),
        "zscore": _nan_to_none(last.get("zscore")),
    }
    indicator_snapshot.update({k: v for k, v in feature_snapshot.items() if v is not None})

    # --- Compute final signal ---
    if not votes or atr_gate_suppressed:
        reasons = reasoning if reasoning else ["No clear signal from indicators"]
        if atr_gate_suppressed and not votes:
            reasons = [f"ATR volatility gate suppressed: {reasons[0] if reasons else 'no indicator votes'}"]
        return Signal(
            direction="call",
            confidence=0.5,
            reasoning=reasons,
            indicators=indicator_snapshot,
            has_signal=False,
        )

    call_weight = sum(w for d, w in votes if d == "call")
    put_weight = sum(w for d, w in votes if d == "put")
    total_weight = call_weight + put_weight

    if total_weight == 0:
        return Signal(
            direction="call",
            confidence=0.5,
            reasoning=["No weighted indicators produced a usable vote"],
            indicators=indicator_snapshot,
            has_signal=False,
        )

    if call_weight > put_weight:
        direction = "call"
        confidence = call_weight / total_weight
    elif put_weight > call_weight:
        direction = "put"
        confidence = put_weight / total_weight
    else:
        direction = "call"
        confidence = 0.5

    # Clamp confidence to [0.55, 0.95]
    confidence = max(0.55, min(0.95, confidence))

    # Add summary bullet
    summary = f"Bullish consensus from {len(votes)} indicators" if direction == "call" else f"Bearish consensus from {len(votes)} indicators"
    reasoning.insert(0, summary)
    reasoning.insert(1, f"Regime: {regime.summary()}")

    return Signal(
        direction=direction,
        confidence=round(confidence, 2),
        reasoning=reasoning[:6],
        indicators=_clean_nan(indicator_snapshot),
        has_signal=True,
    )


def _clean_nan(d: dict[str, float]) -> dict[str, float | None]:
    """Replace NaN/Inf float values with None for JSON safety.

    PostgreSQL JSON columns reject ``NaN`` tokens.  Converting them
    to ``None`` ensures ``json.dumps`` emits ``null`` instead.
    """
    return {
        k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v)
        for k, v in d.items()
    }


def _safe(row: pd.Series, col: str) -> float:
    """Safely extract a float from a row, returning NaN if missing."""
    val = row.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    return float(val)


def _nan_to_none(val: float) -> float | None:
    """Return None for NaN/Inf, otherwise the float."""
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def _compute_atr_deciles(df_with_indicators: pd.DataFrame, window: int = 20) -> tuple[float, float] | None:
    """Compute the 10th and 90th percentile of ATR% over a recent window.

    Returns (low_decile, high_decile) or None if not enough data.
    """
    if len(df_with_indicators) < window:
        return None
    recent = df_with_indicators["atr_pct"].tail(window).dropna()
    if len(recent) < window:
        return None
    return float(recent.quantile(0.10)), float(recent.quantile(0.90))
