"""
Multi-timeframe confirmation — computes the same regime + trend read on a
higher timeframe and gates or downgrades confidence on disagreement.

Integration point: in _handle_quick_duration, after generate_signal::

    htf = HigherTimeframeConfirmation()
    htf_df = await collector.get_candles(symbol)  # already has higher timeframe?
    # or use a separate higher-timeframe stream
    htf_result = htf.check(htf_df, signal.direction, signal.confidence)
    if htf_result.disagreement != HTFDisagreement.NONE:
        confidence = htf_result.adjusted_confidence
        reasoning.append(htf_result.note)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class HTFDisagreement(str, Enum):
    NONE = "none"  # agreeing / no HTF data
    TREND_REVERSAL = "trend_reversal"  # HTF going the other way
    WEAK_TREND = "weak_trend"  # HTF trend inconclusive


@dataclass(frozen=True)
class HTFConfirmationResult:
    disagreement: HTFDisagreement
    adjusted_confidence: float  # confidence after HTF adjustment (same or lower)
    note: str  # human-readable explanation for the signal message


# How much to downgrade confidence when HTF disagrees.
HTF_DISAGREEMENT_PENALTY: float = 0.20  # subtract from confidence

# What counts as "agreeing" — HTF close must be on the same side of its
# own recent average as the signal's direction predicts.
HTF_AGREEMENT_WINDOW: int = 5  # bars to compute HTF recent average

# If HTF EMA is flatter than this (normalized), treat as no clear trend.
HTF_FLAT_THRESHOLD: float = 0.0003


class HigherTimeframeConfirmation:
    """Checks agreement with a higher timeframe's trend direction."""

    def __init__(
        self,
        disagreement_penalty: float = HTF_DISAGREEMENT_PENALTY,
        agreement_window: int = HTF_AGREEMENT_WINDOW,
        flat_threshold: float = HTF_FLAT_THRESHOLD,
    ) -> None:
        self._penalty = disagreement_penalty
        self._window = agreement_window
        self._flat_threshold = flat_threshold

    def check(
        self,
        df_htf: pd.DataFrame | None,
        direction: str,
        confidence: float,
    ) -> HTFConfirmationResult:
        """Evaluate HTF agreement.

        Args:
            df_htf: higher-timeframe candle DataFrame (same format as main df).
                If None / too short, returns no-agreement (no penalty).
            direction: signal direction ("call" or "put").
            confidence: pre-HTF confidence in [0, 1].
        """
        if df_htf is None or len(df_htf) < self._window + 1:
            return HTFConfirmationResult(
                disagreement=HTFDisagreement.NONE,
                adjusted_confidence=confidence,
                note="Insufficient HTF data to confirm",
            )

        last = df_htf.iloc[-1]
        prev_avg = df_htf["close"].iloc[-self._window - 1:-1].mean()
        htf_diff = last["close"] - prev_avg
        htf_flat = abs(htf_diff) / last["close"] < self._flat_threshold

        if htf_flat:
            return HTFConfirmationResult(
                disagreement=HTFDisagreement.WEAK_TREND,
                adjusted_confidence=confidence,
                note="HTF trend inconclusive (flat)",
            )

        htf_bearish = htf_diff < 0
        htf_bullish = htf_diff > 0

        agrees = (direction == "call" and htf_bullish) or (direction == "put" and htf_bearish)

        if agrees:
            return HTFConfirmationResult(
                disagreement=HTFDisagreement.NONE,
                adjusted_confidence=confidence,
                note=f"HTF agrees ({'bullish' if htf_bullish else 'bearish'})",
            )

        downgraded = max(0.0, confidence - self._penalty)
        return HTFConfirmationResult(
            disagreement=HTFDisagreement.TREND_REVERSAL,
            adjusted_confidence=downgraded,
            note=f"HTF disagrees ({'bearish' if htf_bearish else 'bullish'}) — confidence reduced",
        )
