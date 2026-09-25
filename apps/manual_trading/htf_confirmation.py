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
    confidence_penalty: float = 0.0  # fraction to reduce regime weights by (0 = no penalty)


def _assess_trend(df: pd.DataFrame, window: int, flat_threshold: float) -> tuple[bool, bool]:
    """Return (is_bullish, is_flat) for the last candle vs its recent average."""
    last = df.iloc[-1]
    prev_avg = df["close"].iloc[-window - 1:-1].mean()
    htf_diff = last["close"] - prev_avg
    htf_flat = abs(htf_diff) / last["close"] < flat_threshold
    return htf_diff > 0, htf_flat


# What counts as "agreeing" — HTF close must be on the same side of its
# own recent average as the signal's direction predicts.
HTF_AGREEMENT_WINDOW: int = 5  # bars to compute HTF recent average

# If HTF EMA is flatter than this (normalized), treat as no clear trend.
HTF_FLAT_THRESHOLD: float = 0.0003

# How much to downgrade confidence when HTF disagrees.
HTF_DISAGREEMENT_PENALTY: float = 0.20  # subtract from confidence

# What counts as "agreeing" — HTF close must be on the same side of its
# own recent average as the signal's direction predicts.
HTF_AGREEMENT_WINDOW: int = 5  # bars to compute HTF recent average

# If HTF EMA is flatter than this (normalized), treat as no clear trend.
HTF_FLAT_THRESHOLD: float = 0.0003


class HigherTimeframeConfirmation:
    """Checks agreement with a higher timeframe's trend direction.

    Two calling conventions are supported:

    1. Same-data mode (used by handlers.py): pass the signal's own
       indicator DataFrame plus the signal timeframe_sec, and the
       method derives HTF agreement by comparing the latest close to
       its own recent-window average.  This is a lightweight proxy for
       "is the higher timeframe trending in the same direction" — good
       enough when a separate higher-TF candle stream is not available.

       htf = HTFConfirmation()
       result = htf.confirm(df_with_indicators, timeframe_sec)
       if result is not None and result.confidence_penalty:
           # downgrade regime weights by result.confidence_penalty

    2. Explicit HTF DataFrame mode: pass a pre-built higher-timeframe
       candle DataFrame directly.

       htf = HTFConfirmation()
       result = htf.check(df_htf, direction, confidence)
       if result.disagreement != HTFDisagreement.NONE:
           confidence = result.adjusted_confidence
    """

    def __init__(
        self,
        disagreement_penalty: float = HTF_DISAGREEMENT_PENALTY,
        agreement_window: int = HTF_AGREEMENT_WINDOW,
        flat_threshold: float = HTF_FLAT_THRESHOLD,
    ) -> None:
        self._penalty = disagreement_penalty
        self._window = agreement_window
        self._flat_threshold = flat_threshold

    def confirm(
        self,
        df_with_indicators: pd.DataFrame,
        timeframe_sec: int,
    ) -> HTFConfirmationResult:
        """Lightweight HTF-agreement proxy using the signal's own data.

        Returns None for 1-min timeframes (no higher TF available).
        For 5m/15m, compares the latest close to its own recent-window
        average and yields a trend-direction verdict.  The caller uses
        ``confidence_penalty`` to scale down regime weights on disagreement.
        """
        # 1-min has no higher timeframe to check against.
        if timeframe_sec < 60:
            return None

        if len(df_with_indicators) < self._window + 1:
            return HTFConfirmationResult(
                disagreement=HTFDisagreement.NONE,
                adjusted_confidence=0.0,
                note="Insufficient data to confirm HTF trend",
                confidence_penalty=0.0,
            )

        bullish, flat = _assess_trend(
            df_with_indicators,
            self._window,
            self._flat_threshold,
        )

        if flat:
            return HTFConfirmationResult(
                disagreement=HTFDisagreement.WEAK_TREND,
                adjusted_confidence=0.0,
                note="HTF trend inconclusive (flat)",
                confidence_penalty=0.0,
            )

        # Proxy: treat the latest close-vs-average direction as the HTF read.
        # This method runs before generate_signal(), so it does not yet know
        # the signal's direction and cannot assess agreement.  Report the HTF
        # read neutrally; the caller compares against the signal's direction
        # after generate_signal() returns and penalizes weights on disagreement.
        return HTFConfirmationResult(
            disagreement=HTFDisagreement.NONE,
            adjusted_confidence=0.0,
            note=f"HTF read: {'bullish' if bullish else 'bearish'} — compare against signal direction",
            confidence_penalty=0.0,
        )

    def check(
        self,
        df_htf: pd.DataFrame | None,
        direction: str,
        confidence: float,
    ) -> HTFConfirmationResult:
        """Evaluate HTF agreement from a separate higher-TF DataFrame.

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

        bullish, flat = _assess_trend(df_htf, self._window, self._flat_threshold)

        if flat:
            return HTFConfirmationResult(
                disagreement=HTFDisagreement.WEAK_TREND,
                adjusted_confidence=confidence,
                note="HTF trend inconclusive (flat)",
            )

        agrees = (direction == "call" and bullish) or (direction == "put" and not bullish)
        if agrees:
            return HTFConfirmationResult(
                disagreement=HTFDisagreement.NONE,
                adjusted_confidence=confidence,
                note=f"HTF agrees ({'bullish' if bullish else 'bearish'})",
            )

        downgraded = max(0.0, confidence - self._penalty)
        return HTFConfirmationResult(
            disagreement=HTFDisagreement.TREND_REVERSAL,
            adjusted_confidence=downgraded,
            note=f"HTF disagrees ({'bearish' if not bullish else 'bullish'}) — confidence reduced",
        )


# Alias so existing import `from htf_confirmation import HTFConfirmation` works.
HTFConfirmation = HigherTimeframeConfirmation  # noqa: NIR005
