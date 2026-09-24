"""
Market regime classifier — trending vs. ranging vs. undefined.

Wired in after TechnicalIndicators.compute() and before generate_signal():
tells the scorer which indicator family to weight higher for the current bar.

Trend indicators (MACD, EMA-cross) and mean-reversion indicators (RSI,
Stochastic, Bollinger %b) are largely redundant with each other within a
family but disagree in predictive value across regimes — this is what lets
the scorer stop treating all 7 votes as independent and equally relevant
regardless of market conditions.

Uses ADX (trend strength) and Bollinger bandwidth percentile (squeeze vs.
expansion), both already computed by TechnicalIndicators — no new data
requirements, no extra candles needed.

Integration point: in signal_generator.generate_signal(), after computing
last/prev and before tallying votes::

    classifier = RegimeClassifier()
    reading = classifier.classify(df_with_indicators)
    ...
    votes.append(("call", 0.8 * reading.reversion_weight))   # RSI vote
    votes.append(("call", 0.85 * reading.trend_weight))      # MACD vote
    ...
    reasoning.append(f"Regime: {reading.summary()}")

i.e. multiply each indicator's existing weight by trend_weight or
reversion_weight depending on which family it belongs to, rather than
changing the per-indicator logic itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    UNDEFINED = "undefined"  # not enough signal to classify confidently


@dataclass(frozen=True)
class RegimeReading:
    regime: Regime
    adx: float
    bb_width: float
    bb_width_percentile: float
    trend_weight: float  # multiplier for the trend indicator group's votes
    reversion_weight: float  # multiplier for the mean-reversion group's votes
    neutral_weight: float = 1.0  # multiplier for indicators not in either group

    def summary(self) -> str:
        return (
            f"{self.regime.value} "
            f"(ADX={self.adx:.1f}, BB-width pctile={self.bb_width_percentile:.0%})"
        )


# ADX thresholds — Wilder's convention: <20 weak/no trend, >25 trending.
# The 18-25 band is deliberately treated as ambiguous (see UNDEFINED).
ADX_TRENDING_THRESHOLD: float = 25.0
ADX_RANGING_THRESHOLD: float = 18.0

# Bollinger bandwidth percentile (relative to its own recent history):
# above this = expanding/breakout conditions (favors trend following);
# below this = squeezed/consolidating (favors mean reversion).
BB_WIDTH_EXPANDING_PCTL: float = 0.6
BB_WIDTH_SQUEEZE_PCTL: float = 0.4
BB_WIDTH_LOOKBACK: int = 30

# How strongly a classified regime tilts vote weights between the two
# indicator families. 1.0 = no tilt. E.g. 1.6 means the favored family's
# votes count 1.6x and the other family's count 1/1.6x — a ~2.6x swing
# in relative influence, without ever fully silencing either family.
REGIME_TILT_STRENGTH: float = 1.6
STRONG_AGREEMENT_BONUS: float = 1.2  # extra tilt when ADX and BB-width agree


class RegimeClassifier:
    """Classifies the current bar's regime from ADX + Bollinger bandwidth."""

    def __init__(
        self,
        adx_trending_threshold: float = ADX_TRENDING_THRESHOLD,
        adx_ranging_threshold: float = ADX_RANGING_THRESHOLD,
        bb_width_lookback: int = BB_WIDTH_LOOKBACK,
        tilt_strength: float = REGIME_TILT_STRENGTH,
    ) -> None:
        self._adx_trending = adx_trending_threshold
        self._adx_ranging = adx_ranging_threshold
        self._bb_lookback = bb_width_lookback
        self._tilt = tilt_strength

    def classify(self, df_with_indicators: pd.DataFrame) -> RegimeReading:
        """Classify the regime of the *last* row in an indicator-enriched df.

        Args:
            df_with_indicators: output of `TechnicalIndicators.compute()`;
                must contain `adx` and `bb_width` columns.
        """
        if len(df_with_indicators) == 0:
            return RegimeReading(
                regime=Regime.UNDEFINED,
                adx=float("nan"),
                bb_width=float("nan"),
                bb_width_percentile=0.5,
                trend_weight=1.0,
                reversion_weight=1.0,
            )

        last = df_with_indicators.iloc[-1]
        adx = _safe_float(last.get("adx"))
        window = df_with_indicators["bb_width"].tail(self._bb_lookback)
        bb_width = _safe_float(last.get("bb_width"))
        bb_width_pctl = _percentile_rank(window, bb_width)
        regime, trend_w, reversion_w = self._decide(adx, bb_width_pctl)

        return RegimeReading(
            regime=regime,
            adx=adx if not np.isnan(adx) else 0.0,
            bb_width=bb_width if not np.isnan(bb_width) else 0.0,
            bb_width_percentile=bb_width_pctl,
            trend_weight=trend_w,
            reversion_weight=reversion_w,
        )

    def _decide(
        self, adx: float, bb_width_pctl: float
    ) -> tuple[Regime, float, float]:
        if np.isnan(adx):
            return Regime.UNDEFINED, 1.0, 1.0

        adx_trending = adx >= self._adx_trending
        adx_ranging = adx <= self._adx_ranging
        bb_expanding = bb_width_pctl >= BB_WIDTH_EXPANDING_PCTL
        bb_squeezed = bb_width_pctl <= BB_WIDTH_SQUEEZE_PCTL

        if adx_trending and bb_expanding:
            # Strong agreement between both signals — lean harder.
            tilt = self._tilt * STRONG_AGREEMENT_BONUS
            return Regime.TRENDING, tilt, 1.0 / tilt

        if adx_ranging and bb_squeezed:
            tilt = self._tilt * STRONG_AGREEMENT_BONUS
            return Regime.RANGING, 1.0 / tilt, tilt

        if adx_trending and not bb_squeezed:
            return Regime.TRENDING, self._tilt, 1.0 / self._tilt

        if adx_ranging:
            return Regime.RANGING, 1.0 / self._tilt, self._tilt

        # ADX sits in the 18-25 "no man's land", or ADX/BB-width disagree —
        # don't tilt the vote; let both families compete on equal terms.
        return Regime.UNDEFINED, 1.0, 1.0


def _safe_float(val: object) -> float:
    if val is None:
        return float("nan")
    try:
        result = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return result


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Fraction of series that is <= value. NaN-safe, defaults to 0.5."""
    clean = series.dropna()
    if clean.empty or np.isnan(value):
        return 0.5
    return float((clean <= value).mean())
