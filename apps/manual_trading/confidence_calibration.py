"""Confidence calibration — reliability curve mapping raw vote ratios to actual win rates.

Turns "confidence" from "how much the indicators agreed" into "how often
signals like this actually won", using historical signal+outcome pairs
already logged in the predictions table.

Build order (suggested):
  1. Collect enough labeled predictions (e.g. 50+ with win/loss results).
  2. Call CalibrationStore.build_curve() to bucket by raw confidence and
     compute empirical win rate per bucket.
  3. Feed the curve back into signal output via ConfidenceCalibrator.

Integration:
  - After signal is generated but before sending to user:
      calibrator = ConfidenceCalibrator(curve)
      calibrated = calibrator.calibrate(signal.confidence)

  - Periodically (cron / startup): reload curve from DB.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from apps.manual_trading.database import PredictionStore

logger = logging.getLogger(__name__)

# Default curve used when no labeled data is available yet.
# Uniform 50% — every confidence maps to coin-flip, which is honest.
DEFAULT_CURVE: dict[float, float] = {0.55: 0.50, 0.65: 0.50, 0.75: 0.50, 0.85: 0.50, 0.95: 0.50}

BUCKET_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
BUCKET_LABELS = [
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
]


@dataclass
class CalibrationBucket:
    """One bucket of the reliability curve."""

    label: str
    raw_min: float
    raw_max: float
    sample_count: int
    win_count: int
    empirical_win_rate: float
    calibrated_value: float  # mid-point of raw range, mapped through curve


@dataclass
class CalibrationCurve:
    """A reliability curve: raw confidence → calibrated win probability.

    Loaded from DB or built from labeled predictions.  Provides a
    monotonically-increasing mapping so higher raw confidence always
    maps to >= calibrated confidence.
    """

    buckets: list[CalibrationBucket] = field(default_factory=list)
    created_at: str = ""

    @property
    def is_usable(self) -> bool:
        """True when we have at least MIN_SAMPLES total labeled predictions."""
        total = sum(b.sample_count for b in self.buckets)
        return total >= MIN_SAMPLES

    def calibrate(self, raw_confidence: float) -> float:
        """Map a raw confidence through the curve.

        Falls back to raw_confidence when the curve has too few samples
        or the value is outside the curve range.
        """
        if not self.is_usable:
            return raw_confidence
        if raw_confidence <= 0.50:
            return raw_confidence
        # Find the bucket this raw_confidence falls into.
        for b in self.buckets:
            if b.raw_min <= raw_confidence < b.raw_max:
                return b.calibrated_value
        # Edge case: raw_confidence == 1.0 or in the top partial bucket.
        if self.buckets:
            top = self.buckets[-1]
            if raw_confidence >= top.raw_min:
                return top.calibrated_value
        return raw_confidence


MIN_SAMPLES = 30  # minimum labeled predictions before curve is used
DEFAULT_BUCKET_COUNT = 5


class CalibrationStore:
    """Loads / rebuilds a CalibrationCurve from labeled predictions."""

    def __init__(self, prediction_store: PredictionStore) -> None:
        self._store = prediction_store

    async def build_curve(self) -> CalibrationCurve:
        """Read labeled predictions, bucket by confidence, compute win rates.

        Returns a CalibrationCurve.  When too few samples exist, returns
        a curve with the DEFAULT_CURVE baked in and is_usable=False.
        """
        rows = await self._store.get_labeled_for_calibration()
        if not rows:
            logger.warning("calibration_no_labeled_predictions")
            return CalibrationCurve(
                buckets=[
                    CalibrationBucket(
                        label=label,
                        raw_min=lo,
                        raw_max=hi,
                        sample_count=0,
                        win_count=0,
                        empirical_win_rate=0.5,
                        calibrated_value=curve_val,
                    )
                    for (label, lo, hi), curve_val in zip(
                        zip(BUCKET_LABELS, BUCKET_EDGES[:-1], BUCKET_EDGES[1:]),
                        [DEFAULT_CURVE.get(v, 0.5) for v in [0.55, 0.65, 0.75, 0.85, 0.95]],
                    )
                ],
                created_at="empty",
            )

        df = pd.DataFrame(rows)
        df["confidence"] = df["confidence"].astype(float)
        df["result"] = df["result"].astype(str)

        # Bucket
        df["bucket_idx"] = pd.cut(
            df["confidence"],
            bins=BUCKET_EDGES,
            labels=BUCKET_LABELS,
            right=False,
            include_lowest=True,
        )

        buckets: list[CalibrationBucket] = []
        total_samples = 0
        for label, lo, hi in zip(BUCKET_LABELS, BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
            chunk = df[df["bucket_idx"] == label]
            n = len(chunk)
            wins = int((chunk["result"] == "win").sum()) if n else 0
            win_rate = wins / n if n else 0.5
            total_samples += n
            buckets.append(
                CalibrationBucket(
                    label=label,
                    raw_min=lo,
                    raw_max=hi,
                    sample_count=n,
                    win_count=wins,
                    empirical_win_rate=win_rate,
                    calibrated_value=win_rate,
                )
            )

        # Monotonicity enforcement: make sure higher raw buckets are >= lower ones.
        # Walk top-down and pull any bucket below its predecessor up.
        for i in range(1, len(buckets)):
            if buckets[i].calibrated_value < buckets[i - 1].calibrated_value:
                buckets[i].calibrated_value = buckets[i - 1].calibrated_value
        # Walk bottom-up to smooth: each bucket can't be above the next one + small epsilon.
        for i in range(len(buckets) - 2, -1, -1):
            if buckets[i].calibrated_value > buckets[i + 1].calibrated_value + 0.05:
                buckets[i].calibrated_value = buckets[i + 1].calibrated_value + 0.05

        curve = CalibrationCurve(
            buckets=buckets,
            created_at=pd.Timestamp.now().isoformat(),
        )
        logger.info(
            "calibration_curve_built",
            total_samples=total_samples,
            usable=curve.is_usable,
            buckets=[
                {
                    "label": b.label,
                    "n": b.sample_count,
                    "win_rate": round(b.empirical_win_rate, 3),
                    "calibrated": round(b.calibrated_value, 3),
                }
                for b in buckets
            ],
        )
        return curve


class ConfidenceCalibrator:
    """Wraps a CalibrationCurve and applies it to signal confidence values.

    Usage:
        curve = await calibration_store.build_curve()
        calibrator = ConfidenceCalibrator(curve)
        signal = generate_signal(df, regime, quality_report)
        if signal.has_signal:
            signal.confidence = calibrator.calibrate(signal.confidence)
    """

    def __init__(self, curve: CalibrationCurve) -> None:
        self._curve = curve

    def calibrate(self, raw_confidence: float) -> float:
        """Return calibrated confidence (0.50-0.95)."""
        calibrated = self._curve.calibrate(raw_confidence)
        # Clamp to [0.50, 0.95] — never claim > 95% certainty.
        return round(max(0.50, min(0.95, calibrated)), 2)

    @property
    def curve(self) -> CalibrationCurve:
        return self._curve

    @property
    def is_calibrated(self) -> bool:
        return self._curve.is_usable
