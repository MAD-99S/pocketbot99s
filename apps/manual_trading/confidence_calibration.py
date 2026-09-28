"""Confidence calibration — reliability curve mapping raw vote ratios to actual win rates.

Turns "confidence" from "how much the indicators agreed" into "how often
signals like this actually won", using historical signal+outcome pairs
already logged in the predictions table.

The curve is REGIME-AWARE: separate buckets per market regime (TRENDING /
RANGING / UNDEFINED) so the bot can adapt when market conditions shift.
Global (regime=None) buckets act as a fallback when a regime has too few
samples.

Build order (suggested):
  1. Collect enough labeled predictions (50+ with win/loss results).
  2. Call CalibrationStore.build_curve() to bucket by raw confidence AND
     regime, computing empirical win rate per (bucket, regime) pair.
  3. Feed the curve back into signal output via ConfidenceCalibrator.

Integration:
  - After signal is generated but before sending to user:
      calibrator = ConfidenceCalibrator(curve)
      calibrated = calibrator.calibrate(signal.confidence, regime=regime_name)
  - Periodically (startup / scheduled): reload curve from DB.
  - Decay: recent outcomes (within DECAY_HALF_LIFE_DAYS) are weighted more
    heavily so the bot adapts to market shifts without discarding history.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from apps.manual_trading.database import PredictionStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket geometry
# ---------------------------------------------------------------------------

BUCKET_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
BUCKET_LABELS = [
    "0.50-0.60",
    "0.60-0.70",
    "0.70-0.80",
    "0.80-0.90",
    "0.90-1.00",
]

# Default curve used when no labeled data is available yet.
# Uniform 50% — every confidence maps to coin-flip, which is honest.
DEFAULT_CURVE: dict[float, float] = {
    0.55: 0.50, 0.65: 0.50, 0.75: 0.50, 0.85: 0.50, 0.95: 0.50,
}

# Minimum labeled samples in a bucket before its calibrated value is used.
# Below this, fall back to the global bucket for that raw range, or to raw.
MIN_SAMPLES_PER_BUCKET: int = 8

# Minimum labeled samples in a regime before per-regime buckets are used.
# Below this, fall back to global buckets for that regime's signals.
MIN_SAMPLES_PER_REGIME: int = 15

# Decay: outcomes older than this many days get half weight.  Set to 0 to
# disable decay (all outcomes weighted equally).  30 days is a reasonable
# default — it lets the bot adapt to market shifts over ~1 month while
# retaining enough history to be statistically stable.
DECAY_HALF_LIFE_DAYS: int = 30

# Never display a calibrated confidence below this floor.  Even if the data
# says a raw 0.95 signal only wins 55% of the time, we don't tell the user
# "55% confidence" — we clamp at CALIBRATED_CONFIDENCE_FLOOR.  This keeps
# the displayed confidence meaningful and avoids teaching the user to ignore
# low numbers.  The raw floor (CONFIDENCE_FLOOR in signal_generator) governs
# whether to signal at all; this governs what to display.
CALIBRATED_CONFIDENCE_FLOOR: float = 0.55


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationBucket:
    """One bucket of the reliability curve.

    ``regime`` is optional: when set, this bucket applies only to signals
    generated in that market regime (TRENDING / RANGING / UNDEFINED).
    When ``None``, the bucket is global (all regimes combined).
    """

    label: str
    raw_min: float
    raw_max: float
    sample_count: int
    win_count: int
    empirical_win_rate: float
    calibrated_value: float
    regime: str | None = None  # None = global; otherwise regime name


@dataclass
class CalibrationCurve:
    """A regime-aware reliability curve: raw confidence → calibrated win probability.

    Buckets are organised in two tiers:
      1. Per-regime buckets: one set per regime (TRENDING, RANGING, UNDEFINED),
         each with 5 confidence buckets.  Used when the regime has enough samples.
      2. Global buckets: 5 confidence buckets across all regimes.  Fallback when
         a regime has too few samples, or when the signal's regime is unknown.
    """

    global_buckets: list[CalibrationBucket] = field(default_factory=list)
    regime_buckets: dict[str, list[CalibrationBucket]] = field(
        default_factory=dict
    )  # regime_name -> [5 buckets]
    created_at: str = ""

    @property
    def is_usable(self) -> bool:
        """True when we have enough samples for a meaningful curve."""
        return sum(b.sample_count for b in self.global_buckets) >= MIN_SAMPLES_PER_BUCKET

    def calibrate(
        self, raw_confidence: float, regime: str | None = None
    ) -> float:
        """Map a raw confidence through the curve.

        Uses per-regime buckets when available and the regime has enough
        samples; falls back to global buckets otherwise.  Falls back to
        raw_confidence when the curve has too few samples or the value is
        outside the curve range.
        """
        if not self.is_usable:
            return raw_confidence
        if raw_confidence <= 0.50:
            return raw_confidence

        # Try per-regime bucket first (if we have a regime and it's usable).
        if regime is not None and regime in self.regime_buckets:
            rbuckets = self.regime_buckets[regime]
            if sum(b.sample_count for b in rbuckets) >= MIN_SAMPLES_PER_REGIME:
                for b in rbuckets:
                    if b.raw_min <= raw_confidence < b.raw_max:
                        return b.calibrated_value
                # Edge: raw_confidence in top partial bucket.
                top = rbuckets[-1]
                if raw_confidence >= top.raw_min:
                    return top.calibrated_value

        # Fall back to global bucket.
        for b in self.global_buckets:
            if b.raw_min <= raw_confidence < b.raw_max:
                return max(CALIBRATED_CONFIDENCE_FLOOR, b.calibrated_value)
        if self.global_buckets:
            top = self.global_buckets[-1]
            if raw_confidence >= top.raw_min:
                return max(CALIBRATED_CONFIDENCE_FLOOR, top.calibrated_value)
        return raw_confidence


# ---------------------------------------------------------------------------
# Building the curve
# ---------------------------------------------------------------------------

class CalibrationStore:
    """Loads / rebuilds a CalibrationCurve from labeled predictions."""

    def __init__(self, prediction_store: PredictionStore) -> None:
        self._store = prediction_store

    async def build_curve(self) -> CalibrationCurve:
        """Read labeled predictions, bucket by confidence AND regime.

        Returns a CalibrationCurve.  When too few samples exist, returns
        a curve with the DEFAULT_CURVE baked in and is_usable=False.
        """
        rows = await self._store.get_labeled_for_calibration()
        if not rows:
            logger.warning("calibration_no_labeled_predictions")
            return CalibrationCurve(
                global_buckets=[
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

        # Extract regime from feature_snapshot (if present).
        if "feature_snapshot" in df.columns:
            df["regime"] = df["feature_snapshot"].apply(_extract_regime_from_snapshot)
        else:
            df["regime"] = None

        # Apply time-decay weights.
        df["weight"] = _compute_decay_weights(
            df.get("created_at"),
            half_life_days=DECAY_HALF_LIFE_DAYS,
        )

        # Bucket by confidence.
        df["bucket_idx"] = pd.cut(
            df["confidence"],
            bins=BUCKET_EDGES,
            labels=BUCKET_LABELS,
            right=False,
            include_lowest=True,
        )

        # --- Global buckets (all regimes combined) ---
        global_buckets: list[CalibrationBucket] = []
        for label, lo, hi in zip(BUCKET_LABELS, BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
            chunk = df[df["bucket_idx"] == label]
            n = len(chunk)
            wins = int((chunk["result"] == "win").sum()) if n else 0
            weighted_wins = float((chunk["result"] == "win").mul(chunk["weight"]).sum()) if n else 0.0
            weighted_total = float(chunk["weight"].sum()) if n else 0.0
            win_rate = weighted_wins / weighted_total if weighted_total > 0 else (wins / n if n else 0.5)
            global_buckets.append(
                CalibrationBucket(
                    label=label,
                    raw_min=lo,
                    raw_max=hi,
                    sample_count=n,
                    win_count=wins,
                    empirical_win_rate=win_rate,
                    calibrated_value=win_rate,
                    regime=None,
                )
            )

        # --- Per-regime buckets ---
        regime_buckets: dict[str, list[CalibrationBucket]] = {}
        for regime in ("TRENDING", "RANGING", "UNDEFINED"):
            rdf = df[df["regime"] == regime]
            if rdf.empty:
                continue
            rbuckets = []
            for label, lo, hi in zip(BUCKET_LABELS, BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
                chunk = rdf[rdf["bucket_idx"] == label]
                n = len(chunk)
                wins = int((chunk["result"] == "win").sum()) if n else 0
                weighted_wins = float((chunk["result"] == "win").mul(chunk["weight"]).sum()) if n else 0.0
                weighted_total = float(chunk["weight"].sum()) if n else 0.0
                win_rate = weighted_wins / weighted_total if weighted_total > 0 else (wins / n if n else 0.5)
                rbuckets.append(
                    CalibrationBucket(
                        label=label,
                        raw_min=lo,
                        raw_max=hi,
                        sample_count=n,
                        win_count=wins,
                        empirical_win_rate=win_rate,
                        calibrated_value=win_rate,
                        regime=regime,
                    )
                )
            regime_buckets[regime] = rbuckets

        # --- Monotonicity enforcement (per tier) ---
        global_buckets = _enforce_monotonicity(global_buckets)
        for regime, rbuckets in regime_buckets.items():
            regime_buckets[regime] = _enforce_monotonicity(rbuckets)

        curve = CalibrationCurve(
            global_buckets=global_buckets,
            regime_buckets=regime_buckets,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _log_curve(curve)
        return curve


def _extract_regime_from_snapshot(snapshot: object) -> str | None:
    """Pull the regime string from a feature_snapshot dict, if present."""
    if not isinstance(snapshot, dict):
        return None
    regime = snapshot.get("regime")
    if isinstance(regime, str) and regime in ("TRENDING", "RANGING", "UNDEFINED"):
        return regime
    return None


def _compute_decay_weights(
    timestamps: pd.Series | None,
    half_life_days: int,
) -> pd.Series:
    """Compute time-decay weights for labeled predictions.

    Recent outcomes get higher weight so the bot adapts to market shifts.
    When half_life_days is 0 or timestamps is empty, all weights are 1.0.
    """
    if half_life_days <= 0 or timestamps is None or timestamps.empty:
        return pd.Series([1.0] * len(timestamps)) if timestamps is not None else pd.Series(dtype=float)

    now = pd.Timestamp.now(tz=timezone.utc)
    ages_days = (now - pd.to_datetime(timestamps)).dt.total_seconds() / 86400.0
    weights = np.exp(-np.log(2) * ages_days / half_life_days)
    return pd.Series(weights, index=timestamps.index)


def _enforce_monotonicity(buckets: list[CalibrationBucket]) -> list[CalibrationBucket]:
    """Ensure calibrated values increase (non-strictly) with confidence.

    1. Top-down pass: each bucket must be >= its predecessor.
    2. Bottom-up pass: each bucket must be <= next bucket + epsilon.
       This prevents the bottom-up smoothing from collapsing the curve.
    """
    if len(buckets) < 2:
        return buckets

    result = list(buckets)

    # Top-down: pull up.
    for i in range(1, len(result)):
        if result[i].calibrated_value < result[i - 1].calibrated_value:
            result[i].calibrated_value = result[i - 1].calibrated_value

    # Bottom-up: cap.  Previously this used +0.05 which could collapse a
    # well-differentiated curve (e.g. 0.5/0.5/0.5/0.5/0.71 → all 0.5).
    # Now we only cap when the gap is unreasonably large (> 0.20), which
    # catches genuine inversions while preserving real signal differences.
    for i in range(len(result) - 2, -1, -1):
        gap = result[i].calibrated_value - result[i + 1].calibrated_value
        if gap > 0.20:
            result[i].calibrated_value = result[i + 1].calibrated_value + 0.20

    return result


def _log_curve(curve: CalibrationCurve) -> None:
    """Log the built curve at INFO level."""
    total = sum(b.sample_count for b in curve.global_buckets)
    logger.info(
        "calibration_curve_built",
        extra={
            "total_samples": total,
            "usable": curve.is_usable,
            "global_buckets": [
                {
                    "label": b.label,
                    "n": b.sample_count,
                    "win_rate": round(b.empirical_win_rate, 3),
                    "calibrated": round(b.calibrated_value, 3),
                }
                for b in curve.global_buckets
            ],
            "regime_buckets": {
                regime: [
                    {
                        "label": b.label,
                        "n": b.sample_count,
                        "win_rate": round(b.empirical_win_rate, 3),
                        "calibrated": round(b.calibrated_value, 3),
                    }
                    for b in buckets
                ]
                for regime, buckets in curve.regime_buckets.items()
            },
        },
    )


class ConfidenceCalibrator:
    """Wraps a CalibrationCurve and applies it to signal confidence values.

    Usage:
        curve = await CalibrationStore(store).build_curve()
        calibrator = ConfidenceCalibrator(curve)
        display_conf = calibrator.calibrate(
            signal.confidence, regime=signal_regime_value
        )
    """

    def __init__(self, curve: CalibrationCurve) -> None:
        self._curve = curve

    def calibrate(
        self, raw_confidence: float, regime: str | None = None
    ) -> float:
        """Return calibrated confidence for display.

        Delegates to ``CalibrationCurve.calibrate()`` and clamps the result
        to [CALIBRATED_CONFIDENCE_FLOOR, 0.95].
        """
        calibrated = self._curve.calibrate(raw_confidence, regime=regime)
        return round(
            max(CALIBRATED_CONFIDENCE_FLOOR, min(0.95, calibrated)), 2
        )

    @property
    def curve(self) -> CalibrationCurve:
        return self._curve

    @property
    def is_calibrated(self) -> bool:
        return self._curve.is_usable
